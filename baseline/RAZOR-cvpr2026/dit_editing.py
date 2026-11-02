"""RAZOR DiT port: first-CLIP ratio scoring and progressive layer editing.

The original demo is single-style SD3 and has inverted loss signs and a
constant presence metric. This implementation makes those intended operations
explicit; all departures are recorded in DIT_BENCHMARK_ADAPTATION.md.
"""
import copy
from pathlib import Path

from benchmark.common import file_digest, write_json
from benchmark.execution import runtime_device


def fit(checkpoint, settings, config):
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPImageProcessor
    from safetensors.torch import save_file
    from benchmark.dit_adapter import DiTAdapter

    torch.manual_seed(settings["seed"])
    device = runtime_device(settings["edit_device"])
    cfg = copy.deepcopy(config)
    cfg["adapter"] = {"kind": "original"}
    cfg["sampling"] = copy.deepcopy(settings["calibration_sampling"])
    sampler = DiTAdapter(cfg)
    pipe = sampler.pipe
    # Keep a float32 editing copy separate from the bf16 deployment pipeline
    # and its Accelerate offload hooks. Only final selected parameters are saved.
    encoder = type(pipe.text_encoder)(pipe.text_encoder.config).to(device=device, dtype=torch.float32)
    encoder.load_state_dict(pipe.text_encoder.state_dict())
    clip = CLIPModel.from_pretrained(settings["clip_model"], torch_dtype=torch.float32,
                                     local_files_only=cfg["sampling"]["local_files_only"]).to(device).eval()
    clip.requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(settings["clip_model"],
                                                   local_files_only=cfg["sampling"]["local_files_only"])

    # Calibration templates are method data, not held-out benchmark prompts.
    # Replace the target by its explicit anchor, equally often for every target.
    if settings["group"] == "style":
        templates = ["a landscape in {concept} style", "a portrait in {concept} style",
                     "a city street painted in {concept} style", "a still life in {concept} style"]
        label = lambda concept: f"an image in {concept} style"
    elif settings["group"] == "i2p":
        templates = ["a photo of a person, {concept}", "a person on a beach, {concept}",
                     "a portrait of a person, {concept}", "a person in a room, {concept}"]
        label = lambda concept: f"a person, {concept}"
    else:
        templates = ["a photo of {concept}", "a detailed illustration of {concept}",
                     "{concept} outdoors", "a close-up of {concept}"]
        label = lambda concept: f"an image of {concept}"
    templates = [templates[i % len(templates)] for i in range(settings["prompts_per_concept"])]
    forget_prompts = [template.format(concept=target) for target in settings["targets"] for template in templates]
    retain_prompts = [template.format(concept=anchor) for anchor in settings["anchors"] for template in templates]

    def text_features(prompts, model=encoder):
        tokens = pipe.tokenizer(prompts, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").to(device)
        result = model(**tokens)
        if hasattr(result, "text_embeds"):
            features = result.text_embeds
        else:
            # FLUX's first encoder is CLIPTextModel without a projection head.
            # The matching pretrained CLIP-L/14 projection supplies that map.
            features = clip.text_projection(result.pooler_output)
        return F.normalize(features.float(), dim=-1)

    def image_features(prompts):
        images = []
        for i, prompt in enumerate(prompts):
            images.extend(sampler.generate([{"prompt": prompt, "seed": settings["seed"] + i}], cfg["sampling"]))
        with torch.no_grad():
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
            return F.normalize(clip.visual_projection(clip.vision_model(pixels).pooler_output).float(), dim=-1)

    def forget_loss(features):
        text = text_features(forget_prompts)
        # AdamW minimizes: +matched cosine erases alignment; +mismatched
        # cosine suppresses accidental cross-image alignment without self-pairs.
        matched = (text * features).sum(-1).mean()
        mismatch = (text * features.roll(1, dims=0)).sum(-1).mean()
        return settings["forget_weight"] * matched + settings["mismatch_weight"] * mismatch

    # Use the frozen, matched CLIP verifier to distinguish target from anchor.
    # The original one-label argmax was always a hit and could never stop.
    with torch.no_grad():
        labels = [[label(target), label(anchor)] for target, anchor in zip(settings["targets"], settings["anchors"])]
        tokens = pipe.tokenizer([text for pair in labels for text in pair], padding="max_length",
                                max_length=77, truncation=True, return_tensors="pt").to(device)
        label_embeddings = F.normalize(clip.text_projection(clip.text_model(**tokens).pooler_output), dim=-1)
        label_embeddings = label_embeddings.reshape(len(labels), 2, -1)

    modules = {name: module for name, module in encoder.named_modules()
               if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm))}
    weights = [module.weight for module in modules.values()]
    encoder.train()
    forget_images = image_features(forget_prompts)
    retain_images = image_features(retain_prompts)
    gf = torch.autograd.grad(forget_loss(forget_images), weights, allow_unused=True)
    retain_loss = -settings["retain_weight"] * (text_features(retain_prompts) * retain_images).sum(-1).mean()
    gr = torch.autograd.grad(retain_loss, weights, allow_unused=True)
    scores = []
    for (name, module), forget_grad, retain_grad in zip(modules.items(), gf, gr):
        if forget_grad is None or retain_grad is None:
            continue
        importance = forget_grad.float().norm() / module.weight.detach().float().norm().clamp_min(1e-8)
        alignment = F.cosine_similarity(forget_grad.flatten(), retain_grad.flatten(), dim=0)
        score = importance / (1 + alignment.clamp_min(0))
        scores.append({"layer": name, "importance": float(importance), "alignment": float(alignment), "score": float(score)})
    del gf, gr
    scores.sort(key=lambda row: row["score"], reverse=True)
    importance = torch.tensor([row["importance"] for row in scores])
    threshold = .5 * (importance.mean() + importance.std(correction=0))
    selected = [row["layer"] for row in scores if row["importance"] >= float(threshold)][:settings["max_layers"]]
    selected = selected or [scores[0]["layer"]]
    print(f"[RAZOR] selected: {selected}; threshold={float(threshold):.6g}", flush=True)

    history = []
    active = []
    for layer in selected:
        active.append(layer)
        encoder.requires_grad_(False)
        parameters = []
        for name in active:
            for parameter in modules[name].parameters(recurse=False):
                parameter.requires_grad_(True)
                parameters.append(parameter)
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"])
        for step in range(settings["inner_steps"]):
            optimizer.zero_grad(set_to_none=True)
            loss = forget_loss(forget_images) - settings["retain_weight"] * (
                text_features(retain_prompts) * retain_images).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            print(f"[RAZOR] layers={len(active)} step={step + 1}/{settings['inner_steps']} loss={float(loss.detach()):.6g}", flush=True)
        pipe.text_encoder.load_state_dict(encoder.state_dict())
        forget_images = image_features(forget_prompts)
        retain_images = image_features(retain_prompts)
        uas, retained = [], []
        n = len(templates)
        for i in range(len(settings["targets"])):
            logits = forget_images[i*n:(i+1)*n] @ label_embeddings[i].T
            uas.append(float((logits.argmax(-1) != 0).float().mean()))
            retain_logits = retain_images[i*n:(i+1)*n] @ label_embeddings[i].T
            retained.append(float((retain_logits.argmax(-1) == 1).float().mean()))
        history.append({"active_layers": list(active), "ua_per_target": uas, "anchor_accuracy": retained})
        print(f"[RAZOR] calibration UA={uas}; anchor accuracy={retained}", flush=True)
        if min(uas) >= settings["target_ua"]:
            break

    edited = {name: parameter.detach().cpu().contiguous() for name, parameter in encoder.named_parameters()
              if any(name.startswith(layer + ".") for layer in active)}
    sampler.close()
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    path = checkpoint / "text_encoder.safetensors"
    save_file(edited, str(path))
    write_json(checkpoint / "metadata.json", {"settings": settings, "weights_sha256": file_digest(path),
        "edited_component": "text_encoder", "edited_parameters": list(edited), "layer_scores": scores,
        "selection_threshold": float(threshold), "calibration_history": history,
        "forget_prompts": forget_prompts, "retain_prompts": retain_prompts,
        "calibration_sampling": sampler.config["sampling"]})
