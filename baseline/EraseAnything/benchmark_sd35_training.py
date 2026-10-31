"""EraseAnything's four objectives on SD3.5, without changing the FLUX losses."""
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from benchmark.adapters import release_memory
from benchmark.execution import runtime_device
from benchmark.common import file_digest, read_json, write_json
from utils.infoNCE import calculate_steer_loss


def attention_output(transformer):
    """Expose the last joint attention, reordered to upstream's text-first layout.

    SD3 uses image-first Q/K and no FLUX RoPE. Permuting both token axes leaves
    native attention unchanged; the model's normal prediction is never replaced.
    """
    attention = None

    def capture(module, inputs, kwargs):
        nonlocal attention
        image, text = kwargs["hidden_states"], kwargs["encoder_hidden_states"]
        batch = image.shape[0]
        head_dim = module.to_q.out_features // module.heads
        q = module.to_q(image).view(batch, -1, module.heads, head_dim).transpose(1, 2)
        k = module.to_k(image).view(batch, -1, module.heads, head_dim).transpose(1, 2)
        tq = module.add_q_proj(text).view(batch, -1, module.heads, head_dim).transpose(1, 2)
        tk = module.add_k_proj(text).view(batch, -1, module.heads, head_dim).transpose(1, 2)
        if module.norm_q is not None:
            q = module.norm_q(q)
        if module.norm_k is not None:
            k = module.norm_k(k)
        if module.norm_added_q is not None:
            tq = module.norm_added_q(tq)
        if module.norm_added_k is not None:
            tk = module.norm_added_k(tk)
        q, k = torch.cat([tq, q], dim=2), torch.cat([tk, k], dim=2)
        attention = torch.softmax((q @ k.transpose(-2, -1)) * head_dim ** -0.5, dim=-1)

    def append(module, inputs, output):
        return output[0], attention

    return (transformer.transformer_blocks[-1].attn.register_forward_pre_hook(capture, with_kwargs=True),
            transformer.register_forward_hook(append))


def prediction(model, latents, timestep, embeddings):
    sequence, pooled = embeddings
    # The upstream noise interpolation promotes BF16 latents to float32.
    return model(hidden_states=latents.to(model.dtype), timestep=timestep.expand(latents.shape[0]),
                 encoder_hidden_states=sequence, pooled_projections=pooled, return_dict=False)


@torch.no_grad()
def rollout(model, scheduler, embeddings, null_embeddings, start, settings, attention_index=None):
    """Upstream full rollout with SD3 CFG; keep only the requested attention map."""
    scheduler.set_train_timesteps(settings["training_steps"], device=start.device)
    latents, selected = start.clone(), None
    guidance = settings["rollout_guidance"]
    context = (tuple(torch.cat([null, cond]) for null, cond in zip(null_embeddings, embeddings))
               if guidance > 1 else embeddings)
    for index, timestep in enumerate(scheduler.timesteps):
        model_input = torch.cat([latents, latents]) if guidance > 1 else latents
        velocity, attention = prediction(model, model_input, timestep, context)
        if guidance > 1:
            null, conditional = velocity.chunk(2)
            velocity = null + guidance * (conditional - null)
            attention = attention.chunk(2)[1]
        if index == attention_index:
            selected = attention
        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
    return latents, selected


@torch.no_grad()
def image_latents(vae, pixels):
    latents = vae.encode(pixels.to(device=vae.device, dtype=vae.dtype)).latent_dist.sample()
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def make_batch(concept, images, size, tokenizer, sequence_length, text_offset, rng):
    """Keep upstream augmentation; index the actual augmented prompt's T5 tokens."""
    from PIL import Image
    from torchvision.transforms import functional as transform

    image = Image.open(rng.choice(images)).convert("RGB")
    image = transform.resize(image, size)
    top, left = rng.randint(0, image.height - size), rng.randint(0, image.width - size)
    pixels = transform.to_tensor(transform.crop(image, top, left, size, size)) * 2 - 1
    prompt, target = concept["prompt"], concept["name"]
    if rng.random() >= 0.1:
        words = prompt.split(" ")
        rng.shuffle(words)
        prompt = " ".join(words)
    synonym = rng.choice(concept["synonyms"])
    if rng.random() >= 0.5 and target in prompt:
        prompt = prompt.replace(target, synonym)
        target = synonym
    tokens = tokenizer(prompt, padding="max_length", max_length=sequence_length, truncation=True)
    target_ids = set(tokenizer(target).input_ids) - set(tokenizer.all_special_ids)
    indices = [text_offset + index for index, (token, active) in
               enumerate(zip(tokens.input_ids, tokens.attention_mask)) if active and token in target_ids]
    return {"pixel_values": pixels.unsqueeze(0), "prompt": prompt,
            "remove_indices": indices, "synonym": synonym}


def upper_loss(model, vae, scheduler, batch, encode, concept, settings):
    """Same current-model teacher target and attention norm as calculate_upper_loss."""
    clean = image_latents(vae, batch["pixel_values"])
    context, anchor, null = encode(batch["prompt"]), encode(concept["anchor"]), encode("")
    steps = settings["training_steps"]
    index = int(torch.randint(steps, (1,), device=model.device))
    low, high = round(index / steps * 1000), round((index + 1) / steps * 1000)
    timestep = torch.randint(low, high, (1,), device=model.device)
    with torch.no_grad():
        # Upstream samples all the way to the clean endpoint, then evaluates at
        # its independently selected t; it does not stop the rollout at t.
        generated, _ = rollout(model, scheduler, context, null, torch.randn_like(clean), settings)
        e0 = prediction(model, generated, timestep, anchor)[0]
        ep = prediction(model, generated, timestep, context)[0]
        target = e0 - settings["negative_guidance"] * (ep - e0)
    en = prediction(model, generated, timestep, context)[0]
    erase = F.mse_loss(en.float(), target.float())
    noisy = scheduler.add_noise(clean, torch.randn_like(clean), timestep)
    _, attention = prediction(model, noisy, timestep, context)
    # Equal to the original full zero-mask norm: omitted columns contribute 0.
    deactivate = torch.norm(attention[..., batch["remove_indices"]].float(), dim=(0, 1)).sum()
    return [erase, deactivate], timestep


def lower_loss(model, vae, scheduler, batch, encode, concept, settings, timestep, rng):
    """Same Flow Matching target and unchanged upstream InfoNCE sign/temperature."""
    clean = image_latents(vae, batch["pixel_values"])
    noise = torch.randn_like(clean)
    noisy = scheduler.add_noise(clean, noise, timestep)
    velocity, attention = prediction(model, noisy, timestep, encode(batch["prompt"]))
    flow = F.mse_loss(velocity.float(), (noise - clean).float())
    indices = batch["remove_indices"]
    center = attention[..., indices][:, 0].transpose(1, 2)
    start, null = torch.randn_like(clean), encode("")
    index = rng.randrange(settings["training_steps"])
    samples = []
    for prompt in [batch["synonym"], *concept["irrelevant"]]:
        _, selected = rollout(model, scheduler, encode(prompt), null, start, settings, attention_index=index)
        # Preserve upstream's shared column indices and first-head selection.
        samples.append(selected[..., indices][:, 0].transpose(1, 2))
    contrast = calculate_steer_loss(center, samples[0], samples[1:], temperature=0.07)
    return [flow, contrast]


def train(checkpoint, data_directory, settings, *, offline=False):
    from diffusers import AutoencoderKL, SD3Transformer2DModel, StableDiffusion3Pipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from tools.scheduler_process import CustomFlowMatchEulerDiscreteScheduler

    checkpoint, data_directory = Path(checkpoint), Path(data_directory)
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    if checkpoint.exists():
        info = read_json(checkpoint / "metadata.json")
        if info["settings"] != settings or info["weights_sha256"] != file_digest(weights):
            raise ValueError("SD3.5 checkpoint changed; choose another --run-root.")
        print(f"[reuse training] {checkpoint}", flush=True)
        return
    data = read_json(data_directory / "dataset.json")
    if data["settings"] != settings:
        raise ValueError("Training data and training settings disagree.")
    for record in data["records"]:
        for image in record["images"]:
            if file_digest(image["path"]) != image["sha256"]:
                raise ValueError(f"Training image changed: {image['path']}")
    torch.manual_seed(settings["training_seed"])
    rng = random.Random(settings["training_seed"])
    random.seed(settings["training_seed"])
    dtype = getattr(torch, settings["training_dtype"])
    load = dict(torch_dtype=dtype, revision=settings["model_revision"], local_files_only=offline)
    device, text_device = runtime_device(settings["device"]), runtime_device(settings["text_device"])
    text_pipe = StableDiffusion3Pipeline.from_pretrained(
        settings["base_model"], transformer=None, vae=None, **load).to(text_device)
    for encoder in (text_pipe.text_encoder, text_pipe.text_encoder_2, text_pipe.text_encoder_3):
        encoder.requires_grad_(False)
    tokenizer, text_offset = text_pipe.tokenizer_3, text_pipe.tokenizer_max_length
    vae = AutoencoderKL.from_pretrained(settings["base_model"], subfolder="vae", **load).to(device)
    vae.requires_grad_(False)
    model = SD3Transformer2DModel.from_pretrained(settings["base_model"], subfolder="transformer", **load).to(device)
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=settings["rank"], lora_alpha=settings["rank"],
                                init_lora_weights="gaussian", target_modules=["attn.add_k_proj", "attn.add_q_proj"]))
    if settings["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    handles = attention_output(model)
    scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        settings["base_model"], subfolder="scheduler", revision=settings["model_revision"], local_files_only=offline)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings["lr"],
                                  betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8)

    @lru_cache(maxsize=32)
    def encode_cpu(text):
        with torch.no_grad():
            sequence, _, pooled, _ = text_pipe.encode_prompt(
                prompt=text, prompt_2=None, prompt_3=None, device=text_device,
                do_classifier_free_guidance=False, max_sequence_length=settings["training_sequence_length"])
        return sequence.cpu(), pooled.cpu()

    def encode(text):
        return tuple(value.to(device) for value in encode_cpu(text))

    history, order = [], []
    counts = {concept["name"]: {"upper": 0, "lower": 0} for concept in settings["concepts"]}
    model.train()
    for pair in range(settings["iterations"] // 2):
        if not order:
            order = list(range(len(settings["concepts"])))
            rng.shuffle(order)
        index = order.pop()
        concept = settings["concepts"][index]
        images = [image["path"] for image in data["records"][index]["images"]]
        batch = make_batch(concept, images, settings["training_resolution"], tokenizer,
                           settings["training_sequence_length"], text_offset, rng)
        optimizer.zero_grad(set_to_none=True)
        upper, timestep = upper_loss(model, vae, scheduler, batch, encode, concept, settings)
        loss = settings["loss_weights"][0] * upper[0] + settings["loss_weights"][1] * upper[1]
        loss.backward()
        optimizer.step()
        upper_values = [float(value.detach()) for value in upper]
        del upper, loss
        batch = make_batch(concept, images, settings["training_resolution"], tokenizer,
                           settings["training_sequence_length"], text_offset, rng)
        optimizer.zero_grad(set_to_none=True)
        lower = lower_loss(model, vae, scheduler, batch, encode, concept, settings, timestep, rng)
        loss = settings["loss_weights"][2] * lower[0] + settings["loss_weights"][3] * lower[1]
        loss.backward()
        optimizer.step()
        counts[concept["name"]]["upper"] += 1
        counts[concept["name"]]["lower"] += 1
        row = {"pair": pair, "target": concept["name"],
               "losses": upper_values + [float(value.detach()) for value in lower]}
        history.append(row)
        print(f"[training] {pair + 1}/{settings['iterations'] // 2} {concept['name']} {row['losses']}", flush=True)
        del lower, loss
    state = get_peft_model_state_dict(model)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("SD3.5 training produced non-finite LoRA weights.")
    StableDiffusion3Pipeline.save_lora_weights(str(checkpoint), transformer_lora_layers=state)
    write_json(checkpoint / "metadata.json", {"settings": settings, "target_counts": counts,
               "dataset_sha256": file_digest(data_directory / "dataset.json"),
               "weights_sha256": file_digest(weights), "training_history": history,
               "versions": {name: version(name) for name in ("torch", "diffusers", "peft")}})
    for handle in handles:
        handle.remove()
    encode_cpu.cache_clear()
    del text_pipe, model, vae, optimizer, state
    release_memory()
