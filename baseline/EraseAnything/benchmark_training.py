"""FLUX compatibility and a small runner around EraseAnything's four losses."""
from functools import lru_cache
from pathlib import Path
import random
from types import SimpleNamespace

import torch

from benchmark.adapters import release_memory
from benchmark.execution import runtime_device
from benchmark.common import file_digest, read_json, slug, write_json


def attention_output(transformer):
    """Expose the final dual-stream attention map expected by the upstream losses.

    The normal Diffusers forward remains intact. The side calculation uses the
    same Q/K projections, normalization, RoPE, scale and text-first ordering.
    """
    from diffusers.models.embeddings import apply_rotary_emb

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
        q = torch.cat([module.norm_added_q(tq), module.norm_q(q)], dim=2)
        k = torch.cat([module.norm_added_k(tk), module.norm_k(k)], dim=2)
        rotary = kwargs.get("image_rotary_emb")
        if rotary is not None:
            q, k = apply_rotary_emb(q, rotary), apply_rotary_emb(k, rotary)
        attention = torch.softmax((q @ k.transpose(-2, -1)) * q.shape[-1] ** -0.5, dim=-1)

    def append(module, inputs, output):
        return output[0], attention

    return (transformer.transformer_blocks[-1].attn.register_forward_pre_hook(capture, with_kwargs=True),
            transformer.register_forward_hook(append))


def prepare_images(directory, settings, sampling_config, *, offline=False):
    """Generate separate training images, without reading benchmark test prompts."""
    from benchmark.dit_adapter import DiTAdapter

    directory = Path(directory)
    metadata = directory / "dataset.json"
    if metadata.exists():
        if read_json(metadata)["settings"] != settings:
            raise ValueError("Training data settings changed; choose a new --run-root.")
        print(f"[reuse training data] {directory}", flush=True)
        return
    config = dict(sampling_config)
    config["adapter"] = {"kind": "original"}
    adapter = DiTAdapter(config)
    records = []
    for target in settings["concepts"]:
        folder = directory / slug(target["name"])
        folder.mkdir(parents=True, exist_ok=True)
        images = []
        for index in range(settings["training_images_per_concept"]):
            path = folder / f"training-{index:03d}.png"
            record = {"prompt": target["prompt"], "seed": settings["training_seed"] + index}
            if not path.exists():
                adapter.generate([record], config["sampling"])[0].save(path)
            images.append({"path": str(path), "sha256": file_digest(path), **record})
        records.append({"concept": target["name"], "images": images})
    adapter.close()
    write_json(metadata, {"settings": settings, "records": records,
                          "sampling": config["sampling"], "base_model": config["base_model"]})


def make_batch(concept, images, size, tokenizer, rng):
    """Original word shuffle and synonym replacement, with explicit local aliases."""
    from PIL import Image
    from torchvision.transforms import functional as transform

    image = Image.open(rng.choice(images)).convert("RGB")
    image = transform.resize(image, size)
    top = rng.randint(0, image.height - size)
    left = rng.randint(0, image.width - size)
    pixels = transform.to_tensor(transform.crop(image, top, left, size, size)) * 2 - 1
    prompt = concept["prompt"]
    if rng.random() >= 0.1:
        words = prompt.split(" ")
        rng.shuffle(words)
        prompt = " ".join(words)
    synonym = rng.choice(concept["synonyms"])
    # Match upstream's token-ID membership rule, excluding EOS and padding.
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    word = tokenizer(concept["name"], return_tensors="pt").input_ids[0].tolist()[:-1]
    indices = [index for index, token in enumerate(tokens[:-1]) if token in word]
    if rng.random() >= 0.5:
        prompt = prompt.replace(concept["name"], synonym)
    return {"pixel_values": pixels.unsqueeze(0), "prompts": [prompt],
            "remove_indices": [indices], "synonym_words": [synonym]}


def train(checkpoint, data_directory, settings, *, offline=False):
    from importlib.metadata import version
    from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from tools.scheduler_process import CustomFlowMatchEulerDiscreteScheduler
    from utils.calc_loss import calculate_upper_loss, calculate_lower_loss

    checkpoint, data_directory = Path(checkpoint), Path(data_directory)
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    if checkpoint.exists():
        info = read_json(checkpoint / "metadata.json")
        if info["settings"] != settings or info["weights_sha256"] != file_digest(weights):
            raise ValueError("EraseAnything checkpoint changed; choose another --run-root.")
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
    text_pipe = FluxPipeline.from_pretrained(settings["base_model"], transformer=None, vae=None,
                                             **load).to(text_device)
    text_pipe.text_encoder.requires_grad_(False)
    text_pipe.text_encoder_2.requires_grad_(False)
    tokenizer = text_pipe.tokenizer_2
    vae = AutoencoderKL.from_pretrained(settings["base_model"], subfolder="vae", **load).to(device)
    vae.requires_grad_(False)
    model = FluxTransformer2DModel.from_pretrained(settings["base_model"], subfolder="transformer", **load).to(device)
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
    args = SimpleNamespace(cache_latents=False, resolution=settings["training_resolution"],
                           guidance_scale=settings["training_guidance"])

    @lru_cache(maxsize=32)
    def encode(text):
        with torch.no_grad():
            values = text_pipe.encode_prompt(prompt=text, prompt_2=None, device=text_device,
                                             max_sequence_length=settings["training_sequence_length"])
        return tuple(value.cpu() for value in values)

    def embeddings(prompts, text_encoders, tokenizers):
        text = prompts if isinstance(prompts, str) else prompts[0]
        return tuple(value.to(device) for value in encode(text))

    history, order = [], []
    counts = {concept["name"]: {"upper": 0, "lower": 0} for concept in settings["concepts"]}
    criteria = torch.nn.MSELoss()
    model.train()
    for pair in range(settings["iterations"] // 2):
        if not order:
            order = list(range(len(settings["concepts"])))
            rng.shuffle(order)
        index = order.pop()
        concept = settings["concepts"][index]
        images = [image["path"] for image in data["records"][index]["images"]]
        upper_batch = make_batch(concept, images, args.resolution, tokenizer, rng)
        optimizer.zero_grad(set_to_none=True)
        upper, timestep = calculate_upper_loss(
            args, upper_batch, embeddings, None, None, model, scheduler, upper_batch["prompts"], vae, criteria,
            settings["negative_guidance"], dtype, neg_prompts=concept["anchor"], ddim_steps=settings["training_steps"])
        loss = settings["loss_weights"][0] * upper[0] + settings["loss_weights"][1] * upper[1]
        loss.backward()
        optimizer.step()
        upper_values = [float(value.detach()) for value in upper]
        del upper, loss
        lower_batch = make_batch(concept, images, args.resolution, tokenizer, rng)
        optimizer.zero_grad(set_to_none=True)
        lower = calculate_lower_loss(
            args, lower_batch, embeddings, None, None, model, scheduler, lower_batch["prompts"], vae, criteria,
            settings["negative_guidance"], dtype, t_enc_ddpm=timestep, ddim_steps=settings["training_steps"],
            K=3, ir_concept_lst=concept["irrelevant"])
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
        raise RuntimeError("EraseAnything produced non-finite LoRA weights.")
    FluxPipeline.save_lora_weights(str(checkpoint), transformer_lora_layers=state)
    write_json(checkpoint / "metadata.json", {"settings": settings, "target_counts": counts,
               "dataset_sha256": file_digest(data_directory / "dataset.json"),
               "weights_sha256": file_digest(weights), "training_history": history,
               "versions": {name: version(name) for name in ("torch", "diffusers", "peft")}})
    for handle in handles:
        handle.remove()
    encode.cache_clear()
    del text_pipe, model, vae, optimizer, state
    release_memory()
