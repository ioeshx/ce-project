"""UCE's FLUX text-projection edit, with the equivalent SD3.5 input layout.

Only two projection weights are saved. Sampling stays in benchmark.DiTAdapter.
"""
import copy
import json
from pathlib import Path
import time

from benchmark.common import file_digest, read_json, write_json
from benchmark.adapters import release_memory

PROJECTIONS = ("context_embedder", "time_text_embed.text_embedder.linear_1")
FORMAT = "uce-dit-text-projections-v1"


def load_projections(settings, offline):
    """Read just the two input projections, without allocating the transformer."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors import safe_open

    def model_file(name):
        model = Path(settings["base_model"])
        if model.is_dir():
            return model / "transformer" / name
        return Path(hf_hub_download(settings["base_model"], name, subfolder="transformer",
                                   revision=settings["model_revision"], local_files_only=offline))

    # Both standard single-file and sharded Diffusers checkpoints are supported.
    try:
        index = json.loads(model_file("diffusion_pytorch_model.safetensors.index.json").read_text())
        locations = index["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):
        locations = {f"{name}.{field}": "diffusion_pytorch_model.safetensors"
                     for name in PROJECTIONS for field in ("weight", "bias")}
    keys = [f"{name}.{field}" for name in PROJECTIONS for field in ("weight", "bias")]
    state = {}
    for shard in dict.fromkeys(locations[key] for key in keys):
        with safe_open(model_file(shard), framework="pt", device="cpu") as handle:
            for key in keys:
                if locations[key] == shard:
                    state[key] = handle.get_tensor(key).float()
    return state


def closed_form(weight, bias, erase, guide, preserve, erase_scale, preserve_scale, lamb):
    """Original UCE normal equation; columns are concept vectors, solve in FP32.

    Guide/retain outputs include the original bias, as in uce_flux_edit.py.
    Biases themselves are not edited. solve(B.T, A.T).T equals A @ inverse(B).
    """
    import torch

    numerator = lamb * weight + erase_scale * ((weight @ guide + bias[:, None]) @ erase.T)
    denominator = lamb * torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device)
    denominator += erase_scale * (erase @ erase.T)
    numerator += preserve_scale * ((weight @ preserve + bias[:, None]) @ preserve.T)
    denominator += preserve_scale * (preserve @ preserve.T)
    return torch.linalg.solve(denominator.T, numerator.T).T.contiguous()


def edit(checkpoint, settings, *, offline=False):
    """Encode names, release text encoders, then jointly edit all target concepts."""
    import torch
    from diffusers import FluxPipeline, StableDiffusion3Pipeline
    from safetensors.torch import save_file

    checkpoint = Path(checkpoint)
    metadata = checkpoint / "metadata.json"
    weights_path = checkpoint / "uce.safetensors"
    if checkpoint.exists():
        info = read_json(metadata)
        if info["settings"] != settings or info["weights_sha256"] != file_digest(weights_path):
            raise ValueError(f"Checkpoint differs from requested edit; choose another --run-root: {checkpoint}")
        print(f"[reuse edit] {checkpoint}", flush=True)
        return

    started = time.perf_counter()
    flux = settings["model_name"] == "flux1_dev"
    cls = FluxPipeline if flux else StableDiffusion3Pipeline
    device = settings["edit_device"]
    # No denoiser or VAE is needed to obtain the concept representations.
    pipe = cls.from_pretrained(settings["base_model"], transformer=None, vae=None,
                               torch_dtype=getattr(torch, settings["embedding_dtype"]),
                               revision=settings["model_revision"], local_files_only=offline).to(device)
    tokenizer = pipe.tokenizer_2 if flux else pipe.tokenizer_3
    length = settings["max_sequence_length"]
    embeddings = {}
    with torch.no_grad():
        for concept in dict.fromkeys(settings["targets"] + settings["guides"] + settings["preserve_concepts"]):
            kwargs = dict(prompt=concept, prompt_2=None, device=device,
                          num_images_per_prompt=1, max_sequence_length=length)
            if not flux:
                kwargs.update(prompt_3=None, do_classifier_free_guidance=False)
            encoded = pipe.encode_prompt(**kwargs)
            tokens = tokenizer(concept, padding="max_length", max_length=length,
                               truncation=True, return_tensors="pt")
            # Preserve upstream's attention_mask.sum()-2 token selection.
            # SD3 prepends the padded CLIP sequence before the T5 sequence.
            index = int(tokens.attention_mask.sum()) - 2
            if not flux:
                index = pipe.tokenizer_max_length + index % length
            embeddings[concept] = (encoded[0][:, index, :].float().cpu(),
                                   encoded[1 if flux else 2].float().cpu())
    del pipe, encoded, tokenizer
    release_memory()

    # The normal equation is small enough for CPU; the 4096-dim solve never
    # competes with text encoders or the denoiser for GPU memory.
    state = load_projections(settings, offline)
    edited = {}
    for index, name in enumerate(PROJECTIONS):
        erase = torch.cat([embeddings[c][index] for c in settings["targets"]]).T
        guide = torch.cat([embeddings[c][index] for c in settings["guides"]]).T
        preserve = (torch.cat([embeddings[c][index] for c in settings["preserve_concepts"]]).T
                    if settings["preserve_concepts"] else erase[:, :0])
        edited[name + ".weight"] = closed_form(
            state[name + ".weight"], state[name + ".bias"], erase, guide, preserve,
            settings["erase_scale"], settings["preserve_scale"], settings["lamb"])

    checkpoint.mkdir(parents=True)
    temporary = checkpoint / "uce-pending.safetensors"
    save_file(edited, str(temporary))
    temporary.replace(weights_path)
    write_json(metadata, {"format": FORMAT, "settings": settings,
                          "weights_sha256": file_digest(weights_path),
                          "edit_seconds": time.perf_counter() - started})
    print(f"[edited] {checkpoint}", flush=True)


def create(config):
    """Load a partial UCE checkpoint once, then return the shared native sampler."""
    import torch
    from safetensors.torch import load_file
    from benchmark.dit_adapter import DiTAdapter

    checkpoint = Path(config["adapter"]["checkpoint"])
    info = read_json(checkpoint / "metadata.json")
    settings = info["settings"]
    if (info["format"] != FORMAT or settings != config["erasure_parameters"]
            or settings["base_model"] != config["base_model"] or settings["targets"] != config["targets"]
            or settings["model_revision"] != config["sampling"].get("model_revision")
            or info["weights_sha256"] != file_digest(checkpoint / "uce.safetensors")):
        raise ValueError("UCE checkpoint does not match the requested DiT experiment.")
    native_config = copy.deepcopy(config)
    native_config["adapter"] = {"kind": "original"}
    adapter = DiTAdapter(native_config)
    with torch.no_grad():
        for name, value in load_file(str(checkpoint / "uce.safetensors")).items():
            parameter = adapter.pipe.transformer.get_parameter(name)
            parameter.copy_(value.to(parameter))
    return adapter
