"""Sampling adapters. No evaluation code or erasure algorithms live here."""
from __future__ import annotations

import gc
import importlib
import os
from pathlib import Path
import tempfile
from typing import Protocol

from .common import locked, read_json
from .data import valid_image


class GeneratorAdapter(Protocol):
    def generate(self, records: list[dict], sampling_config: dict):
        """Return one PIL.Image per record, in the same order (or an iterable)."""
        ...

    def close(self):
        ...


def release_memory():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DiffusersAdapter:
    def __init__(self, config):
        import torch
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline

        self.config = config
        opts, spec = config["sampling"], config["adapter"]
        device = opts["device"]
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; set sampling.device=cpu.")
        dtype = opts["dtype"]
        dtype = (torch.float16 if str(device).startswith("cuda") else torch.float32) if dtype == "auto" else {
            "float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        model = spec.get("model_path") if spec["kind"] == "diffusers" else config["base_model"]
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=dtype, local_files_only=opts["local_files_only"],
            revision=opts.get("model_revision"),
            safety_checker=None, requires_safety_checker=False)
        from .sampling_profile import make_scheduler
        self.pipe.scheduler = make_scheduler(self.pipe.scheduler.config, opts["scheduler"])
        kind = spec["kind"]
        if kind == "unet":
            self.pipe.unet.load_state_dict(load_state(spec["checkpoint"]), strict=True)
        elif kind == "lora":
            self.pipe.load_lora_weights(spec["checkpoint"], local_files_only=opts["local_files_only"])
            self.pipe.fuse_lora(lora_scale=float(spec.get("scale", 1.0)))
        elif kind == "casu2":
            apply_casu(self.pipe.unet, spec["checkpoint"], config["base_model"])
        self.pipe.to(device)
        from .sampling_profile import PROFILE, DPM_NAMES
        if opts.get("noise") == PROFILE["noise"] and opts["scheduler"] in DPM_NAMES:
            # Legacy SD1 uses SlicedAttnProcessor, not the default SDPA kernel.
            # At float16 this changes pixels despite identical noise/settings.
            self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, records, sampling_config):
        import torch
        pipe, opts = self.pipe, sampling_config
        # Match CASU's record-wise CPU latents, independent of batch boundaries.
        generators = [torch.Generator(device="cpu").manual_seed(r["seed"]) for r in records]
        shape = (pipe.unet.config.in_channels, opts["height"] // pipe.vae_scale_factor,
                 opts["width"] // pipe.vae_scale_factor)
        from .sampling_profile import initial_noise, PROFILE
        if opts.get("noise") == PROFILE["noise"]:
            latents = torch.stack([initial_noise(shape, r["seed"], dtype=pipe.unet.dtype, device="cpu")
                                  for r in records])
        else:
            # Historical manifests without a noise declaration used this path.
            latents = torch.stack([torch.randn(shape, generator=g, dtype=pipe.unet.dtype) for g in generators])
        with torch.inference_mode():
            return pipe(prompt=[r["prompt"] for r in records], latents=latents.to(pipe.device),
                        generator=generators, height=opts["height"], width=opts["width"],
                        num_inference_steps=opts["steps"], guidance_scale=opts["guidance_scale"],
                        negative_prompt=[opts["negative_prompt"]] * len(records)).images

    def close(self):
        self.pipe = None
        release_memory()


def load_state(path):
    path = Path(path).expanduser()
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state.get("state_dict", state)


def apply_casu(unet, checkpoint, base_model):
    """Load CASU's absolute edited SD1 attention weights without importing DiT."""
    import torch
    directory = Path(checkpoint).expanduser()
    if directory.is_file():
        directory = directory.parent
    metadata = read_json(directory / "metadata.json")
    if metadata.get("format") != "casu2-sd1-v1" or metadata.get("base_model") != base_model:
        raise ValueError("CASU adapter needs a casu2-sd1-v1 checkpoint with the same base model.")
    weights = load_state(directory / "weights.safetensors")
    keys = {key for group in metadata["active_groups"] for key in metadata["groups"][group]}
    if keys != set(weights):
        raise ValueError("CASU checkpoint keys do not match active_groups.")
    params = dict(unet.named_parameters())
    for key in keys:
        if key not in params or key not in weights or params[key].shape != weights[key].shape:
            raise ValueError(f"CASU checkpoint/model mismatch: {key}")
    with torch.no_grad():
        for key in keys:
            params[key].copy_(weights[key].to(params[key]))


def create_adapter(config):
    spec = config["adapter"]
    if spec["kind"] == "plugin":
        module, factory = spec["factory"].split(":", 1)
        adapter = getattr(importlib.import_module(module), factory)(config)
        if not callable(getattr(adapter, "generate", None)) or not callable(getattr(adapter, "close", None)):
            raise ValueError("Plugin factory must return generate(records, sampling_config) and close().")
        return adapter
    if spec["kind"] in {"original", "diffusers", "unet", "lora", "casu2"}:
        return DiffusersAdapter(config)
    raise ValueError(f"Unknown sampling adapter: {spec['kind']}")


def atomic_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix=".sample-", dir=path.parent)
    os.close(fd)
    try:
        image.convert("RGB").save(tmp, format="PNG")
        if not valid_image(tmp):
            raise ValueError("Adapter produced an unreadable image.")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sample_manifest(manifest, mode="both"):
    if manifest.get("dataset", {}).get("imported_from"):
        raise ValueError("Imported CASU sampling is immutable. Reuse the images or prepare a new protocol.")
    if not manifest.get("config") or manifest.get("adapter_identity", {}).get("kind") == "existing":
        raise ValueError("Existing-image manifests can be evaluated; prepare a new sampling manifest to generate images.")
    config = manifest["config"]
    from .data import adapter_identity
    if adapter_identity(config) != manifest.get("adapter_identity"):
        raise ValueError("Adapter weights/source changed after preparation; prepare a new manifest.")
    if config["adapter"]["kind"] == "existing":
        raise ValueError("Existing-image adapter does not generate images.")
    modes = ["reference", "image"] if mode == "both" else [mode]
    for field in modes:
        # Serializes shared original fills; lock released even if generation fails.
        lock_root = Path(manifest["records"][0][field]).parent
        with locked(lock_root / ".sampling.lock"):
            missing = [r for r in manifest["records"] if not valid_image(r[field])]
            print(f"[sample:{field}] {len(missing)} missing/corrupt of {len(manifest['records'])}", flush=True)
            if not missing:
                continue
            cfg = dict(config)
            if field == "reference":
                cfg["adapter"] = {"kind": "original"}
            adapter = create_adapter(cfg)
            try:
                size = cfg["sampling"]["batch_size"]
                for start in range(0, len(missing), size):
                    batch = missing[start:start + size]
                    images = list(adapter.generate(batch, cfg["sampling"]))
                    if len(images) != len(batch):
                        raise ValueError("Adapter must return exactly one image per record, in order.")
                    for row, image in zip(batch, images):
                        if image.size != (cfg["sampling"]["width"], cfg["sampling"]["height"]):
                            raise ValueError(f"Adapter returned incorrect image dimensions for {row['sample_id']}.")
                        atomic_image(image, row[field])
                    print(f"[sample:{field}] {min(start + size, len(missing))}/{len(missing)}", flush=True)
            finally:
                adapter.close()
                del adapter
                release_memory()
