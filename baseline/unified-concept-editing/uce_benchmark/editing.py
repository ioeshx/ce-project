"""Load and call upstream UCE(), without replacing its token logic or solve."""
import importlib.util
from pathlib import Path
import shutil

from benchmark.common import file_digest, locked, read_json, write_json
from benchmark.layout import timestamp, unique_directory
from benchmark.sampling_profile import make_scheduler
from . import METHOD_ROOT


def load_upstream():
    import torch
    spec = importlib.util.spec_from_file_location("uce_upstream_sd", METHOD_ROOT / "trainscripts/uce_sd_erase.py")
    module = importlib.util.module_from_spec(spec)
    # The original module calls set_grad_enabled(False) on import. Keep that
    # side effect local; the edit call itself runs under no_grad below.
    previous = torch.is_grad_enabled()
    try:
        spec.loader.exec_module(module)
    finally:
        torch.set_grad_enabled(previous)
    return module


def edit(checkpoint, settings, *, offline=False, cpu_threads=4):
    import torch
    from diffusers import DiffusionPipeline
    from safetensors.torch import load_file

    torch.set_num_threads(cpu_threads)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with locked(checkpoint.parent / ".erase.lock"):
        metadata = checkpoint / "metadata.json"
        if checkpoint.exists():
            if not metadata.is_file():
                raise ValueError(f"Incomplete checkpoint: {checkpoint}; use another --run-root.")
            info = read_json(metadata)
            if info["settings"] != settings or file_digest(checkpoint / "uce.safetensors") != info["weights_sha256"]:
                raise ValueError("Checkpoint settings/source/weights changed; choose a new --run-root.")
            print(f"[resume] {checkpoint}", flush=True)
            return checkpoint
        pending = unique_directory(checkpoint.parent, f".checkpoint-pending-{timestamp()}")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                settings["base_model"], torch_dtype=torch.float32,
                revision=settings["model_revision"], local_files_only=offline,
                safety_checker=None, vae=None)
            pipe.scheduler = make_scheduler(pipe.scheduler.config)
            pipe.enable_attention_slicing()
            pipe.to(settings["device"])
            upstream = load_upstream()
            # Native UCE() expects these two globals, as initialized by its CLI.
            upstream.device, upstream.torch_dtype = settings["device"], torch.float32
            with torch.no_grad():
                upstream.UCE(pipe, settings["targets"], settings["guide_concepts"],
                             settings["preserve_concepts"], settings["erase_scale"],
                             settings["preserve_scale"], settings["lamb"], str(pending), "uce")
            weight_path = pending / "uce.safetensors"
            weights = load_file(str(weight_path), device="cpu")
            expected = {name + ".weight" for name, _ in pipe.unet.named_modules()
                        if "attn2" in name and name.endswith(("to_k", "to_v"))}
            if not expected or set(weights) != expected or not all(torch.isfinite(w).all() for w in weights.values()):
                raise ValueError("Native UCE did not produce the complete finite K/V checkpoint.")
            write_json(pending / "metadata.json", {
                "format": "uce-sd14-partial-kv-v1", "base_model": settings["base_model"],
                "task": settings["task"], "targets": settings["targets"], "settings": settings,
                "weights_sha256": file_digest(weight_path), "created_at": timestamp()})
            pending.rename(checkpoint)
            print(f"[checkpoint] {checkpoint}", flush=True)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
    return checkpoint
