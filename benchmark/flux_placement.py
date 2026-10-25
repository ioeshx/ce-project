"""Place FLUX transformer blocks across GPUs after reserving resident components."""
from __future__ import annotations


def transformer_memory_budget(options):
    """Return GPU weight budgets after resident models and activation headroom."""
    import torch
    from accelerate.utils import convert_file_size_to_int

    caps = options.get("max_memory")
    devices = sorted(int(k) for k in caps) if caps else list(range(torch.cuda.device_count()))
    reserve = int(options.get("gpu_memory_reserve_gib", 2.0) * 1024 ** 3)
    budgets = {}
    for device in devices:
        if not 0 <= device < torch.cuda.device_count():
            raise ValueError(f"GPU {device} is not visible; max_memory uses logical GPU indices.")
        free, _ = torch.cuda.mem_get_info(device)
        available = free - reserve
        if caps:
            available = min(available, convert_file_size_to_int(caps.get(str(device), caps.get(device))))
        if available <= 0:
            raise ValueError(f"GPU {device} has insufficient free memory after reserving {reserve / 1024**3:g} GiB.")
        budgets[device] = available
    if not budgets:
        raise ValueError("FLUX transformer placement requires at least one visible CUDA GPU.")
    return budgets


def load_sharded_flux(source, options, load_kwargs):
    import torch
    from diffusers import FluxPipeline, FluxTransformer2DModel

    # Place resident components first, so their actual allocations count against
    # the GPU hosting the text encoders and VAE when computing transformer caps.
    pipe = FluxPipeline.from_pretrained(source, transformer=None, **load_kwargs)
    component_device = torch.device(options["device"])
    components = {}
    for name in ("text_encoder", "text_encoder_2", "vae", "image_encoder"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.to(component_device)
            components[name] = str(next(module.parameters()).device)
    budgets = transformer_memory_budget(options)
    print("[transformer memory budgets] " + str({k: round(v / 1024**3, 3) for k, v in budgets.items()}) +
          " GiB (resident components and activation reserve already excluded)", flush=True)
    transformer = FluxTransformer2DModel.from_pretrained(
        source, subfolder="transformer", device_map=options["device_map"], max_memory=budgets, **load_kwargs)
    device_map = transformer.hf_device_map
    if any(str(device) in {"cpu", "disk", "meta"} for device in device_map.values()):
        raise ValueError("The FLUX transformer does not fit in the available GPU budgets. "
                         "Free GPU memory or adjust max_memory/gpu_memory_reserve_gib. "
                         f"Resulting device map: {device_map}")
    pipe.register_modules(transformer=transformer)
    # Keep Accelerate's per-block hooks; never call pipe.to() or transformer.to().
    info = {"scope": "transformer_blocks", "strategy": options["device_map"],
            "components": components, "transformer_device_map": {k: str(v) for k, v in device_map.items()},
            "transformer_max_memory_bytes": {str(k): v for k, v in budgets.items()},
            "gpu_memory_reserve_gib": options.get("gpu_memory_reserve_gib", 2.0)}
    return pipe, info
