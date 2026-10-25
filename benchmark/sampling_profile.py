"""Single source of truth for SD1.4 sampling and legacy cache compatibility."""
from __future__ import annotations

import copy
import json
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "configs" / "sd14_sampling.json"
PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
DPM_NAMES = {"dpm_solver_multistep", "DPMSolverMultistepScheduler"}


def defaults():
    return copy.deepcopy(PROFILE)


def initial_noise(shape, seed, *, dtype, device):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def make_scheduler(config, name=None):
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
    name = name or PROFILE["scheduler"]
    if name in DPM_NAMES:
        # Preserve the model's training noise schedule; pin inference choices.
        keys = ("solver_order", "algorithm_type", "solver_type", "lower_order_final",
                "euler_at_final", "final_sigmas_type", "timestep_spacing", "steps_offset",
                "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas",
                "use_lu_lambdas", "use_flow_sigmas", "rescale_betas_zero_snr")
        return DPMSolverMultistepScheduler.from_config(
            config, **{k: PROFILE["scheduler_config"][k] for k in keys})
    if name in {"ddim", "DDIMScheduler"}:
        return DDIMScheduler.from_config(config)
    raise ValueError(f"Unsupported scheduler: {name}")


def canonical_protocol(sampling):
    """Keep historical protocols intact; normalize the declared shared profile.

    Half-precision deployment does not use TF32. None revision and the TF32
    bookkeeping fields have no effect there and must not split legacy caches.
    Unknown substantive overrides remain in the key.
    """
    result = {k: copy.deepcopy(v) for k, v in sampling.items()
              if k not in {"device", "batch_size", "local_files_only", "cpu_offload", "device_map", "max_memory", "gpu_memory_reserve_gib"}}
    if "device" in sampling and "execution_backend" in result:
        result["execution_backend"] = str(sampling["device"]).split(":", 1)[0]
    if (result.get("scheduler") in DPM_NAMES
            and result.get("noise") == PROFILE["noise"]):
        result["scheduler"] = PROFILE["scheduler"]
        if result.get("model_revision") is None:
            result.pop("model_revision", None)
        if result.get("dtype") == "float16":
            result.pop("float32_matmul_precision", None)
            result.pop("allow_tf32", None)
    return result
