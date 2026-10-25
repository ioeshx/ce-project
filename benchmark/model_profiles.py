"""Model selection, native sampling defaults, and model-scoped result roots."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MODEL_IDS = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "flux1_dev": "black-forest-labs/FLUX.1-dev",
    "flux1_schnell": "black-forest-labs/FLUX.1-schnell",
    "sd30_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35_medium": "stabilityai/stable-diffusion-3.5-medium",
}
DIT_MODELS = tuple(k for k in MODEL_IDS if k != "sd14")
BASE_MODEL_ALIASES = {"stabilityai/stable-diffusion-3-medium": MODEL_IDS["sd30_medium"]}
ALIASES = {"sd3_medium": "sd30_medium", **{v: k for k, v in MODEL_IDS.items()},
           **{alias: "sd30_medium" for alias in BASE_MODEL_ALIASES}}
PROFILE_PATH = ROOT / "configs/dit_sampling.json"


def model_name(base_model=None, explicit=None):
    if explicit:
        name = ALIASES.get(explicit, explicit)
    else:
        name = ALIASES.get(base_model, None) if base_model else "sd14"
        if name is None:
            name = re.sub(r"[^\w.-]+", "-", str(base_model).rstrip("/").split("/")[-1]).strip("-.")
    if not name or not re.fullmatch(r"[\w.-]+", name) or name in {".", ".."}:
        raise ValueError("model_name must be a readable directory name.")
    return name


def profile(name):
    name = model_name(explicit=name)
    if name == "sd14":
        from .sampling_profile import defaults
        return {"base_model": MODEL_IDS[name], "pipeline": "StableDiffusionPipeline", "sampling": defaults()}
    if name not in DIT_MODELS:
        raise ValueError(f"No built-in sampling profile for {name}; use one of {', '.join(MODEL_IDS)}.")
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))["profiles"]
    key = "sd3_medium" if name == "sd30_medium" and name not in profiles else name
    result = copy.deepcopy(profiles[key])
    result["sampling"].update(pipeline=result["pipeline"],
                               execution_backend="cuda",
                               noise="per-record CPU float32 normal, cast to model dtype" +
                               ("; FLUX 2x2 packing" if name.startswith("flux1_") else ""))
    return result


def results_directory(root, name):
    root = Path(root).expanduser().resolve()
    name = model_name(explicit=name)
    if root.is_relative_to(ROOT / "results"):
        parts = root.relative_to(ROOT / "results").parts
        if parts and parts[0] == name:
            return root
        if parts and parts[0] in MODEL_IDS:
            raise ValueError(f"Result directory {root} belongs to {parts[0]}, not {name}.")
        return ROOT / "results" / name / root.relative_to(ROOT / "results")
    if root.name == name:
        return root
    if root.name in MODEL_IDS:
        raise ValueError(f"Result directory {root} belongs to {root.name}, not {name}.")
    return root / name


def resolve_scheduler(config):
    """Resolve only the small native scheduler config before allocating caches."""
    if config["model_name"] not in DIT_MODELS:
        return
    sampling = config["sampling"]
    if sampling["scheduler"] != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("DiT profiles require their native FlowMatchEulerDiscreteScheduler.")
    if not sampling.get("scheduler_config"):
        from diffusers import FlowMatchEulerDiscreteScheduler
        sampling["scheduler_config"] = dict(FlowMatchEulerDiscreteScheduler.load_config(
            config["base_model"], subfolder="scheduler", revision=sampling.get("model_revision"),
            local_files_only=sampling.get("local_files_only", False)))
