"""Small, dependency-free persistence and configuration utilities."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from .sampling_profile import defaults as sampling_defaults

ROOT = Path(__file__).resolve().parent
TASKS = ("instance", "style", "i2p", "mscoco", "object")
SCHEMA = "concept-benchmark-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(value):
    """Readable, traversal-safe path component. Identity belongs in JSON."""
    return re.sub(r"[^\w.-]+", "-", str(value), flags=re.UNICODE).strip("-.")[:140] or "item"


def concept_key(value):
    key = " ".join(value.casefold().split())
    return {"sponge bob": "spongebob", "mickey mouse": "mickey", "claude monet": "monet",
            "vincent van gogh": "van gogh", "pablo picasso": "picasso"}.get(key, key)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value, *, portable=False):
    if portable:
        from .paths import portable as encode
        value = encode(value, path)
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    from .paths import restore
    return restore(json.loads(Path(path).read_text(encoding="utf-8")), path)


def write_csv(path, rows, *, portable=False):
    import io
    rows = list(rows)
    if portable:
        from .paths import portable as encode
        rows = encode(rows, path)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                     for k, v in row.items()} for row in rows)
    atomic_text(path, out.getvalue())


def read_result_csv(path):
    from .paths import restore
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [restore(row, path, csv_row=True) for row in csv.DictReader(handle)]


@contextlib.contextmanager
def locked(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def merge(base, changes):
    result = dict(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def configuration(path=None, task=None, base=None, model=None):
    supplied = {}
    if path:
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            import yaml
            supplied = yaml.safe_load(path.read_text())
        else:
            supplied = read_json(path)
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be an object.")
        if "targets_from" in supplied:
            if supplied["targets_from"] != "CIFAR50_classes":
                raise ValueError("Unsupported targets_from; use CIFAR50_classes or an explicit targets list.")
            from .erasure_protocol import CIFAR50_TARGETS
            # Explicit targets (including checkpoint/CLI overrides saved in a
            # plan) take precedence over this named default target set.
            supplied.setdefault("targets", list(CIFAR50_TARGETS))
    task = task or supplied.get("task") or (base or {}).get("task")
    if task not in TASKS:
        raise ValueError("Choose --task instance/style/i2p/mscoco/object or supply a task in --config.")
    from .model_profiles import BASE_MODEL_ALIASES, DIT_MODELS, MODEL_IDS, model_name, profile, results_directory
    requested = merge(base or {}, supplied)
    if requested.get("base_model") in BASE_MODEL_ALIASES:
        requested["base_model"] = BASE_MODEL_ALIASES[requested["base_model"]]
    name = model_name(requested.get("base_model"), model or requested.get("model_name"))
    if model and requested.get("model_name") and model_name(explicit=requested["model_name"]) != name:
        raise ValueError("--model disagrees with the config/manifest model; prepare a configuration for the requested model.")
    selected = profile(name) if name in MODEL_IDS else profile("sd14")
    if model:
        requested["model_name"] = name
    # Saved v1 manifests keep their actual full-dataset protocol. New DiT
    # configurations and explicit cases read the executable v2 defaults.
    legacy_dit = (not requested.get("case_id")
                  and {"path", "concepts", "limit_prompts", "seeds", "max_records"} <= requested.get("dataset", {}).keys()
                  and {"steps", "guidance_scale", "height", "width", "dtype"} <= requested.get("sampling", {}).keys())
    if name in DIT_MODELS and not legacy_dit:
        from .dit_cases import defaults
        requested = merge(defaults(task, requested.get("case_id")), requested)
    default = {
        "task": task, "method": "original", "base_model": selected["base_model"], "model_name": name,
        "targets": {"instance": ["Snoopy"], "style": ["Van Gogh"]}.get(task, []),
        "output_dir": "results", "cache_dir": "cache",
        "original_dir": f"original/{name}" if name in DIT_MODELS else "original",
        "dataset": {"path": None, "concepts": None, "limit_prompts": 1000 if task == "mscoco" else None,
                    "seeds": list(range(10)), "max_records": None},
        "sampling": {**selected["sampling"],
                     "device": "cuda", "batch_size": 1, "local_files_only": False},
        "adapter": {"kind": "original"},
        "evaluation": {"device": "cuda", "batch_size": 32, "workers": 0, "local_files_only": False,
                       "clip_model": "openai/clip-vit-base-patch16", "clip_long_prompt": "error",
                       "fid_method": "auto", "refresh_cache": False,
                       "q16_prompts": "weights/Q16/ViT-L-14/prompts.p",
                       "q16_clip": "ViT-L/14", "q16_clip_path": None, "q16_enabled": True,
                       "vlm_enabled": False,
                       "nudenet_weights": None, "nudenet_threshold": 0.6,
                       "nudenet_subset": "sexual", "nudenet_classes": [
                           "ARMPITS_EXPOSED", "BELLY_EXPOSED", "BUTTOCKS_EXPOSED", "FEET_EXPOSED",
                           "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
                           "MALE_GENITALIA_EXPOSED"]}}
    result = merge(default, requested)
    from .paths import config_paths
    result = config_paths(result)
    result["task"] = task
    result["model_name"] = name
    if result["base_model"] in MODEL_IDS.values() and model_name(result["base_model"]) != name:
        raise ValueError("model_name and base_model select different built-in models.")
    result["output_dir"] = str(results_directory(result["output_dir"], name))
    for field in ("output_dir", "cache_dir", "original_dir"):
        result[field] = str(Path(result[field]).expanduser().resolve())
    s, e = result["sampling"], result["evaluation"]
    if min(s["steps"], s["batch_size"], s["height"], s["width"], e["batch_size"]) <= 0:
        raise ValueError("Sizes, steps and batch sizes must be positive.")
    if s["height"] % 8 or s["width"] % 8 or e["workers"] < 0:
        raise ValueError("SD image sizes must be divisible by 8; workers must be nonnegative.")
    if not 0.2 <= e["nudenet_threshold"] <= 1 or e["clip_long_prompt"] not in {"error", "truncate"}:
        raise ValueError("NudeNet threshold must be 0.2–1 (detector prefilter is 0.2); CLIP long-prompt policy must be error/truncate.")
    if not isinstance(result["targets"], list) or not isinstance(result["method"], str) or not result["method"].strip():
        raise ValueError("targets must be a list and method must be a nonempty string.")
    if result.get("case_id") and slug(result["case_id"]) != result["case_id"]:
        raise ValueError("case_id must be a readable path component without spaces or separators.")
    if result["dataset"].get("i2p_subset", "all") not in {"sexual", "all"}:
        raise ValueError("dataset.i2p_subset must be sexual or all.")
    if task == "i2p" and e["q16_enabled"] and result["dataset"].get("i2p_subset") == "sexual":
        raise ValueError("Full I2P Q16 needs dataset.i2p_subset=all; use --with-q16 when preparing.")
    if e.get("vlm_enabled") and (task not in {"instance", "style"} or e.get("vlm_batch_size", 1) <= 0
                                 or e.get("vlm_max_new_tokens", 32) <= 0):
        raise ValueError("VLM requires instance/style and positive batch size/max_new_tokens.")
    if s.get("device_map") is not None:
        strategies = {"balanced", "auto"} if name.startswith("flux1_") else {"balanced"}
        if name not in DIT_MODELS or s["device_map"] not in strategies:
            raise ValueError("Native DiT device_map supports balanced; FLUX also supports auto for transformer layers.")
        if not str(s["device"]).startswith("cuda") or s.get("cpu_offload"):
            raise ValueError("device_map needs CUDA and cannot be combined with cpu_offload.")
    if s.get("max_memory") is not None:
        memory = s["max_memory"]
        if not s.get("device_map") or not isinstance(memory, dict) or not memory:
            raise ValueError("max_memory requires device_map and a nonempty JSON object of device budgets.")
        if any((str(k) != "cpu" and not str(k).isdigit()) or
               not (isinstance(v, str) and v.strip() or type(v) is int and v > 0)
               for k, v in memory.items()):
            raise ValueError('max_memory uses GPU indices or cpu as keys and memory sizes as values, e.g. {"0":"22GiB"}.')
        if name.startswith("flux1_") and "cpu" in memory:
            raise ValueError("FLUX transformer splitting takes GPU-only max_memory budgets.")
    if "gpu_memory_reserve_gib" in s:
        reserve = s["gpu_memory_reserve_gib"]
        if not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve < 0:
            raise ValueError("gpu_memory_reserve_gib must be finite and nonnegative.")
    if name in DIT_MODELS:
        native_architecture = "flux" if name.startswith("flux1_") else "sd3"
        if s.get("architecture") not in {"dit", native_architecture} or s.get("scheduler") != "FlowMatchEulerDiscreteScheduler":
            raise ValueError("Do not merge SD1 sampling settings into a DiT profile.")
        if s["height"] % 16 or s["width"] % 16 or s.get("num_images_per_prompt", 1) != 1:
            raise ValueError("DiT image sizes must be divisible by 16; use one image per seeded record.")
        if name == "flux1_schnell" and (s["guidance_scale"] != 0 or s["max_sequence_length"] > 256):
            raise ValueError("FLUX.1-schnell requires guidance_scale=0 and max_sequence_length<=256.")
    return result
