"""CPU orchestration and short-lived workers bound to one visible CUDA device."""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .common import read_json

DEVICE_ENV = "BENCHMARK_WORKER_DEVICE"

# Exact before/after source pairs for this execution-only migration. Old
# checkpoints/plans remain immutable; any later source change is still rejected.
SOURCE_MIGRATIONS = {
    "unified-concept-editing": {
        "dit.py": [
            "90a51e01f32c5fba5d44a7d1b3fbaaf7bae5690a8f453317c70e9f0af4def191",
            "659d0a3456e885ea3f493857187896d257b5c2153b54b640973db2325b52665a"
        ],
        "dit_editing.py": [
            "fbb30c70ce75f992ad4cc7c8f65c7e3d34c838c972bde459d6454acf5bea0ecf",
            "4b4221ce66a54295e5b0ee5885ea7e75bef4eff79304eb4b65a94ce6e604399d"
        ]
    },
    "ESD": {
        "esd_benchmark/dit.py": [
            "dd8bf25ac4e4eb6467ceab7e993ff62c66759f1e5d9e09dea5541eb71f9105ea",
            "5aacb058190bf9f45a9e7cddd2898d9364568e7441ccc812a0cf4c674dfd8bc4"
        ],
        "esd_benchmark/dit_training.py": [
            "ecf785857fddb7370da44546852fe9014ca22b4efbda17a895cea6c37859e9b5",
            "b4d18a1776d1348a32c4dec4210abd2866cc0de70954db5bd279f6ca4eff317a"
        ]
    },
    "EraseAnything": {
        "benchmark_dit.py": [
            "11adc4878a51e5bdd043be49f390a10d617adfdd1251cfe3ec617f1100adfe09",
            "31e6b821e28c244c61b8e3dc4c23bee38cf9b9e628bf69f33c5f2243ba499421"
        ],
        "benchmark_training.py": [
            "acf96d0110913d6228673b43c0a4dfebe8354106bd53fdc477405b18b1d840e6",
            "b726ad98618a9945d3d29483c4e1c0b7c374337e3bd63035f553d30b7aee04ae"
        ],
        "benchmark_sd35.py": [
            "c011f22a70bb0a2b57644a3bd128dafa0cfdd1c8ebd52d2aa74517be06bd253d",
            "c35465e23bab69d5e6c837f4f216bbab3889962b93c51ad29fa7c407763dbb81"
        ],
        "benchmark_sd35_training.py": [
            "55c23051a24ee0c9822782e7fd60656ac40ce1822414f55bcb755300327c548f",
            "213cac6b5ebb99445451b823b679988c8cee9235efc416e9a8e16f8e84f08573"
        ]
    }
}


@contextmanager
def cpu_parent(args, parser, *, device=None, extra_devices=()):
    """Validate device choices once, hide GPUs in the parent, yield worker env."""
    selected = device or args.device or "cuda:0"
    if selected != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", selected):
        parser.error("--device must be cpu, cuda, or cuda:N.")
    index = int(selected.split(":")[-1]) if ":" in selected else 0
    canonical = f"cuda:{index}" if selected != "cpu" else "cpu"
    secondary = [getattr(args, name, None) for name in
                 ("edit_device", "text_device", "metric_device", "vlm_device")]
    for value in [*secondary, *extra_devices]:
        if value and value != "cpu" and value != canonical and not (canonical != "cpu" and value == "cuda" and index == 0):
            parser.error("All CUDA components must use --device; auxiliary devices may also be cpu.")
    memory = getattr(args, "max_memory", None)
    if memory and any(str(key) not in {str(index), "cpu"} for key in memory):
        parser.error("--max-memory may only name the GPU selected by --device (and cpu where supported).")

    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if canonical == "cpu":
        selector = ""
    elif env.get(DEVICE_ENV):
        # A benchmark child inherits an already restricted physical GPU.
        selector = visible
    elif visible is not None:
        devices = [value.strip() for value in visible.split(",") if value.strip() and value.strip() != "-1"]
        if index >= len(devices):
            parser.error("--device is outside the GPUs exposed by CUDA_VISIBLE_DEVICES.")
        selector = devices[index]
    else:
        selector = str(index)
    env["CUDA_VISIBLE_DEVICES"] = selector
    env[DEVICE_ENV] = canonical
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield env
    finally:
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible


def runtime_device(device):
    """Keep recorded device names; the worker's only CUDA device is cuda:0."""
    return "cuda:0" if os.environ.get(DEVICE_ENV) and str(device).startswith("cuda") else device


def runtime_memory(memory):
    if not memory or not os.environ.get(DEVICE_ENV):
        return memory
    return {("cpu" if str(key) == "cpu" else 0): value for key, value in memory.items()}


def run_function(factory, kwargs, *, env, cwd, cpu_threads):
    """Run a method stage with original settings and no CUDA state in its parent."""
    payload = {"factory": factory, "kwargs": kwargs, "cpu_threads": cpu_threads}
    subprocess.run([sys.executable, "-m", "benchmark.execution"], input=json.dumps(payload),
                   text=True, cwd=cwd, env=env, check=True)


def compatible_settings(case_root, requested, method):
    """Reuse a checkpoint only across the exact approved execution-only change."""
    case_root = Path(case_root)
    metadata = next((path for path in (case_root / "checkpoint/metadata.json",
                                      case_root / "training-data/dataset.json") if path.is_file()), None)
    if metadata is None:
        return requested
    stored = read_json(metadata)["settings"]
    if ({key: value for key, value in stored.items() if key != "sources"} !=
            {key: value for key, value in requested.items() if key != "sources"}):
        return requested
    old, new = stored["sources"], requested["sources"]
    migrations = SOURCE_MIGRATIONS.get(method, {})
    if old.keys() == new.keys() and all(old[key] == new[key] or
            (old[key], new[key]) == tuple(migrations.get(key, ())) for key in old):
        return stored
    return requested


def main():
    request = json.load(sys.stdin)
    import torch
    torch.set_num_threads(request["cpu_threads"])
    module, function = request["factory"].split(":")
    getattr(importlib.import_module(module), function)(**request["kwargs"])


if __name__ == "__main__":
    main()
