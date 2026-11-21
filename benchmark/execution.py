"""CPU orchestration and CUDA placement for short-lived benchmark workers."""
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

# Known source revisions for execution-only changes, ending with the current hash.
SOURCE_MIGRATIONS = {
    "unified-concept-editing": {
        "dit.py": [
            "90a51e01f32c5fba5d44a7d1b3fbaaf7bae5690a8f453317c70e9f0af4def191",
            "659d0a3456e885ea3f493857187896d257b5c2153b54b640973db2325b52665a",
            "95b49c400e355543e404476dce93804d4fe57811542e51bd36d2a11f13e5139d"
        ],
        "dit_editing.py": [
            "fbb30c70ce75f992ad4cc7c8f65c7e3d34c838c972bde459d6454acf5bea0ecf",
            "4b4221ce66a54295e5b0ee5885ea7e75bef4eff79304eb4b65a94ce6e604399d"
        ]
    },
    "ESD": {
        "esd_benchmark/dit.py": [
            "dd8bf25ac4e4eb6467ceab7e993ff62c66759f1e5d9e09dea5541eb71f9105ea",
            "5aacb058190bf9f45a9e7cddd2898d9364568e7441ccc812a0cf4c674dfd8bc4",
            "e6e3e5436b0564184cb7d67de9d36b3e273bd88659245af5ff9ea2a86215e559",
            "80241480cfb770d4ac4d9ad6dc81f3442409b08aafc75d2d3b569efba5174e9a"
        ],
        "esd_benchmark/dit_training.py": [
            "ecf785857fddb7370da44546852fe9014ca22b4efbda17a895cea6c37859e9b5",
            "b4d18a1776d1348a32c4dec4210abd2866cc0de70954db5bd279f6ca4eff317a",
            "890115093dce2171d1f1c5cdb1a0a795f7d7b4d7ad93479209a74910abd74885"
        ],
        "utils/esd_trainer.py": [
            "5b76dd2ba4cfad0064bfc8c2c049e57968ad146db0b26c406ad2bf75a56e7a60",
            "21b141b4e3a3769bd5ba07b0f0fb18413719bb40ba70a2d3c91edd7c739fd1fa"
        ]
    },
    "EraseAnything": {
        "benchmark_dit.py": [
            "11adc4878a51e5bdd043be49f390a10d617adfdd1251cfe3ec617f1100adfe09",
            "31e6b821e28c244c61b8e3dc4c23bee38cf9b9e628bf69f33c5f2243ba499421",
            "72957ed3b143fdbfc8ad837134d779138bb728be3703839ce257508f1a62e060"
        ],
        "benchmark_training.py": [
            "acf96d0110913d6228673b43c0a4dfebe8354106bd53fdc477405b18b1d840e6",
            "b726ad98618a9945d3d29483c4e1c0b7c374337e3bd63035f553d30b7aee04ae",
            "08f2cd794a542151636ec21c7b097073f6d5d898e57a936b711fa21ed741a087"
        ],
        "utils/calc_loss.py": [
            "6310d497926024a7d7fa19720b55868ecef76ef3ec2e528761cbf741911378da",
            "6fc658424d986a1cd1bd17a417fb7c50c4a0dd0f293ce635860f6bf486ecd5db"
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
def cpu_parent(args, parser, *, device=None, extra_devices=(), shard_flux=False):
    """Validate device choices once, hide GPUs in the parent, yield worker env."""
    selected = device or args.device or "cuda:0"
    if selected != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", selected):
        parser.error("--device must be cpu, cuda, or cuda:N.")
    index = int(selected.split(":")[-1]) if ":" in selected else 0
    canonical = f"cuda:{index}" if selected != "cpu" else "cpu"
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = [value.strip() for value in (visible if visible is not None else "0,1").split(",")
               if value.strip() not in {"", "-1"}]
    sharded = shard_flux and canonical == "cuda:0" and len(devices) >= 2
    if sharded and devices[0] == devices[1]:
        parser.error("FLUX sharding needs two distinct visible GPUs.")
    secondary = [getattr(args, name, None) for name in
                 ("edit_device", "text_device", "metric_device", "vlm_device")]
    for value in [*secondary, *extra_devices]:
        if value and value != "cpu" and value != canonical and not (canonical != "cpu" and value == "cuda" and index == 0):
            parser.error("All CUDA components must use --device; auxiliary devices may also be cpu.")
    memory = getattr(args, "max_memory", None)
    memory_devices = {"0", "1"} if sharded else {str(index), "cpu"}
    if memory and any(str(key) not in memory_devices for key in memory):
        parser.error(f"--max-memory keys must be selected from {', '.join(sorted(memory_devices))}.")

    if canonical == "cpu":
        selector = ""
    elif env.get(DEVICE_ENV):
        # A nested benchmark inherits the worker's visible GPUs.
        selector = visible
    elif sharded:
        selector = ",".join(devices[:2])
    elif visible is not None:
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
    """Keep recorded device names; the worker's primary CUDA device is cuda:0."""
    return "cuda:0" if os.environ.get(DEVICE_ENV) and str(device).startswith("cuda") else device


def runtime_memory(memory):
    if not memory or not os.environ.get(DEVICE_ENV) or "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        return memory
    return {("cpu" if str(key) == "cpu" else 0): value for key, value in memory.items()}


def run_function(factory, kwargs, *, env, cwd, cpu_threads):
    """Run a method stage with original settings and no CUDA state in its parent."""
    payload = {"factory": factory, "kwargs": kwargs, "cpu_threads": cpu_threads}
    subprocess.run([sys.executable, "-m", "benchmark.execution"], input=json.dumps(payload),
                   text=True, cwd=cwd, env=env, check=True)


def compatible_source(stored, current, revisions):
    return stored == current or bool(revisions and current == revisions[-1] and stored in revisions[:-1])


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
    if old.keys() == new.keys() and all(
            compatible_source(old[key], new[key], migrations.get(key, ())) for key in old):
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
