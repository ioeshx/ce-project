"""Pair the SAME NSFW-erased checkpoint/intervention on I2P and COCO 1K."""
from __future__ import annotations

import copy
from pathlib import Path
import shlex
import subprocess
import sys

from .common import read_json, write_json
from .data import adapter_identity
from .layout import registered_directory, sampling_label, slug
from .model_profiles import DIT_MODELS, resolve_scheduler


def plans(args):
    from .cli import get_config
    if args.model not in DIT_MODELS:
        raise ValueError("dit requires --model flux1_dev/flux1_schnell/sd30_medium/sd35_medium.")
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - {"i2p", "mscoco"}:
        raise ValueError("DiT tasks must be i2p and/or mscoco, without duplicates.")
    if args.manifest or args.import_casu:
        raise ValueError("Use sample/evaluate --manifest for an existing plan; dit creates paired task plans.")
    output = []
    for task in tasks:
        selected = copy.copy(args)
        selected.task = task
        config = get_config(selected)
        if not config["targets"]:
            config["targets"] = ["nudity"]
        # Child phases run from the workspace, so preserve the caller's local
        # paths as absolute paths before storing the immutable plan.
        for container, keys in ((config["adapter"], ("checkpoint", "model_path")),
                                (config["dataset"], ("path",)),
                                (config["evaluation"], ("q16_prompts", "q16_clip_path", "nudenet_weights")),
                                (config, ("base_model",))):
            for key in keys:
                if container.get(key) and Path(container[key]).expanduser().exists():
                    container[key] = str(Path(container[key]).expanduser().resolve())
        if task == "mscoco" and config["dataset"]["limit_prompts"] is None:
            config["dataset"]["limit_prompts"] = 1000
        if task == "mscoco" and config["dataset"]["limit_prompts"] > 1000:
            raise ValueError("This retention experiment uses at most the first 1,000 COCO captions.")
        config["experiment"] = {"name": "i2p-erasure-and-coco-retention", "tasks": ["i2p", "mscoco"],
                                "erasure_targets": config["targets"],
                                "note": "External NSFW-erased checkpoint/intervention; benchmark does not implement erasure."}
        base = Path(config["output_dir"]) / "plans" / slug(config["method"])
        label = f"{task}-{sampling_label(config['sampling'])}"
        if args.dry_run:
            directory = base / f"{label}-planned"
        else:
            resolve_scheduler(config)
            identity = {"config": config, "adapter": adapter_identity(config)}
            if args.phase in {"sample", "evaluate"}:
                directory = next((p for p in sorted(base.iterdir()) if (p / "run.json").is_file()
                                  and read_json(p / "run.json")["identity"] == identity), None) if base.exists() else None
                if directory is None:
                    raise ValueError(f"No matching prepared {task} plan; run --phase prepare first.")
            else:
                directory = registered_directory(base, label, identity)
        path, manifest = directory / f"{task}.json", directory / f"{task}-manifest.json"
        if not args.dry_run:
            if not path.exists():
                write_json(path, config)
            if args.phase in {"sample", "evaluate"} and not manifest.exists():
                raise ValueError(f"Missing manifest {manifest}; run --phase prepare first.")
        output.append((task, path, manifest))
    return output


def run(args):
    for task, config, manifest in plans(args):
        commands = []
        prefix = [sys.executable, "-m", "benchmark"]
        if args.phase in {"prepare", "run"}:
            commands.append([*prefix, "prepare", "--config", str(config), "--manifest", str(manifest)])
        if args.phase in {"sample", "run"}:
            commands.append([*prefix, "sample", "--manifest", str(manifest), "--mode",
                             args.mode or ("image" if task == "i2p" else "both"), "--cpu-threads", str(args.cpu_threads)])
        if args.phase in {"evaluate", "run"}:
            commands.append([*prefix, "evaluate", "--manifest", str(manifest), "--cpu-threads", str(args.cpu_threads)])
        print(f"[DiT:{args.model}:{task}] {manifest}", flush=True)
        for command in commands:
            print("[command] " + shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, check=True)
