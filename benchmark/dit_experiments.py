"""Compile method-independent DiT cases, then run prepare/sample/evaluate phases."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .common import read_json, write_json
from .data import adapter_identity
from .dit_cases import definitions, select_cases
from .layout import registered_directory, sampling_label, slug
from .model_profiles import DIT_MODELS, resolve_scheduler


def plans(args):
    from .cli import get_config
    if args.model not in DIT_MODELS:
        raise ValueError("dit requires --model flux1_dev/flux1_schnell/sd30_medium/sd35_medium.")
    if args.manifest or args.import_casu:
        raise ValueError("Use sample/evaluate --manifest for an existing plan; dit creates case plans.")
    cases = select_cases(getattr(args, "cases", None) or getattr(args, "case_id", None))
    requested_tasks = [s.strip() for s in args.tasks.split(",") if s.strip()] if args.tasks else None
    if requested_tasks and (len(requested_tasks) != len(set(requested_tasks)) or set(requested_tasks) - {"i2p", "mscoco", "instance", "style", "object"}):
        raise ValueError("Choose unique tasks from i2p,mscoco,instance,style,object.")
    output = []
    found_tasks = set()
    for case_id in cases:
        for task in definitions()["cases"][case_id]["tasks"]:
            if requested_tasks and task not in requested_tasks:
                continue
            found_tasks.add(task)
            selected = copy.copy(args)
            selected.task, selected.case_id = task, case_id
            if task != "i2p":
                selected.reuse_manifest = None  # Paired-run reuse input is the old I2P manifest.
            config = get_config(selected)
            if config.get("case_id") != case_id:
                raise ValueError("--config case_id conflicts with selected --cases; use task-neutral overrides.")
            if task == "mscoco" and (config["dataset"]["limit_prompts"] is None or config["dataset"]["limit_prompts"] > 1000):
                raise ValueError("DiT COCO retention uses at most the first 1,000 captions.")
            if len(cases) > 1 and config["adapter"]["kind"] in {"transformer", "lora", "diffusers"}:
                if not getattr(args, "method_config", None):
                    raise ValueError("Multiple weight-erasure cases need --method-config with a per-case checkpoint/model_path mapping.")
                from .common import read_json as read
                overrides = read(args.method_config).get("cases", {}).get(case_id, {}).get("adapter", {})
                field = "model_path" if config["adapter"]["kind"] == "diffusers" else "checkpoint"
                if not overrides.get(field):
                    raise ValueError(f"Missing explicit checkpoint mapping for {case_id} in --method-config.")
            base = Path(config["output_dir"]) / "plans" / slug(config["method"]) / case_id
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
                        raise ValueError(f"No matching prepared {case_id}/{task} plan; run --phase prepare first.")
                else:
                    directory = registered_directory(base, label, identity)
            path, manifest = directory / f"{task}.json", directory / f"{task}-manifest.json"
            if not args.dry_run:
                if not path.exists():
                    write_json(path, config, portable=True)
                if args.phase in {"sample", "evaluate"} and not manifest.exists():
                    raise ValueError(f"Missing manifest {manifest}; run --phase prepare first.")
            output.append((task, path, manifest))
    if not output or requested_tasks and set(requested_tasks) - found_tasks:
        raise ValueError("Selected cases do not define all requested tasks.")
    return output


def run(args):
    for task, config, manifest in plans(args):
        commands = []
        workspace = Path(__file__).resolve().parent.parent
        config, manifest = (Path(os.path.relpath(path, workspace)) for path in (config, manifest))
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
                child_env = dict(os.environ)
                child_env.pop("BENCHMARK_CALLER_DIR", None)
                subprocess.run(command, cwd=workspace, env=child_env, check=True)
