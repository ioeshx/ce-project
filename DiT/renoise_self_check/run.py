"""Run the shared DiT benchmark cases with Renoise Self Check."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

# Also support `python DiT/renoise_self_check/run.py ...` from any directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DiT.renoise_self_check.method import DEFAULT_SETTINGS, SUPPORTED_MODELS, check_steps


def main(argv=None):
    own = argparse.ArgumentParser(allow_abbrev=False,
        description="Renoise Self Check; remaining options match `python -m benchmark dit`.")
    own.add_argument("--model", choices=SUPPORTED_MODELS, default="sd35_medium")
    own.add_argument("--lambda-ref", "--strength", type=float, default=DEFAULT_SETTINGS["lambda_ref"])
    own.add_argument("--rho", type=float, default=DEFAULT_SETTINGS["rho"])
    own.add_argument("--theta", type=float, default=DEFAULT_SETTINGS["theta"])
    own.add_argument("--margin", "--m", type=float, default=DEFAULT_SETTINGS["margin"])
    own.add_argument("--check-fraction", type=float, default=DEFAULT_SETTINGS["check_fraction"])
    own.add_argument("--probe-seed", type=int, default=DEFAULT_SETTINGS["probe_seed"])
    opts, remaining = own.parse_known_args(argv)
    settings = {name: getattr(opts, name) for name in DEFAULT_SETTINGS}
    from benchmark.cli import get_config, parser
    from benchmark.common import configuration, merge, write_json
    from benchmark.data import adapter_identity, load_manifest, prepare, prompt_records
    from benchmark.dit_cases import definitions, select_cases
    from benchmark.layout import method_directory, registered_directory, run_label
    from benchmark.model_profiles import resolve_scheduler

    args = parser().parse_args(["dit", "--model", opts.model, *remaining])
    selected = select_cases(args.cases or args.case_id or "all")
    tasks = set(args.tasks.split(",")) if args.tasks else None
    method = {
        "method": args.method or (f"RenoiseSelfCheck-l{opts.lambda_ref:g}-r{opts.rho:g}"
                                  f"-t{opts.theta:g}-m{opts.margin:g}-c{opts.check_fraction:g}-s{opts.probe_seed}"),
        "image_output_dir": str(Path(__file__).resolve().parent / "results"),
        "adapter": {"kind": "plugin", "factory": "DiT.renoise_self_check.method:RenoiseSelfCheck",
                    "source_files": ["method.py", "model.py", "run.py"], "self_check": settings}}

    for case_id in selected:
        for task in definitions()["cases"][case_id]["tasks"]:
            if tasks is not None and task not in tasks:
                continue
            current = copy.copy(args)
            current.task, current.case_id = task, case_id
            config = configuration(task=task, base=merge(get_config(current), method))
            records, _ = prompt_records(config)
            checks = [i + 1 for i in check_steps(config["sampling"]["steps"], settings["check_fraction"])]
            print(f"[{config['model_name']}:{case_id}:{task}] {len(records)} images; "
                  f"checkpoints={checks}; lambda={opts.lambda_ref:g}; rho={opts.rho:g}", flush=True)
            if args.dry_run:
                continue
            resolve_scheduler(config)
            root = method_directory(config["output_dir"], task, config["method"], case_id)
            plan = registered_directory(root / "plans", run_label(task, records, config["sampling"]),
                                        {"config": config, "adapter": adapter_identity(config)})
            write_json(plan / "config.json", config, portable=True)
            manifest_path = prepare(config, plan / "manifest.json")
            print(f"[manifest] {manifest_path}", flush=True)
            if args.phase in {"sample", "run"}:
                import torch
                from benchmark.adapters import sample_manifest
                torch.set_num_threads(args.cpu_threads)
                sample_manifest(load_manifest(manifest_path), args.mode or "image", manifest_path=manifest_path)
            if args.phase in {"evaluate", "run"}:
                from benchmark.evaluate import evaluate
                evaluate(manifest_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
