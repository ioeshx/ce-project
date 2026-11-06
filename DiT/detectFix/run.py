"""Sample detectFix prompts or run the shared DiT benchmarks."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.common import merge, write_json
from DiT.detectFix.method import DEFAULTS, DETECTION_MODES, SUPPORTED_MODELS, DetectFix, resolve_settings

DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = DIRECTORY.parents[1]


def main(argv=None):
    own = argparse.ArgumentParser(allow_abbrev=False, description=(
        "detectFix: detect target regions, then replace the text condition with anchors. "
        "Sampling and evaluation options follow `python -m benchmark dit`."))
    own.add_argument("action", choices=("sample", "benchmark", "evaluate"))
    own.add_argument("--model", choices=SUPPORTED_MODELS,
                     help="Default: sd35_medium; existing manifests retain their saved model")
    own.add_argument("--detection-mode", choices=DETECTION_MODES,
                     help="once (default): retain anchor conditions after detection; every_step: detect each step")
    own.add_argument("--detection-start-step", type=int,
                     help="Number of denoising steps to complete before detecting; default: 0")
    own.add_argument("--rho", type=float,
                     help="Detect when target mask area / image area is strictly above this value; default: 0")
    own.add_argument("--localization-layers", help="all (default), or comma-separated 0-based block indices")
    own.add_argument("--concept-include-prompt", action=argparse.BooleanOptionalAction, default=None,
                     help="Include prompt K/V in concept attention; default: image and concept K/V only")
    own.add_argument("--background", help="Background concept text; default: background")
    own.add_argument("--pairs", type=json.loads, help='JSON: [{"target":"nudity","anchor":"clothed"}]')
    own.add_argument("--anchor-overrides", type=json.loads, help='JSON: {"nudity":"clothed"}')
    own.add_argument("--trace", action=argparse.BooleanOptionalAction, default=None,
                     help="Save masks, latent decodes and detection statistics (default: off; enable with --trace)")
    own.add_argument("--trace-root", type=Path, help="Custom diagnostic root; default: traces beside the image setting")
    own.add_argument("--prompt", action="append", help="sample: repeat for multiple prompts")
    own.add_argument("--seed", type=int, default=42, help="sample: initial noise seed")
    own.add_argument("--run-dir", type=Path, help="sample: explicit output directory")
    own.add_argument("--result-root", type=Path, default=DIRECTORY / "results")
    own.add_argument("--cases", help="all/i2p/ip/style/object or comma-separated case IDs; default: all")
    own.add_argument("--tasks", help="Restrict benchmark tasks, e.g. i2p,mscoco")
    own.add_argument("--phase", choices=("prepare", "sample", "evaluate", "run"), default="run")
    own.add_argument("--dry-run", action="store_true", help="Print configuration and image counts without loading models")
    opts, remaining = own.parse_known_args(argv)
    opts.result_root = opts.result_root.expanduser().resolve()

    from benchmark.cli import get_config, parser
    from benchmark.data import adapter_identity, load_manifest, prepare, prompt_records
    from benchmark.dit_cases import definitions, select_cases
    from benchmark.layout import method_directory, registered_directory, run_label, timestamp, unique_directory
    from benchmark.model_profiles import resolve_scheduler

    args = parser().parse_args(["dit", *(["--model", opts.model] if opts.model else []), *remaining])
    overrides = {name: getattr(opts, name) for name in (*DEFAULTS, "pairs", "anchor_overrides")
                 if hasattr(opts, name) and getattr(opts, name) is not None}

    if args.manifest:
        if overrides or opts.prompt:
            own.error("An existing manifest fixes detectFix settings and prompts; prepare a new run to change them")
        command = ("evaluate" if opts.action == "evaluate" else "sample" if opts.action == "sample" else
                   {"sample": "sample", "run": "run", "evaluate": "evaluate"}.get(opts.phase))
        if command is None:
            own.error("--manifest selects an existing plan; choose --phase sample/evaluate/run")
        if opts.dry_run:
            manifest = load_manifest(args.manifest)
            print(f"[{command}] {args.manifest}: {len(manifest['records'])} records; model={manifest['model_name']}")
            return 0
        subprocess.run([sys.executable, "-m", "benchmark", command,
                        *(["--model", opts.model] if opts.model else []), *remaining],
                       cwd=WORKSPACE, check=True,
                       env={**os.environ, "BENCHMARK_CALLER_DIR": str(Path.cwd())})
        return 0

    if opts.action == "evaluate" or opts.phase == "evaluate":
        own.error("Evaluation requires --manifest from a prepared/sampled benchmark run")
    if opts.action == "sample" and not opts.prompt:
        own.error("sample requires --prompt; use benchmark for dataset sampling")

    args.model = opts.model or "sd35_medium"
    selected = select_cases(opts.cases or args.case_id or ("i2p" if opts.action == "sample" else "all"))
    requested_tasks = {task.strip() for task in opts.tasks.split(",") if task.strip()} if opts.tasks else None
    configs = []
    started_at = timestamp()
    for case_id in selected:
        for task in definitions()["cases"][case_id]["tasks"]:
            if requested_tasks and task not in requested_tasks:
                continue
            current = copy.copy(args)
            current.case_id, current.task = case_id, task
            config = get_config(current)
            if config["sampling"]["dtype"] == "auto":
                config["sampling"]["dtype"] = "bfloat16" if str(config["sampling"]["device"]).startswith("cuda") else "float32"
            settings = resolve_settings(config, overrides)
            layers = settings["localization_layers"]
            layer_label = layers if isinstance(layers, str) else "-".join(map(str, layers))
            config["method"] = args.method or (
                f"detectFix-{settings['detection_mode'].replace('_', '-')}-"
                f"start-{settings['detection_start_step']}-rho-{settings['rho']:g}-loc-{layer_label}-{started_at}")
            config = merge(config, {
                "image_output_dir": str(opts.result_root),
                "adapter": {
                    "kind": "plugin", "factory": "DiT.detectFix.method:DetectFix",
                    "source_files": ["method.py", "run.py", "diagnostics.py", "../queryGraft/method.py",
                                     "../queryGraft/source.py", "../queryGraft/diagnostics.py"],
                    "detect_fix": settings}})
            configs.append(config)
            if opts.action == "sample":
                break
        if opts.action == "sample" and configs:
            break
    if not configs or requested_tasks and requested_tasks - {config["task"] for config in configs}:
        own.error("Selected cases do not contain the requested tasks")

    if opts.dry_run:
        for config in configs:
            count = len(opts.prompt) if opts.action == "sample" else len(prompt_records(config)[0])
            print(f"[{config['model_name']}:{config['case_id']}:{config['task']}] {count} images; "
                  f"{json.dumps(config['adapter']['detect_fix'], ensure_ascii=False)}")
        return 0

    for config in configs:
        scheduler_source = copy.deepcopy(config)
        if config["adapter"].get("model_path"):
            scheduler_source["base_model"] = config["adapter"]["model_path"]
        resolve_scheduler(scheduler_source)
        config["sampling"]["scheduler_config"] = scheduler_source["sampling"]["scheduler_config"]

        if opts.action == "sample":
            import torch
            torch.set_num_threads(args.cpu_threads)
            directory = opts.run_dir.expanduser().resolve() if opts.run_dir else unique_directory(
                opts.result_root / config["model_name"] / "samples", config["method"])
            records = [{"sample_id": f"sample_{index:04d}", "prompt": prompt, "seed": opts.seed,
                        "image": str(directory / "images" / f"sample_{index:04d}.png")}
                       for index, prompt in enumerate(opts.prompt)]
            write_json(directory / "config.json", config)
            write_json(directory / "records.json", records)
            adapter = DetectFix(config)
            try:
                size = config["sampling"]["batch_size"]
                for start in range(0, len(records), size):
                    batch = records[start:start + size]
                    for record, image in zip(batch, adapter.generate(batch, config["sampling"])):
                        destination = Path(record["image"])
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        image.save(destination)
            finally:
                adapter.close()
            print(f"[sample] {directory}", flush=True)
            continue

        records, _ = prompt_records(config)
        root = method_directory(config["output_dir"], config["task"], config["method"], config["case_id"])
        plan = registered_directory(root / "plans", run_label(config["task"], records, config["sampling"]),
                                    {"config": config, "adapter": adapter_identity(config)})
        write_json(plan / "config.json", config, portable=True)
        manifest_path = prepare(config, plan / "manifest.json")
        print(f"[manifest] {manifest_path}", flush=True)
        if opts.phase != "prepare":
            command = [sys.executable, "-m", "benchmark", "sample" if opts.phase == "sample" else "run",
                       "--manifest", str(manifest_path), "--cpu-threads", str(args.cpu_threads)]
            if args.mode:
                command.extend(["--mode", args.mode])
            subprocess.run(command, cwd=WORKSPACE, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
