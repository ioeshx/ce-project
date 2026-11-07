"""Sample queryGraft prompts and run the shared DiT benchmark protocols."""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"


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
from DiT.queryGraft.method import DEFAULTS, MASK_RULES, REPRESENTATIONS, SOURCE_MODES, SUPPORTED_MODELS, UPDATE_RULES, QueryGraft, resolve_settings

DIRECTORY = Path(__file__).resolve().parent
WORKSPACE = DIRECTORY.parents[1]


def main(argv=None):
    own = argparse.ArgumentParser(allow_abbrev=False, description=(
        "queryGraft. Additional options follow `python -m benchmark dit`: "
        "--cases all/i2p/ip/style/object, --phase, --steps, --device, --offline, ..."))
    own.add_argument("action", choices=("sample", "benchmark", "evaluate"))
    own.add_argument("--model", choices=SUPPORTED_MODELS,
                     help="Default: sd35_medium; existing manifests retain their saved model")
    own.add_argument("--localization-layers", help="all (default), or comma-separated 0-based block indices")
    own.add_argument("--edit-layers", help="all (default), last, or comma-separated 0-based block indices; velocity defaults to/requires last")
    own.add_argument("--mask-rule", choices=MASK_RULES,
                     help="With --my-layer-policy: target_max (default) requires target > anchor and background; "
                          "anchor_min requires anchor < target and background; mean_target_max additionally "
                          "requires target > its spatial mean on soft maps; all comparisons are strict")
    own.add_argument("--my-layer-aggre", action=argparse.BooleanOptionalAction, default=None,
                     help="Enable binary per-layer masks and vote_fraction > eta; default: average soft maps")
    own.add_argument("--my-layer-policy", action=argparse.BooleanOptionalAction, default=None,
                     help="Enable the target/anchor/background --mask-rule; default: single-target spatial mean "
                          "threshold or multi-target global concept argmax")
    own.add_argument("--concept-include-prompt", action=argparse.BooleanOptionalAction, default=None,
                     help="Include user prompt K/V in concept attention (default: off; concepts read only "
                          "image tokens and their own target/anchor/background group)")
    own.add_argument("--representation", choices=REPRESENTATIONS,
                     help="text_output (default): text-value contribution after the output projection, with user replay; "
                          "av: before output projection; residual: after attention residual; "
                          "block_output: final block output after MLP and residual updates; "
                          "velocity: final Transformer prediction after the output head, before CFG")
    own.add_argument("--eta", type=float,
                     help="With --my-layer-aggre: strict layer-vote threshold; default: 0 (any supporting layer)")
    own.add_argument("--rho", type=float, help="Strict target-area threshold; default: 0 (any target token)")
    own.add_argument("--update-rule", choices=UPDATE_RULES,
                     help="Default: replace; zero sets masked features to 0 without a source forward "
                          "and ignores strength/source options")
    own.add_argument("--strength", type=float, help="Anchor coefficient for blend/add; default: 1")
    own.add_argument("--source-mode", choices=SOURCE_MODES,
                     help="Default: anchor (original); joint, logits, values, extrapolate use user prompt + anchor")
    own.add_argument("--source-user-weight", type=float,
                     help="logits/values: user text attention/value weight in masked source image rows; default: 1")
    own.add_argument("--source-anchor-weight", type=float,
                     help="logits/values: appended anchor attention/value weight; default: 1")
    own.add_argument("--source-scale", type=float,
                     help="extrapolate: H_user + scale * (H_joint - H_user), using velocity for that representation; default: 1")
    own.add_argument("--background", help="Background concept text; default: background")
    own.add_argument("--pairs", type=json.loads, help='JSON: [{"target":"nudity","anchor":"clothed"}]')
    own.add_argument("--anchor-overrides", type=json.loads, help='JSON: {"nudity":"clothed"}')
    own.add_argument("--trace", action=argparse.BooleanOptionalAction, default=None,
                     help="Save per-step PNGs, maps and activation statistics plus sample temporal summaries "
                          "(default: off; enable with --trace)")
    own.add_argument("--trace-root", type=Path, help="Root for readable per-model/case/run/sample diagnostic directories")
    own.add_argument("--result-root", type=Path, default=DIRECTORY / "results")
    own.add_argument("--prompt", action="append", help="sample: repeat for multiple prompts")
    own.add_argument("--seed", type=int, default=42, help="sample: initial noise seed")
    own.add_argument("--run-dir", type=Path, help="sample: explicit output directory")
    own.add_argument("--cases", help="all/i2p/ip/style/object or comma-separated case IDs; default: all")
    own.add_argument("--tasks", help="Restrict benchmark tasks, e.g. i2p,mscoco")
    own.add_argument("--phase", choices=("prepare", "sample", "evaluate", "run"), default="run")
    own.add_argument("--dry-run", action="store_true", help="Print configuration and sample counts without model loading")
    opts, remaining = own.parse_known_args(argv)
    opts.result_root = opts.result_root.expanduser().resolve()

    from benchmark.cli import get_config, parser
    from benchmark.data import adapter_identity, load_manifest, prepare, prompt_records
    from benchmark.dit_cases import definitions, select_cases
    from benchmark.layout import method_directory, registered_directory, run_label, timestamp, unique_directory
    from benchmark.model_profiles import resolve_scheduler

    args = parser().parse_args(["dit", *(["--model", opts.model] if opts.model else []), *remaining])
    overrides = {name: getattr(opts, name) for name in DEFAULTS
                 if hasattr(opts, name) and getattr(opts, name) is not None}
    for name in ("pairs", "anchor_overrides"):
        if getattr(opts, name) is not None:
            overrides[name] = getattr(opts, name)
    for name in ("localization_layers", "edit_layers"):
        if name in overrides and overrides[name] not in {"all", "last"}:
            overrides[name] = [int(index) for index in overrides[name].split(",")]
    if opts.trace_root:
        overrides["trace_root"] = str(opts.trace_root.expanduser().resolve())

    # Existing manifests use the benchmark's own resume and evaluation path.
    if args.manifest:
        if overrides or opts.prompt:
            own.error("An existing manifest fixes queryGraft settings and prompts; prepare a new run to change them")
        command = ("evaluate" if opts.action == "evaluate" else "sample" if opts.action == "sample" else
                   {"sample": "sample", "run": "run", "evaluate": "evaluate"}.get(opts.phase))
        if command is None:
            own.error("--manifest selects an existing plan; choose --phase sample/evaluate/run")
        if opts.dry_run:
            manifest = load_manifest(args.manifest)
            print(f"[{command}] {args.manifest}: {len(manifest['records'])} records; model={manifest['model_name']}")
            return 0
        command_args = [sys.executable, "-m", "benchmark", command,
                        *(["--model", opts.model] if opts.model else []), *remaining]
        subprocess.run(command_args, cwd=WORKSPACE, check=True,
                       env={**os.environ, "BENCHMARK_CALLER_DIR": str(Path.cwd())})
        return 0

    if opts.action == "evaluate" or opts.phase == "evaluate":
        own.error("Evaluation requires --manifest from a prepared/sampled benchmark run")
    if opts.action == "sample" and not opts.prompt:
        own.error("sample requires --prompt; use benchmark for dataset sampling")
    if args.cpu_threads <= 0 or not 0 <= opts.seed < 2**63:
        own.error("cpu-threads must be positive and seed must be in [0, 2**63)")

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
            local_layers = settings["localization_layers"]
            edit_layers = settings["edit_layers"]
            local_label = (local_layers if isinstance(local_layers, str) else
                           "-".join(map(str, local_layers)) if len(local_layers) <= 5 else f"{len(local_layers)}-selected")
            edit_label = (edit_layers if isinstance(edit_layers, str) else
                          "-".join(map(str, edit_layers)) if len(edit_layers) <= 5 else f"{len(edit_layers)}-selected")
            update_label = settings["update_rule"]
            if update_label in {"blend", "add"}:
                update_label += f"-{settings['strength']:g}"
            aggregation_label = "layer-vote-" if settings["my_layer_aggre"] else "layer-soft-mean-"
            policy_label = f"my-policy-mask-{settings['mask_rule']}-" if settings["my_layer_policy"] else "ca-policy-"
            concept_label = "concept-image-prompt-" if settings["concept_include_prompt"] else "concept-image-"
            source_label = ""
            if settings["update_rule"] != "zero" and settings["source_mode"] != "anchor":
                source_label = f"source-{settings['source_mode']}-"
                if settings["source_mode"] in {"logits", "values"}:
                    source_label += f"u-{settings['source_user_weight']:g}-a-{settings['source_anchor_weight']:g}-"
                elif settings["source_mode"] == "extrapolate":
                    source_label += f"scale-{settings['source_scale']:g}-"
            config["method"] = args.method or (
                f"queryGraft-concept-dot-zero-pooled-{settings['representation']}-{update_label}-"
                f"{concept_label}{aggregation_label}{policy_label}{source_label}"
                f"eta-{settings['eta']:g}-rho-{settings['rho']:g}-"
                f"loc-{local_label}-edit-{edit_label}-{started_at}")
            config = merge(config, {"image_output_dir": str(opts.result_root.expanduser().resolve()),
                "adapter": {"kind": "plugin", "factory": "DiT.queryGraft.method:QueryGraft",
                            "source_files": ["method.py", "run.py", "diagnostics.py", "source.py"], "query_graft": settings}})
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
                  f"{json.dumps(config['adapter']['query_graft'], ensure_ascii=False)}")
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
                opts.result_root / config["model_name"] / "diagnostics", config["method"])
            records = [{"sample_id": f"sample_{index:04d}", "prompt": prompt, "seed": opts.seed,
                        "image": str(directory / "images" / "generated" / f"sample_{index:04d}.png")}
                       for index, prompt in enumerate(opts.prompt)]
            write_json(directory / "config.json", config)
            write_json(directory / "records.json", records)
            adapter = QueryGraft(config)
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
