"""Collect prototypes, sample prompts, run the four benchmark groups, inspect traces."""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DiT.attention_replacement.method import (
    AttentionReplacement, DEFAULTS, DIRECTORY, MASK_MODES, SUPPORTED_MODELS,
    cache_identity, read_prototypes, resolve_settings,
)
from benchmark.common import digest, merge, read_json, write_json


def check_trace_space(config, collection, records):
    """Estimate the remaining trace volume once, before loading the sampling model."""
    import errno
    import shutil

    settings = config["adapter"]["attention_replacement"]
    if not settings["trace"]:
        return
    sites = [s for s in collection["sites"] if s["layer"] >= settings["layer_start"] and
             (settings["layer_end"] is None or s["layer"] < settings["layer_end"])]
    concepts = len(settings["pairs"])
    bytes_per_pixel = 13 * concepts + 5
    if settings["mask_mode"] != "layerwise":
        bytes_per_pixel += 5 * concepts + 1  # Probe masks, vote fractions, applied mask.
    per_step = sum(s["grid"][0] * s["grid"][1] * bytes_per_pixel + 1028 for s in sites)
    count = sum(not Path(record["image"]).is_file() for record in records)
    needed = count * (settings["step_end"] - settings["step_start"]) * per_step
    destination = Path(settings["trace_root"] or config["image_output_dir"])
    filesystem = destination
    while not filesystem.exists():
        filesystem = filesystem.parent
    free = shutil.disk_usage(filesystem).free
    print(f"[trace storage] distance/mask maps; "
          f"{count} remaining images; estimated {needed / 2**30:.2f} GiB; "
          f"free {free / 2**30:.2f} GiB on {filesystem}", flush=True)
    if needed > free:
        raise OSError(errno.ENOSPC, f"Trace output needs about {needed / 2**30:.2f} GiB at {destination}. "
                      "Use --trace-root on a larger disk, or --no-trace.")


def analyze(directory, heatmaps=None, output=None, workers=3, examples_only=False, step_end=None):
    """Analyze every saved trace run below directory, without sampling new images."""
    from DiT.attention_replacement.analyze import analyze_results
    return analyze_results(directory, output, workers, heatmaps,
                           examples_only=examples_only, step_end=step_end)


def main(argv=None):
    own = argparse.ArgumentParser(allow_abbrev=False, description=(
        "Attention Replacement. Other flags match `python -m benchmark dit`: "
        "--cases all/i2p/ip/style/object, --steps, --device, --offline, --phase, --dry-run, ..."))
    own.add_argument("action", choices=("collect", "sample", "benchmark", "evaluate", "analyze"))
    own.add_argument("--model", choices=SUPPORTED_MODELS, help="Default: sd35_medium; evaluate --manifest uses its saved model")
    own.add_argument("--layer-start", type=int)
    own.add_argument("--layer-end", type=int)
    own.add_argument("--step-start", type=int)
    window = own.add_mutually_exclusive_group()
    window.add_argument("--step-end", type=int)
    window.add_argument("--step-fraction", type=float)
    own.add_argument("--tau", type=float)
    own.add_argument("--tau-by-concept", type=json.loads, help='JSON, e.g. {"nudity": 10}')
    own.add_argument("--u", type=float, help="Maximum squared target distance; D_tar > u cannot trigger replacement")
    own.add_argument("--anchor-overrides", type=json.loads, help='JSON, e.g. {"nudity": "clothed"}')
    own.add_argument("--pairs", type=json.loads, help='JSON list: [{"target":"nudity","anchor":"clothed"}]')
    own.add_argument("--offline-seeds", help="Comma-separated prototype seeds; separate from benchmark --seeds")
    own.add_argument("--offline-seed-count", type=int)
    own.add_argument("--offline-seed-start", type=int)
    own.add_argument("--prototype-dir", type=Path, help="Reuse an explicit compatible collection (also supports subsets)")
    own.add_argument("--prototype-root", type=Path, help="Offline cache root, distinct from benchmark --cache-dir")
    own.add_argument("--result-root", type=Path, default=DIRECTORY / "results", help="Method image/diagnostic root")
    own.add_argument("--intervention", choices=("replace", "add", "blend", "observe"),
                     help="Masked AV edit; default: replace; observe only records diagnostics")
    own.add_argument("--strength", type=float,
                     help="Anchor coefficient for add/blend; default: 0.1; blend requires [0, 1]")
    own.add_argument("--mask-mode", help=(
        f"{', '.join(MASK_MODES)}; default: layerwise. Majority modes probe first, then edit "
        "all selected layers or only the last selected layer on a second forward"))
    own.add_argument("--trace", action=argparse.BooleanOptionalAction, default=None)
    own.add_argument("--trace-root", type=Path, help="Write online traces on a separate disk")
    own.add_argument("--prompt", action="append", help="sample: repeat for multiple prompts")
    own.add_argument("--seed", type=int, default=42, help="sample: initial noise seed")
    own.add_argument("--run-dir", type=Path, help="analyze: trace root; sample: output directory")
    own.add_argument("--heatmaps", type=int, help="analyze: sample limit per experiment; default: all samples; 0: skip")
    own.add_argument("--examples-only", action="store_true", help="analyze: supplement all sample diagnostics using existing analysis")
    own.add_argument("--analysis-dir", type=Path, default=DIRECTORY / "analysis",
                     help="analyze: output root, organized by experiment")
    own.add_argument("--analysis-workers", type=int, default=3, help="analyze: CPU worker processes")
    opts, remaining = own.parse_known_args(argv)
    if opts.action == "analyze":
        if opts.run_dir is None or remaining:
            own.error("analyze requires --run-dir and accepts only method options")
        if opts.analysis_workers < 1 or (opts.heatmaps is not None and opts.heatmaps < 0):
            own.error("analysis-workers must be positive and heatmaps must be nonnegative")
        analyze(opts.run_dir.expanduser().resolve(), opts.heatmaps,
                opts.analysis_dir.expanduser().resolve(), opts.analysis_workers,
                opts.examples_only, opts.step_end)
        return 0

    from benchmark.cli import get_config, parser
    from benchmark.data import adapter_identity, load_manifest, prepare, prompt_records
    from benchmark.dit_cases import definitions, select_cases
    from benchmark.layout import method_directory, registered_directory, run_label, timestamp, unique_directory
    from benchmark.model_profiles import resolve_scheduler

    args = parser().parse_args(["dit", "--model", opts.model or "sd35_medium", *remaining])
    if opts.action == "evaluate" and args.manifest:
        from benchmark.evaluate import evaluate
        manifest = load_manifest(args.manifest)
        args.model = opts.model
        evaluate(args.manifest, get_config(args, manifest))
        return 0
    if opts.action == "sample" and not opts.prompt:
        own.error("sample requires at least one --prompt; use benchmark for dataset sampling")
    if args.cpu_threads <= 0:
        own.error("--cpu-threads must be positive")
    overrides = {key: getattr(opts, key) for key in DEFAULTS
                 if hasattr(opts, key) and getattr(opts, key) is not None}
    for key in ("anchor_overrides", "pairs"):
        if getattr(opts, key) is not None:
            overrides[key] = getattr(opts, key)
    if opts.step_fraction is not None:
        overrides["step_end"] = None
    if opts.prototype_root:
        overrides["cache_root"] = str(opts.prototype_root)
    if opts.offline_seeds:
        overrides["offline_seeds"] = [int(s) for s in opts.offline_seeds.split(",")]
    elif opts.offline_seed_count is not None or opts.offline_seed_start is not None:
        count = opts.offline_seed_count if opts.offline_seed_count is not None else 16
        start = opts.offline_seed_start if opts.offline_seed_start is not None else 10000
        overrides["offline_seeds"] = list(range(start, start + count))
    if opts.prototype_dir:
        overrides["prototype_dir"] = str(opts.prototype_dir)

    if opts.action == "benchmark" and args.manifest:
        # Resume this interrupted run in place, allowing only diagnostic/runtime changes.
        from benchmark.data import sampling_protocol
        manifest = load_manifest(args.manifest)
        args.model = opts.model
        config = get_config(args, manifest)
        settings = resolve_settings(config, overrides)
        previous = {**copy.deepcopy(DEFAULTS), **manifest["config"]["adapter"]["attention_replacement"]}
        diagnostics = {"trace", "trace_activations", "trace_root", "cache_root", "prototype_dir"}
        same_settings = {k: v for k, v in settings.items() if k not in diagnostics} == {
            k: v for k, v in previous.items() if k not in diagnostics}
        same_run = all(config.get(k) == manifest["config"].get(k) for k in
                       ("model_name", "base_model", "task", "case_id", "targets", "method", "adapter"))
        if not same_settings or not same_run or sampling_protocol(config) != manifest["sampling"]:
            raise ValueError("Resume must preserve the model, sampling protocol and erasure parameters; "
                             "only trace and runtime options may change.")
        config["adapter"]["attention_replacement"] = settings
        _, collection, means = read_prototypes(config, settings, verify=False)
        if {p: info["digest"] for p, info in means.items()} != config["adapter"]["prototype_digests"]:
            raise ValueError("Resume requires the same prototype means as the saved run")
        if args.dry_run:
            check_trace_space(config, collection, manifest["records"])
            print(f"[resume preview] {args.manifest}; existing images will be reused")
            return 0
        import torch
        torch.set_num_threads(args.cpu_threads)
        manifest["config"] = config
        manifest["resumed_from"] = str(args.manifest.expanduser().resolve())
        manifest["previous_adapter_identity"] = manifest["adapter_identity"]
        manifest["adapter_identity"] = adapter_identity(config)
        plan = registered_directory(args.manifest.parent / "resumes", "resume",
                                    {"source": manifest["resumed_from"], "config": config,
                                     "adapter": manifest["adapter_identity"]})
        manifest_path = plan / "manifest.json"
        write_json(manifest_path, manifest, portable=True)
        print(f"[resume manifest] {manifest_path}", flush=True)
        if args.phase in {"sample", "run"}:
            from benchmark.adapters import sample_manifest
            check_trace_space(config, collection, manifest["records"])
            sample_manifest(manifest, args.mode or "both", manifest_path=manifest_path)
        if args.phase in {"evaluate", "run"}:
            from benchmark.evaluate import evaluate
            evaluate(manifest_path, config)
        return 0

    selected = select_cases(args.cases or args.case_id or "all")
    if opts.action == "sample":
        selected = selected[:1]
    tasks = set(args.tasks.split(",")) if args.tasks else None
    configs = []
    for case_id in selected:
        for task in definitions()["cases"][case_id]["tasks"]:
            if tasks is not None and task not in tasks:
                continue
            current = copy.copy(args)
            current.task, current.case_id = task, case_id
            config = get_config(current)
            if config["sampling"]["dtype"] == "auto":
                config["sampling"]["dtype"] = ("bfloat16" if str(config["sampling"]["device"]).startswith("cuda")
                                               else "float32")
            settings = resolve_settings(config, overrides)
            config = merge(config, {
                "image_output_dir": str(opts.result_root.expanduser().resolve()),
                "adapter": {"kind": "plugin", "factory": "DiT.attention_replacement.method:AttentionReplacement",
                            "source_files": ["method.py", "run.py"], "attention_replacement": settings}})
            configs.append(config)
            if opts.action == "sample":
                break
    if not configs:
        own.error("case/task selection is empty")

    if args.dry_run:
        for config in configs:
            settings = config["adapter"]["attention_replacement"]
            count = len(opts.prompt) if opts.action == "sample" else len(prompt_records(config)[0])
            print(f"[{config['model_name']}:{config['case_id']}:{config['task']}] {count} images; "
                  f"steps=[{settings['step_start']},{settings['step_end']}); "
                  f"layers=[{settings['layer_start']},{settings['layer_end'] or 'all'}); "
                  f"mask_mode={settings['mask_mode']}; "
                  f"intervention={settings['intervention']}; strength={settings['strength']:g}; "
                  f"N={len(settings['offline_seeds'])}; pairs={settings['pairs']}")
        return 0

    import torch
    torch.set_num_threads(args.cpu_threads)
    for config in configs:
        # A local Diffusers directory supplies its own scheduler config, too.
        scheduler_source = copy.deepcopy(config)
        if config["adapter"].get("model_path"):
            scheduler_source["base_model"] = config["adapter"]["model_path"]
        resolve_scheduler(scheduler_source)
        config["sampling"]["scheduler_config"] = scheduler_source["sampling"]["scheduler_config"]

    if opts.action == "collect":
        # Group cases by physical cache identity; shared anchors are collected once.
        groups = {}
        for config in configs:
            settings = config["adapter"]["attention_replacement"]
            key = (settings["prototype_dir"] or settings["cache_root"], digest(cache_identity(config, settings)))
            group = groups.setdefault(key, {"config": config, "prompts": []})
            group["prompts"].extend(p[key] for p in settings["pairs"] for key in ("target", "anchor"))
        for group in groups.values():
            group["config"]["adapter"]["attention_replacement"]["collect"] = True
            adapter = AttentionReplacement(group["config"])
            try:
                adapter.collect(group["prompts"])
            finally:
                adapter.close()
            print(f"[prototype collection] {adapter.cache_dir}")
        return 0

    started_at = timestamp()
    for config in configs:
        settings = config["adapter"]["attention_replacement"]
        directory, metadata, means = read_prototypes(config, settings,
            verify=opts.action == "evaluate" or args.phase in {"prepare", "evaluate"})
        settings["prototype_dir"] = str(directory)
        settings["offline_seeds"] = metadata["identity"]["capture"]["offline_seeds"]
        config["adapter"]["prototype_digests"] = {p: info["digest"] for p, info in means.items()}
        threshold = "per-concept" if settings["tau_by_concept"] else f"{settings['tau']:g}"
        distance_limit = "off" if settings["u"] is None else f"{settings['u']:g}"
        intervention_label = settings["intervention"]
        if intervention_label in {"add", "blend"}:
            intervention_label += f"-strength-{settings['strength']:g}"
        if settings["mask_mode"] != "layerwise":
            intervention_label += f"-mask-{settings['mask_mode'].replace('_', '-')}"
        config["method"] = args.method or (
            f"AttentionReplacement-{intervention_label}-tau-{threshold}-u-{distance_limit}-"
            f"layers-{settings['layer_start']}-{settings['layer_end'] or 'all'}-"
            f"steps-{settings['step_start']}-{settings['step_end']}-{started_at}")
        if opts.action == "sample":
            directory = (opts.run_dir.expanduser().resolve() if opts.run_dir else
                         unique_directory(opts.result_root / "diagnostics", timestamp()))
            records = [{"sample_id": f"sample_{i:04d}", "prompt": prompt, "seed": opts.seed,
                        "image": str(directory / "images" / "generated" / f"sample_{i:04d}.png")}
                       for i, prompt in enumerate(opts.prompt)]
            write_json(directory / "config.json", config)
            check_trace_space(config, metadata, records)
            adapter = AttentionReplacement(config)
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
            write_json(directory / "records.json", records)
            print(f"[sample] {directory}")
            continue

        task, case_id = config["task"], config["case_id"]
        records, _ = prompt_records(config)
        root = method_directory(config["output_dir"], task, config["method"], case_id)
        plan = registered_directory(root / "plans", run_label(task, records, config["sampling"]),
                                    {"config": config, "adapter": adapter_identity(config)})
        write_json(plan / "config.json", config, portable=True)
        manifest_path = prepare(config, plan / "manifest.json")
        print(f"[manifest] {manifest_path}", flush=True)
        if opts.action == "benchmark" and args.phase in {"sample", "run"}:
            from benchmark.adapters import sample_manifest
            manifest = load_manifest(manifest_path)
            check_trace_space(config, metadata, manifest["records"])
            sample_manifest(manifest, args.mode or "both", manifest_path=manifest_path)
        if opts.action == "evaluate" or args.phase in {"evaluate", "run"}:
            from benchmark.evaluate import evaluate
            evaluate(manifest_path, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
