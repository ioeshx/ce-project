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
    AttentionReplacement, DEFAULTS, DIRECTORY, SUPPORTED_MODELS,
    cache_identity, read_prototypes, resolve_settings,
)
from benchmark.common import digest, merge, read_json, write_csv, write_json


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
    per_step = sum(s["grid"][0] * s["grid"][1] * (13 * concepts + 5) + 1028 for s in sites)
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


def analyze(directory, heatmaps):
    """CPU-only summaries and native-grid spatial maps from the saved online tensors."""
    import numpy as np
    from safetensors.torch import load_file
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = directory / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows, plotted = [], 0
    for metadata_path in sorted(directory.rglob("metadata.json")):
        metadata = read_json(metadata_path)
        if "prototype_means" not in metadata or not metadata["prototype_means"]:
            continue
        sample = metadata_path.parent
        sites = {s["id"]: s for s in metadata["sites"]}
        for path in sorted(sample.glob("step_*/*.safetensors")):
            values = load_file(str(path))
            step = int(path.parent.name.split("_")[1])
            site = sites[path.stem]
            change = (values["mean_squared_change"].item() if "mean_squared_change" in values else
                      (values["I_new"].float() - values["I_user"].float()).square().sum(-1).mean().item())
            for i, pair in enumerate(metadata["settings"]["pairs"]):
                rows.append({"sample": str(sample.relative_to(directory)), "step": step,
                    "timestep": metadata["schedule"]["timesteps"][step],
                    "layer": site["layer"], "branch": site["branch"], "target": pair["target"],
                    "anchor": pair["anchor"], "coverage": values[f"mask_{i}"].float().mean().item(),
                    "union_coverage": values["mask_union"].float().mean().item(),
                    "mean_d_tar": values[f"d_tar_{i}"].mean().item(),
                    "mean_d_anc": values[f"d_anc_{i}"].mean().item(),
                    "mean_margin": values[f"margin_{i}"].mean().item(),
                    "mean_squared_change": change})
                if plotted < heatmaps:
                    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                    axes[0].imshow(values[f"mask_{i}"].numpy(), vmin=0, vmax=1, interpolation="nearest")
                    axes[0].set_title("mask")
                    image = axes[1].imshow(values[f"margin_{i}"].numpy(), cmap="coolwarm", interpolation="nearest")
                    axes[1].set_title("d_anchor - d_target")
                    fig.colorbar(image, ax=axes[1])
                    fig.suptitle(f"{pair['target']} | step {step}, {site['id']}")
                    fig.tight_layout()
                    fig.savefig(output / f"spatial_{plotted:04d}.png", dpi=150)
                    plt.close(fig)
                    plotted += 1
    write_csv(output / "summary.csv", rows)
    # Average only scalar coverage across samples, keeping distinct attention branches.
    for branch in dict.fromkeys(row["branch"] for row in rows):
        selected = [row for row in rows if row["branch"] == branch]
        steps = sorted({row["step"] for row in selected})
        layers = sorted({row["layer"] for row in selected})
        sums = np.zeros((len(layers), len(steps)))
        counts = np.zeros_like(sums)
        # One union entry per sample/site/step, rather than counting concepts twice.
        unique = {(r["sample"], r["layer"], r["step"]): r for r in selected}
        for row in unique.values():
            location = layers.index(row["layer"]), steps.index(row["step"])
            sums[location] += row["union_coverage"]
            counts[location] += 1
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0),
                          vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(steps)), steps)
        ax.set_yticks(range(len(layers)), layers)
        ax.set(xlabel="step index", ylabel="block index", title=f"Union mask coverage ({branch})")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f"coverage_{branch}.png", dpi=150)
        plt.close(fig)
    print(f"[analysis] {len(rows)} rows: {output}")


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
    own.add_argument("--anchor-overrides", type=json.loads, help='JSON, e.g. {"nudity": "clothed"}')
    own.add_argument("--pairs", type=json.loads, help='JSON list: [{"target":"nudity","anchor":"clothed"}]')
    own.add_argument("--offline-seeds", help="Comma-separated prototype seeds; separate from benchmark --seeds")
    own.add_argument("--offline-seed-count", type=int)
    own.add_argument("--offline-seed-start", type=int)
    own.add_argument("--prototype-dir", type=Path, help="Reuse an explicit compatible collection (also supports subsets)")
    own.add_argument("--prototype-root", type=Path, help="Offline cache root, distinct from benchmark --cache-dir")
    own.add_argument("--result-root", type=Path, default=DIRECTORY / "results", help="Method image/diagnostic root")
    own.add_argument("--intervention", choices=("replace", "observe"))
    own.add_argument("--trace", action=argparse.BooleanOptionalAction, default=None)
    own.add_argument("--trace-root", type=Path, help="Write online traces on a separate disk")
    own.add_argument("--prompt", action="append", help="sample: repeat for multiple prompts")
    own.add_argument("--seed", type=int, default=42, help="sample: initial noise seed")
    own.add_argument("--run-dir", type=Path, help="analyze: trace root; sample: output directory")
    own.add_argument("--heatmaps", type=int, default=8, help="analyze: number of spatial map figures")
    opts, remaining = own.parse_known_args(argv)
    if opts.action == "analyze":
        if opts.run_dir is None or remaining:
            own.error("analyze requires --run-dir and accepts only method options")
        analyze(opts.run_dir.expanduser().resolve(), opts.heatmaps)
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
        config["method"] = args.method or (
            f"AttentionReplacement-{settings['intervention']}-tau-{threshold}-"
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
