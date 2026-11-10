"""Fit steering vectors and run I2P, IP, style and Object via the shared benchmark."""
from __future__ import annotations
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"


import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE))

from benchmark.common import ROOT, configuration, file_digest, locked, read_json, slug, write_json
from benchmark.cli import SUBSET_EVAL_DEFAULTS, add_subset_eval_arguments, subset_eval_dataset
from benchmark.data import prompt_records
from benchmark.dit_cases import definitions, select_cases
from benchmark.execution import cpu_parent, run_function
from benchmark.layout import registered_directory, sampling_label
from benchmark.model_profiles import profile, resolve_scheduler
from benchmark.plugin_plans import plugin_plan
from DiT.steer.method import prompt_pairs


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model", choices=("sd35_medium", "sd30_medium"), default="sd35_medium")
    p.add_argument("--cases", default="all", help="all/i2p/ip/style/object, or comma-separated case IDs")
    p.add_argument("--tasks", help="Optional filter: i2p,mscoco,instance,style,object")
    p.add_argument("--phase", choices=("all", "fit", "prepare", "sample", "evaluate"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--beta", type=float, default=-2.0, help="Signed coefficient: negative erases, positive amplifies")
    p.add_argument("--clipping", action=argparse.BooleanOptionalAction, default=True,
                   help="Only steer positive dot products (CASteer intermediate clipping)")
    p.add_argument("--no-score-gate", dest="score_gate", action="store_false",
                   help="Add beta * vector directly, without computing a dot-product score")
    p.add_argument("--steer-noise", choices=("all", "every"),
                   help="Replace offline directions: all=fixed noise per evaluation concept; every=redraw at each steering call")
    p.add_argument("--trace-score", action=argparse.BooleanOptionalAction, default=False,
                   help="Save raw score maps averaged by evaluation concept (default: off; requires score gating)")
    p.add_argument("--steer-target", choices=("text_value", "text_to_image", "attention_residual", "mlp_output", "block_output"),
                   default="text_to_image", help="Steering site: text Value, text-to-image contribution, attention residual, raw MLP, or full block output")
    p.add_argument("--steer-token", choices=("image", "text", "all"), default="image",
                   help="Tokens used by attention_residual/mlp_output/block_output (default: image)")
    p.add_argument("--steer-pool", action="store_true",
                   help="Also steer the global pooled text embedding, before its projection and timestep addition")
    p.add_argument("--average-patches", action=argparse.BooleanOptionalAction, default=True,
                   help="Average positions (default); --no-average-patches keeps a direction per position")
    p.add_argument("--vector-method", choices=("mean", "robust", "pca", "pca-filter"), default="mean",
                   help="Offline aggregation: mean, median of means, first uncentered PC, or rank-k filtered mean")
    p.add_argument("--normalize-differences", action="store_true",
                   help="Normalize each prompt-pair difference before aggregation/PCA")
    p.add_argument("--robust-groups", type=int, help="Groups for robust median of means; default min(5, pair count)")
    p.add_argument("--robust-max-iter", type=int, default=100, help="Geometric median iteration limit")
    p.add_argument("--robust-tol", type=float, default=1e-5, help="Geometric median relative stopping tolerance")
    p.add_argument("--pca-rank", type=int, default=4, help="Number of components retained by pca-filter")
    p.add_argument("--prompt-mode", choices=("casteer", "bare", "subject", "matched"), default="casteer",
                   help="Offline prompts: CASteer contexts (default), literal concepts, short subject wrappers, or shared scene content")
    p.add_argument("--calibration-prompts", type=int,
                   help="Use the first N pairs; defaults: casteer human=210/other=50, bare=1, subject/matched=8")
    p.add_argument("--use-anchor", action="store_true",
                   help="Append configured anchors to offline negative prompts; default: context only, except nudity -> clothed")
    p.add_argument("--target-anchor-config", type=Path,
                   help="Anchor JSON used with --use-anchor (flat mapping or concept_pair); default: benchmark/configs/dit_target_anchor.json")
    p.add_argument("--seed", type=int, default=0, help="Offline pair seed; benchmark uses each record's seed")
    p.add_argument("--run-root", type=Path, default=DIRECTORY / "runs")
    p.add_argument("--method", help="Result name; default includes beta and clipping")
    p.add_argument("--base-model", help="Optional local Diffusers pipeline or matching Hub ID")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    p.add_argument("--steps", type=int, help="Shared offline and benchmark denoising step count")
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--max-sequence-length", type=int)
    p.add_argument("--metric-device")
    p.add_argument("--metric-batch-size", type=int, default=32)
    p.add_argument("--with-q16", action="store_true")
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--vlm-model")
    p.add_argument("--vlm-device")
    p.add_argument("--resnet-weights-path", type=Path)
    p.add_argument("--q16-prompts", type=Path)
    p.add_argument("--q16-clip-path", type=Path)
    p.add_argument("--nudenet-weights", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--image-output-dir", type=Path, default=DIRECTORY / "results",
                   help="Generated images; metrics remain under --output-dir")
    p.add_argument("--original-dir", type=Path)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--max-records", type=int, help="Small validation subset, not the full benchmark")
    add_subset_eval_arguments(p)
    return p


def load_target_anchors(path, targets):
    """Read offline anchors, accepting the local conf.json and benchmark formats."""
    value = read_json(path)
    anchors = value.get("concept_pair", value) if isinstance(value, dict) else value
    if not isinstance(anchors, dict):
        raise ValueError(f"{path}: expected a target-to-anchor mapping or a concept_pair mapping.")
    for target in targets:
        if target not in anchors:
            raise ValueError(f"{path}: missing anchor for target {target!r}.")
        if not isinstance(anchors[target], str):
            raise ValueError(f"{path}: anchor for {target!r} must be a string (empty string is allowed).")
    return anchors


def case_configs(args, case_id, case):
    """Resolve effective offline anchors, preserving benchmark targets and protocol."""
    selected = profile(args.model)
    anchors = (load_target_anchors(args.target_anchor_config or ROOT / "configs/dit_target_anchor.json", case["targets"])
               if args.use_anchor else
               {target: "clothed" if target == "nudity" else "" for target in case["targets"]})
    source_files = ["method.py", "imagenet_classes.txt"]
    if args.prompt_mode in {"subject", "matched"}:
        source_files.append(f"prompts/{args.prompt_mode}.json")
    variant = args.variant
    checkpoint = args.run_root / args.model / case_id / variant / "checkpoint"
    settings = {
        "implementation": "steer-sd3-mean-difference-v2", "model_name": args.model,
        "steer_target": args.steer_target, "average_patches": args.average_patches,
        "steer_token": args.steer_token, "steer_pool": args.steer_pool,
        "construction": args.construction,
        "base_model": args.base_model or selected["base_model"], "model_revision": args.model_revision,
        "group": case["group"], "targets": case["targets"],
        "anchors": [anchors[target] for target in case["targets"]],
        "seed": args.seed, "calibration_prompts": args.calibration_prompts, "prompt_mode": args.prompt_mode,
        "sources": {name: file_digest(DIRECTORY / name) for name in source_files}}
    configs = []
    for task in case["tasks"]:
        if args.tasks and task not in args.tasks:
            continue
        cfg = {
            "model_name": args.model, "case_id": case_id, "method": args.method,
            "base_model": settings["base_model"], "output_dir": str(args.output_dir),
            "image_output_dir": str(args.image_output_dir),
            "cache_dir": str(args.cache_dir),
            "original_dir": str(args.original_dir or ROOT / "original" / args.model),
            "adapter": {"kind": "plugin", "factory": "DiT.steer.method:SteerAdapter",
                        "source_files": ["method.py", "run.py", "diagnostics.py", "imagenet_classes.txt"],
                        "checkpoint": str(checkpoint), "steer_target": args.steer_target,
                        "steer_token": args.steer_token, "steer_pool": args.steer_pool,
                        "average_patches": args.average_patches, "beta": args.beta, "clipping": args.clipping,
                        "score_gate": args.score_gate, "steer_noise": args.steer_noise,
                        "noise_seed": args.seed, "trace_score": args.trace_score},
            "sampling": {"device": args.device, "batch_size": args.batch_size,
                         "cpu_offload": args.cpu_offload, "local_files_only": args.offline,
                         "model_revision": args.model_revision},
            "evaluation": {"device": args.metric_device or args.device,
                           "batch_size": args.metric_batch_size, "local_files_only": args.offline}}
        for key in ("dtype", "steps", "height", "width", "max_sequence_length"):
            if getattr(args, key) is not None:
                cfg["sampling"][key] = getattr(args, key)
        for key in ("vlm_model", "vlm_device", "resnet_weights_path", "q16_prompts", "q16_clip_path", "nudenet_weights"):
            if getattr(args, key) is not None:
                cfg["evaluation"][key] = str(getattr(args, key))
        if args.skip_vlm:
            cfg["evaluation"]["vlm_enabled"] = False
        if args.with_q16 and task == "i2p":
            cfg["dataset"] = {"i2p_subset": "all"}
            cfg["evaluation"]["q16_enabled"] = True
        if args.max_records is not None:
            cfg.setdefault("dataset", {})["max_records"] = args.max_records
        cfg = configuration(task=task, model=args.model, base=cfg)
        cfg["dataset"].update(subset_eval_dataset(args, cfg))
        configs.append(cfg)
    if configs:
        # Beta/clipping are inference settings: changing them reuses these means.
        settings["sampling"] = {key: configs[0]["sampling"][key] for key in
            ("dtype", "steps", "height", "width", "max_sequence_length", "guidance_scale", "negative_prompt")}
        for cfg in configs:
            cfg["erasure_parameters"] = {**settings, "beta": args.beta, "clipping": args.clipping,
                                         "score_gate": args.score_gate, "steer_noise": args.steer_noise}
    return settings, configs


def offline_identity(settings, sampling, pairs):
    """Compare effective fit inputs, independently of inference and file layout.

    v1 collected the same text-to-image means as v2's default mode. Source
    digests remain provenance; the implementation version identifies the offline
    algorithm, so editing online steering or comments does not invalidate it.
    """
    identity = {key: value for key, value in settings.items()
                if key not in {"sources", "calibration_prompts", "sampling"}}
    if identity["implementation"] == "steer-sd3-mean-difference-v1":
        identity["implementation"] = "steer-sd3-mean-difference-v2"
    identity["model_revision"] = identity.get("model_revision") or None
    identity.setdefault("steer_target", "text_to_image")
    identity.setdefault("average_patches", True)
    identity.setdefault("steer_pool", False)
    identity["steer_token"] = (settings.get("steer_token", "image") if
        identity["steer_target"] in {"attention_residual", "mlp_output", "block_output"} else None)
    identity.setdefault("construction", {"method": "mean", "normalize_differences": False})
    identity.setdefault("prompt_mode", "casteer")
    identity["prompt_pairs"] = pairs
    identity["sampling"] = {key: sampling[key] for key in
        ("dtype", "steps", "height", "width", "max_sequence_length", "guidance_scale", "negative_prompt", "scheduler")}
    # Diffusers version annotations do not change the native scheduler settings.
    identity["sampling"]["scheduler_config"] = {
        key: value for key, value in sampling.get("scheduler_config", {}).items() if not key.startswith("_")}
    return identity


def run(args, experiments, env):
    env["PYTHONPATH"] = os.pathsep.join((str(WORKSPACE), env.get("PYTHONPATH", "")))
    for case_id, settings, configs in experiments:
        checkpoint = Path(configs[0]["adapter"]["checkpoint"])
        case_root = checkpoint.parent
        print(f"[steer:{args.model}:{case_id}] {settings['targets']} -> {settings['anchors']}; "
              f"target={args.steer_target}, token={args.steer_token}, pool={args.steer_pool}, average_patches={args.average_patches}, "
              f"prompts={args.prompt_mode}, construction={settings['construction']}, beta={args.beta:g}, clipping={args.clipping}, "
              f"score_gate={args.score_gate}, noise={args.steer_noise}, trace_score={args.trace_score}", flush=True)
        if args.steer_noise:
            print(f"[fit:skip] Gaussian noise ({args.steer_noise}), seed={args.seed}; no offline vectors", flush=True)
        elif args.phase in {"all", "fit"}:
            if args.dry_run:
                exists = (checkpoint / "metadata.json").is_file() and (checkpoint / "steering.pt").is_file()
                action = "reuse" if exists else "reconstruct" if args.statistics_sources[case_id] else "collect"
                print(f"[fit:{action}] {checkpoint}", flush=True)
            else:
                # A second process using this configuration waits for its vector.
                with locked(checkpoint.parent / ".fit.lock"):
                    exists = (checkpoint / "metadata.json").is_file() and (checkpoint / "steering.pt").is_file()
                    action = "reuse" if exists else "reconstruct" if args.statistics_sources[case_id] else "collect"
                    print(f"[fit:{action}] {checkpoint}", flush=True)
                    if not exists:
                        run_function("DiT.steer.method:fit", {"checkpoint": str(checkpoint),
                            "settings": settings, "config": configs[0],
                            "statistics_checkpoint": args.statistics_sources[case_id]}, env=env, cwd=WORKSPACE,
                            cpu_threads=args.cpu_threads)
        if args.phase == "fit":
            continue
        for cfg in configs:
            print(f"[{cfg['task']}] {args.record_counts[case_id, cfg['task']]} images", flush=True)
            if args.dry_run:
                directory = case_root / "plans" / args.method / f"{cfg['task']}-planned"
                path, manifest = directory / "config.json", directory / "manifest.json"
            else:
                resolve_scheduler(cfg)
                path, manifest = plugin_plan(case_root, "plans", cfg, phase=args.phase)
            commands = []
            prefix = [sys.executable, "-m", "benchmark"]
            if args.phase in {"all", "prepare"}:
                commands.append([*prefix, "prepare", "--config", str(path), "--manifest", str(manifest)])
            if args.phase in {"all", "sample"}:
                commands.append([*prefix, "sample", "--manifest", str(manifest), "--mode",
                                 "image" if cfg["task"] == "i2p" else "both", "--cpu-threads", str(args.cpu_threads)])
            if args.phase in {"all", "evaluate"}:
                commands.append([*prefix, "evaluate", "--manifest", str(manifest), "--cpu-threads", str(args.cpu_threads)])
            for command in commands:
                print("[command] " + shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=WORKSPACE, env=env, check=True)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.trace_score = args.trace_score and args.score_gate
    block_target = args.steer_target in {"attention_residual", "mlp_output", "block_output"}
    if not block_target:
        args.steer_token = "image"  # Token selection only applies to the three block sites.
    target_label = args.steer_target + (f"-{args.steer_token}" if block_target else "")
    target_label += "-pool" if args.steer_pool else ""
    args.variant = f"{target_label}-{'mean' if args.average_patches else 'per-position'}"
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    for name in ("base_model", "vlm_model"):
        value = getattr(args, name)
        if value and (value.startswith((".", "/", "~")) or Path(value).exists()):
            setattr(args, name, str(Path(value).expanduser().resolve()))
    args.cpu_offload = args.device.startswith("cuda") if args.cpu_offload is None else args.cpu_offload
    args.dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
    args.model_revision = args.model_revision or None
    args.construction = {"method": args.vector_method, "normalize_differences": args.normalize_differences}
    args.construction_label = "" if args.vector_method == "mean" else args.vector_method
    if args.vector_method == "robust":
        groups = (args.robust_groups if args.robust_groups is not None else
                  min(5, args.calibration_prompts or (1 if args.prompt_mode == "bare" else 50)))
        args.construction.update(groups=groups, max_iter=args.robust_max_iter, tol=args.robust_tol)
        args.construction_label += f"-g{groups}-i{args.robust_max_iter}-tol{args.robust_tol:g}"
    if args.vector_method == "pca-filter":
        args.construction["rank"] = args.pca_rank
        args.construction_label += f"-k{args.pca_rank}"
    if args.normalize_differences:
        args.construction_label += "-unit-diff"
    args.construction_label = args.construction_label.strip("-")
    args.method = args.method or (f"Steer-{args.variant}-"
                                  f"{args.construction_label + '-' if args.construction_label else ''}"
                                  f"{'erase' if args.beta < 0 else 'add'}-{abs(args.beta):g}-"
                                  f"{('clip' if args.clipping else 'signed') if args.score_gate else 'no-score-gate'}"
                                  f"{'-anchors' if args.use_anchor else '-context'}"
                                  f"{'-' + slug(args.target_anchor_config.stem) if args.use_anchor and args.target_anchor_config else ''}"
                                  f"{'-prompts-' + args.prompt_mode if args.prompt_mode != 'casteer' else ''}"
                                  f"{'-noise-' + args.steer_noise + '-seed' + str(args.seed) if args.steer_noise else ''}")
    args.tasks = {task.strip() for task in args.tasks.split(",")} if args.tasks else None

    # All method-parameter and existing-checkpoint checks live at this entry.
    sizes = [args.cpu_threads, args.batch_size, args.metric_batch_size, args.calibration_prompts,
             args.steps, args.height, args.width, args.max_sequence_length, args.max_records,
             args.robust_groups, args.robust_max_iter, args.pca_rank]
    if args.subset_eval:
        sizes.extend(getattr(args, f"{task}_test_num") for task in SUBSET_EVAL_DEFAULTS)
    if any(value is not None and value <= 0 for value in sizes):
        p.error("Counts, dimensions and sequence lengths must be positive.")
    if not math.isfinite(args.beta) or not 0 <= args.seed < 2**32:
        p.error("Use finite beta and a uint32 seed.")
    if not math.isfinite(args.robust_tol) or args.robust_tol <= 0:
        p.error("--robust-tol must be finite and positive.")
    if args.device == "cpu" and (args.dtype != "float32" or args.cpu_offload):
        p.error("CPU execution uses float32 without CPU offload.")
    cases = definitions()["cases"]
    selected = select_cases(args.cases)
    try:
        experiments = [(key, *case_configs(args, key, cases[key])) for key in selected]
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    experiments = [entry for entry in experiments if entry[2]]
    available = {cfg["task"] for _, _, configs in experiments for cfg in configs}
    if not experiments or args.tasks and args.tasks - available:
        p.error("Requested tasks are absent from the selected cases.")
    args.record_counts = {(key, cfg["task"]): len(prompt_records(cfg)[0])
                          for key, _, configs in experiments for cfg in configs}
    args.statistics_sources = {}
    for case_id, settings, configs in experiments:
        sampling = configs[0]["sampling"]
        if not args.dry_run:
            resolve_scheduler(configs[0])
        if args.steer_noise:
            parent = args.run_root / args.model / case_id / args.variant / "noise"
            label = f"{args.steer_noise}-seed{args.seed}-{sampling_label(sampling)}"
            identity = {"mode": args.steer_noise, "seed": args.seed, "targets": settings["targets"],
                        "base_model": settings["base_model"], "model_revision": args.model_revision,
                        "steer_target": args.steer_target, "steer_token": args.steer_token, "steer_pool": args.steer_pool,
                        "sampling": sampling}
            directory = (parent / f"{label}-planned" if args.dry_run else
                         registered_directory(parent, label, identity))
            checkpoint = directory / "checkpoint"
            if not args.dry_run:
                checkpoint.mkdir(exist_ok=True)
                write_json(checkpoint / "metadata.json", identity, portable=True)
            for cfg in configs:
                cfg["adapter"]["checkpoint"] = str(checkpoint)
                if not args.dry_run:
                    cfg["sampling"]["scheduler_config"] = sampling["scheduler_config"]
            continue
        pairs = [{"target": target, "anchor": anchor,
                  "pairs": prompt_pairs(settings["group"], target, anchor, args.calibration_prompts, mode=args.prompt_mode)}
                 for target, anchor in zip(settings["targets"], settings["anchors"])]
        pair_count = len(pairs[0]["pairs"])
        if args.calibration_prompts and args.calibration_prompts > pair_count:
            p.error(f"{case_id}: --prompt-mode {args.prompt_mode} provides at most {pair_count} pairs; "
                    "reduce --calibration-prompts or omit it. Identical pairs are not repeated.")
        if args.vector_method == "robust" and settings["construction"]["groups"] > pair_count:
            p.error(f"--robust-groups cannot exceed the number of prompt pairs ({pair_count}).")
        if args.vector_method == "pca-filter" and args.pca_rank > pair_count:
            p.error(f"--pca-rank cannot exceed the number of prompt pairs ({pair_count}).")
        identity = offline_identity(settings, sampling, pairs)
        collection_identity = {key: value for key, value in identity.items() if key != "construction"}
        case_root = args.run_root / args.model / case_id
        checkpoint = None
        args.statistics_sources[case_id] = None
        # Include both legacy checkpoints and every saved configuration variant.
        for metadata in sorted(case_root.glob("**/checkpoint/metadata.json")):
            if not (metadata.parent / "steering.pt").is_file():
                continue
            stored = read_json(metadata)
            previous = offline_identity(stored["settings"], stored["sampling"], stored["prompt_pairs"])
            if args.dry_run:
                # Preview without loading even the model's scheduler config.
                previous["sampling"]["scheduler_config"] = identity["sampling"]["scheduler_config"]
            if previous == identity:
                checkpoint = metadata.parent
                break
            if (args.statistics_sources[case_id] is None and stored.get("paired_differences")
                    and Path(stored["paired_differences"]).is_dir()
                    and {key: value for key, value in previous.items() if key != "construction"} == collection_identity):
                args.statistics_sources[case_id] = str(metadata.parent)
        if checkpoint is None:
            if not args.dry_run and args.phase not in {"all", "fit"}:
                p.error(f"No matching offline vector; run --phase fit with these settings first: {case_root}")
            parent = case_root / args.variant / "offline"
            label = f"{sampling_label(sampling)}-pairs{len(pairs[0]['pairs'])}-seed{args.seed}"
            if args.prompt_mode != "casteer":
                label += f"-prompts-{args.prompt_mode}"
            if args.construction_label:
                label += f"-{args.construction_label}"
            directory = (parent / f"{label}-planned" if args.dry_run else
                         registered_directory(parent, label, identity))
            checkpoint = directory / "checkpoint"
        for cfg in configs:
            cfg["adapter"]["checkpoint"] = str(checkpoint)
            if not args.dry_run:
                cfg["sampling"]["scheduler_config"] = sampling["scheduler_config"]
    with cpu_parent(args, p) as env:
        run(args, experiments, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
