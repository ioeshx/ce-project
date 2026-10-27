"""Five public experiments, staged erasure/generation/evaluation processes."""
import argparse
import csv
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from benchmark.common import ROOT, configuration, file_digest, read_json
from benchmark.data import FILES, load_manifest, prepare
from benchmark.plugin_plans import plugin_plan
from benchmark.sampling_profile import PROFILE
from . import METHOD_ROOT, WORKSPACE
from .protocol import EXPERIMENTS, I2P_CASES, case_config, cases, training_settings


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return n


def positive(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("Expected a positive finite number.")
    return n


def seeds(value):
    values = [int(v) for v in value.split(",")]
    if not values or len(set(values)) != len(values) or min(values) < 0 or max(values) >= 2**63:
        raise argparse.ArgumentTypeError("Expected unique nonnegative record seeds below 2**63.")
    return values


def options(p):
    p.add_argument("--run-root", type=Path, default=METHOD_ROOT / "runs/benchmark-sd14")
    p.add_argument("--method-prefix", default="ESD-sd14")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--original-dir", type=Path, default=ROOT / "original")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--metric-device", default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--cpu-threads", type=positive_int, default=4)
    p.add_argument("--batch-size", type=positive_int, default=1)
    p.add_argument("--metric-batch-size", type=positive_int, default=32)
    p.add_argument("--train-method", choices=("esd-x", "esd-u", "esd-all", "esd-x-strict", "selfattn"))
    p.add_argument("--iterations", type=positive_int, help="Total optimization steps; overrides the per-concept budget.")
    p.add_argument("--iterations-per-concept", type=positive_int, default=200)
    p.add_argument("--lr", type=positive, default=5e-5)
    p.add_argument("--negative-guidance", type=positive, default=2.0)
    p.add_argument("--training-seed", type=int, default=1234)
    p.add_argument("--training-steps", type=positive_int, default=PROFILE["steps"])
    p.add_argument("--training-guidance-scale", type=positive, default=PROFILE["guidance_scale"])
    p.add_argument("--training-resolution", type=positive_int, default=PROFILE["width"])
    p.add_argument("--training-dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--q16-targets", default="violence;nudity;harm", help="Matches the UCE comparison targets; separated by ;.")
    p.add_argument("--nudenet-targets", default="nudity", help="Nudity erasure target(s), separated by ;.")
    p.add_argument("--q16-prompts", type=Path)
    p.add_argument("--q16-clip-path", type=Path)
    p.add_argument("--nudenet-weights", type=Path)
    p.add_argument("--nudenet-threshold", type=float)
    p.add_argument("--steps", type=positive_int, default=PROFILE["steps"])
    p.add_argument("--height", type=positive_int, default=PROFILE["height"])
    p.add_argument("--width", type=positive_int, default=PROFILE["width"])
    p.add_argument("--sampling-dtype", choices=("float16", "float32"), default=PROFILE["dtype"])
    p.add_argument("--seeds", type=seeds, default=None)
    p.add_argument("--prompts-per-concept", type=positive_int, help="Explicit reduced protocol for smoke checks.")
    p.add_argument("--max-records", type=positive_int, help="Explicit reduced protocol; never paper scores.")


def parser():
    p = argparse.ArgumentParser(description="ESD SD1.4 -> shared concept-erasure benchmark")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "edit", "benchmark"):
        q = sub.add_parser(name)
        options(q)
        if name == "run":
            q.add_argument("--experiments", default=",".join(EXPERIMENTS))
            q.add_argument("--phase", choices=("all", "erase", "benchmark", "prepare", "sample", "evaluate"), default="all")
            q.add_argument("--dry-run", action="store_true")
        else:
            q.add_argument("--case", required=True)
            if name == "benchmark":
                q.add_argument("--phase", choices=("run", "prepare", "sample", "evaluate"), default="run")
    return p


def execute(command, dry_run=False):
    command = [str(v) for v in command]
    print("[command] " + shlex.join(command), flush=True)
    if not dry_run:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(METHOD_ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")])
        subprocess.run(command, cwd=WORKSPACE, env=env, check=True)


def forwarded(args):
    omitted = {"command", "experiments", "phase", "dry_run", "case"}
    result = []
    for name, value in vars(args).items():
        if name in omitted or value is None or value is False:
            continue
        flag = "--" + name.replace("_", "-")
        result.append(flag)
        if value is not True:
            result.append(",".join(map(str, value)) if isinstance(value, list) else str(value))
    return result


def suite(args):
    for case in cases(args.experiments):
        settings = training_settings(case, args)  # Validate even in a dry run.
        print(f"[case] {case}: {len(settings['targets'])} target(s); {settings['targets']}", flush=True)
        prefix = [sys.executable, "-m", "esd_benchmark"]
        common = ["--case", case, *forwarded(args)]
        if args.phase in {"all", "erase"}:
            execute([*prefix, "edit", *common], args.dry_run)
        if args.phase != "erase":
            phase = "run" if args.phase in {"all", "benchmark"} else args.phase
            execute([*prefix, "benchmark", *common, "--phase", phase], args.dry_run)


def benchmark_config(case, args, checkpoint):
    cfg = case_config(case, args)
    info = read_json(checkpoint / "metadata.json")
    if info["settings"] != training_settings(case, args):
        raise ValueError("Edit parameters/source differ from the checkpoint; use matching options or a new --run-root.")
    cfg["method"] = f"{args.method_prefix}-{info['settings']['train_method']}-{case}"
    for field in ("output_dir", "cache_dir", "original_dir"):
        cfg[field] = str(getattr(args, field).expanduser().resolve())
    cfg["sampling"].update(device=args.device, batch_size=args.batch_size, local_files_only=args.offline,
                           steps=args.steps, height=args.height, width=args.width,
                           dtype=args.sampling_dtype, model_revision=args.model_revision)
    if args.seeds is not None:
        cfg["dataset"]["seeds"] = args.seeds
    if args.max_records is not None:
        cfg["dataset"]["max_records"] = args.max_records
    cfg["evaluation"].update(device=args.metric_device or args.device, batch_size=args.metric_batch_size,
                             local_files_only=args.offline)
    for name in ("q16_prompts", "q16_clip_path", "nudenet_weights", "nudenet_threshold"):
        value = getattr(args, name)
        if value is not None:
            cfg["evaluation"][name] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    cfg["adapter"] = {"kind": "plugin", "factory": "esd_benchmark.adapter:create", "checkpoint": str(checkpoint)}
    cfg["esd"] = {"training_settings": info["settings"], "detector": case.removeprefix("i2p_") if case in I2P_CASES else None,
                  "dataset_sha256": file_digest(ROOT / "dataset" / FILES[cfg["task"]]),
                  "configuration_sha256": file_digest(cfg["configuration_source"]),
                  "sampling_profile": str(ROOT / "configs/sd14_sampling.json"),
                  "sampling_overrides": {k: cfg["sampling"][k] for k in ("steps", "height", "width", "dtype")
                                         if cfg["sampling"][k] != PROFILE[k]}}
    return configuration(task=cfg["task"], base=cfg)


def benchmark(args):
    case_dir = args.run_root.expanduser().resolve() / args.case
    cfg = benchmark_config(args.case, args, case_dir / "checkpoint")
    subset = None
    if args.prompts_per_concept and cfg["task"] in {"instance", "style"}:
        filename = "ip_imagenet_prompts.csv" if cfg["task"] == "instance" else "style_painting_prompts.csv"
        with (ROOT / "dataset" / filename).open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        subset, counts = [], {}
        for row in rows:
            c = row["concept"]
            if counts.get(c, 0) < args.prompts_per_concept:
                subset.append(row)
                counts[c] = counts.get(c, 0) + 1
    config_path, manifest = plugin_plan(case_dir, "plans", cfg, phase=args.phase, subset=subset)
    if args.phase in {"run", "prepare"}:
        if manifest.is_file():
            if load_manifest(manifest)["config"] != configuration(config_path):
                raise ValueError("Stored manifest changed; refusing to overwrite a historical plan.")
        else:
            prepare(configuration(config_path), manifest)
        print(f"[manifest] {manifest}", flush=True)
    if args.phase in {"run", "sample"}:
        execute([sys.executable, "-m", "benchmark", "sample", "--manifest", manifest,
                 "--mode", "image" if cfg["task"] == "i2p" else "both", "--cpu-threads", args.cpu_threads])
    if args.phase in {"run", "evaluate"}:
        # Generation was in a separate process and has released its model.
        import torch
        torch.set_num_threads(args.cpu_threads)
        if args.case in I2P_CASES:
            from .i2p import evaluate
            evaluate(manifest, args.case.removeprefix("i2p_"))
        else:
            from benchmark.evaluate import evaluate
            evaluate(manifest)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        for name, value in vars(args).items():
            if isinstance(value, Path):
                setattr(args, name, value.expanduser().resolve())
        if args.height % 8 or args.width % 8:
            raise ValueError("SD1.4 dimensions must be divisible by 8.")
        if args.training_resolution % 8 or not 0 <= args.training_seed < 2**32:
            raise ValueError("Training resolution must be divisible by 8; seed must be in [0, 2**32).")
        if str(args.device).startswith("cpu") and args.sampling_dtype != "float32":
            raise ValueError("CPU smoke checks require explicit --sampling-dtype float32.")
        if str(args.device).startswith("cpu") and args.training_dtype != "float32":
            raise ValueError("CPU optimization requires --training-dtype float32.")
        if args.command == "run":
            suite(args)
        elif args.command == "edit":
            from .training import train
            train(args.run_root.expanduser().resolve() / args.case / "checkpoint", training_settings(args.case, args),
                  offline=args.offline, cpu_threads=args.cpu_threads)
        else:
            benchmark(args)
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        p.error(str(error))
