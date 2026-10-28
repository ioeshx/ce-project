"""Four DiT experiment groups: train ESD, then reuse benchmark sampling/evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from benchmark.common import ROOT, configuration, file_digest
from benchmark.dit_cases import definitions, select_cases
from benchmark.execution import compatible_settings, cpu_parent, run_function
from benchmark.model_profiles import profile, resolve_scheduler
from benchmark.plugin_plans import plugin_plan
from . import METHOD_ROOT, WORKSPACE


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("sd35_medium", "flux1_dev"), default="sd35_medium")
    p.add_argument("--cases", default="all", help="all/i2p/ip/style/object or comma-separated case IDs")
    p.add_argument("--tasks", help="Optional task filter, e.g. mscoco for the nudity checkpoint")
    p.add_argument("--phase", choices=("all", "train", "prepare", "sample", "evaluate"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-root", type=Path, default=METHOD_ROOT / "runs/benchmark-dit")
    p.add_argument("--method", help="Default: ESD-x or ESD-x-strict, within each model/case CSV")
    p.add_argument("--base-model", help="Optional local Diffusers base model directory")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text-device", default="cpu", help="Text encoder device; independent of denoiser training")
    p.add_argument("--train-method", choices=("esd-x", "esd-x-strict"), default="esd-x")
    p.add_argument("--iterations-per-concept", type=int, default=1400)
    p.add_argument("--iterations", type=int, help="Override total updates per case")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--negative-guidance", type=float, default=1.0)
    p.add_argument("--training-guidance", type=float, default=1.0, help="FLUX direct transformer guidance")
    p.add_argument("--training-seed", type=int, default=1234)
    p.add_argument("--training-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--training-steps", type=int, help="Default: model's native sampling profile")
    p.add_argument("--training-resolution", type=int, default=512)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--metric-device", help="Defaults to --device")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--metric-batch-size", type=int, default=32)
    p.add_argument("--cpu-offload", action="store_true", help="Offload the sampling pipeline")
    p.add_argument("--device-map", choices=("balanced", "auto"))
    p.add_argument("--max-memory", type=json.loads)
    p.add_argument("--gpu-memory-reserve", type=float)
    p.add_argument("--with-q16", action="store_true")
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--vlm-model")
    p.add_argument("--vlm-device")
    p.add_argument("--resnet-weights-path", type=Path)
    p.add_argument("--q16-prompts", type=Path)
    p.add_argument("--q16-clip-path", type=Path)
    p.add_argument("--nudenet-weights", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--original-dir", type=Path)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--max-records", type=int, help="Reduced validation protocol, not full paper results")
    p.add_argument("--dtype", choices=("bfloat16", "float32", "float16"))
    p.add_argument("--steps", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    return p


def case_configs(args, case_id, case):
    """Use the live DiT JSONs, with only ESD and execution options added."""
    selected = profile(args.model)
    targets = case["targets"]
    rollout = dict(selected["sampling"])
    rollout.update(steps=args.training_steps or rollout["steps"], height=args.training_resolution,
                   width=args.training_resolution, dtype=args.training_dtype,
                   execution_backend=str(args.device).split(":", 1)[0],
                   noise="native ESD per-update device generator; seed sampled from [0, 2**15]")
    settings = {"model_name": args.model, "base_model": args.base_model or selected["base_model"],
                "model_revision": args.model_revision, "case_id": case_id, "targets": targets,
                "train_method": args.train_method,
                "iterations": args.iterations or args.iterations_per_concept * len(targets),
                "lr": args.lr, "negative_guidance": args.negative_guidance,
                "training_guidance": args.training_guidance, "training_seed": args.training_seed,
                "device": args.device, "text_device": args.text_device,
                "training_protocol": rollout, "gradient_checkpointing": args.gradient_checkpointing,
                "sources": {str(path.relative_to(METHOD_ROOT)): file_digest(path) for path in (
                    Path(__file__), Path(__file__).with_name("dit_training.py"),
                    METHOD_ROOT / "esd_flux.py", METHOD_ROOT / "utils/esd_trainer.py",
                    METHOD_ROOT / "utils/flux_utils.py", METHOD_ROOT / "utils/esd_checkpoint.py")}}
    settings = compatible_settings(args.run_root / args.model / case_id, settings, METHOD_ROOT.name)
    configs = []
    for task in case["tasks"]:
        if args.tasks and task not in args.tasks:
            continue
        cfg = {
            "case_id": case_id, "method": args.method or f"ESD-{args.train_method.removeprefix('esd-')}",
            "base_model": settings["base_model"],
            "output_dir": str(args.output_dir), "cache_dir": str(args.cache_dir),
            "original_dir": str(args.original_dir or ROOT / "original" / args.model),
            "dataset": {},
            "sampling": {"device": args.device, "batch_size": args.batch_size,
                         "local_files_only": args.offline, "model_revision": args.model_revision,
                         "cpu_offload": args.cpu_offload},
            "evaluation": {"device": args.metric_device or args.device, "batch_size": args.metric_batch_size,
                           "local_files_only": args.offline}}
        for arg, key in (("dtype", "dtype"), ("steps", "steps"), ("height", "height"), ("width", "width"),
                         ("device_map", "device_map"), ("max_memory", "max_memory"),
                         ("gpu_memory_reserve", "gpu_memory_reserve_gib")):
            if getattr(args, arg) is not None:
                cfg["sampling"][key] = getattr(args, arg)
        for key in ("vlm_model", "vlm_device", "resnet_weights_path", "q16_prompts", "q16_clip_path", "nudenet_weights"):
            if getattr(args, key) is not None:
                cfg["evaluation"][key] = str(getattr(args, key))
        if args.skip_vlm:
            cfg["evaluation"]["vlm_enabled"] = False
        if args.with_q16 and task == "i2p":
            cfg["dataset"]["i2p_subset"] = "all"
            cfg["evaluation"]["q16_enabled"] = True
        if args.max_records is not None:
            cfg["dataset"]["max_records"] = args.max_records
        cfg["adapter"] = {"kind": "plugin", "factory": "esd_benchmark.dit_training:create",
                          "checkpoint": str(args.run_root / args.model / case_id / "checkpoint")}
        cfg["erasure_parameters"] = settings
        configs.append(configuration(task=task, model=args.model, base=cfg))
    return settings, configs


def run(args, experiments, env):
    env["PYTHONPATH"] = os.pathsep.join((str(METHOD_ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")))
    for case_id, settings, configs in experiments:
        case_root = args.run_root / args.model / case_id
        print(f"[ESD:{args.model}:{case_id}] {settings['targets']}; "
              f"{settings['train_method']}, {settings['iterations']} updates", flush=True)
        if args.phase in {"all", "train"}:
            print(f"[train] {case_root / 'checkpoint'}", flush=True)
            if not args.dry_run:
                run_function("esd_benchmark.dit_training:train",
                             {"checkpoint": str(case_root / "checkpoint"), "settings": settings,
                              "offline": args.offline, "cpu_threads": args.cpu_threads},
                             env=env, cwd=WORKSPACE, cpu_threads=args.cpu_threads)
        if args.phase == "train":
            continue
        for cfg in configs:
            if args.dry_run:
                directory = case_root / "plans" / f"{cfg['task']}-planned"
                path, manifest = directory / f"{cfg['task']}.json", directory / f"{cfg['task']}-manifest.json"
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
    # Resolve user paths once, before entering the workspace or spawning workers.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.base_model:
        args.base_model = str(Path(args.base_model).expanduser().resolve())
        if not Path(args.base_model).is_dir():
            p.error("--base-model must name a local Diffusers directory.")
    if args.vlm_model and (args.vlm_model.startswith((".", "/", "~")) or Path(args.vlm_model).exists()):
        args.vlm_model = str(Path(args.vlm_model).expanduser().resolve())
    positive = (args.lr, args.iterations_per_concept, args.cpu_threads, args.batch_size,
                args.metric_batch_size, args.training_resolution)
    optional = (args.iterations, args.training_steps, args.max_records)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or any(v is not None and v <= 0 for v in optional):
        p.error("Learning rate, sizes, iteration counts and batch/thread counts must be positive.")
    if (not math.isfinite(args.negative_guidance) or args.negative_guidance < 0
            or not math.isfinite(args.training_guidance) or args.training_guidance < 0
            or args.training_resolution % 16 or not 0 <= args.training_seed < 2**32):
        p.error("Guidance must be finite/nonnegative; training resolution divisible by 16; seed in [0, 2**32).")
    args.tasks = [task.strip() for task in args.tasks.split(",")] if args.tasks else None
    cases = definitions()["cases"]
    selected = select_cases(args.cases)
    experiments = [(case_id, *case_configs(args, case_id, cases[case_id])) for case_id in selected]
    if args.tasks and set(args.tasks) - {cfg["task"] for _, _, configs in experiments for cfg in configs}:
        p.error("--tasks includes tasks absent from the selected cases.")
    experiments = [entry for entry in experiments if entry[2]]
    for _, settings, _ in experiments:
        if settings["iterations"] < len(settings["targets"]):
            p.error("Training iterations must cover every target at least once.")
    with cpu_parent(args, p) as env:
        run(args, experiments, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
