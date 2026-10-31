"""EraseAnything FLUX training and the four shared DiT experiment groups."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

METHOD_ROOT = Path(__file__).resolve().parent
WORKSPACE = METHOD_ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))

from benchmark.common import ROOT, configuration, file_digest, read_json
from benchmark.dit_cases import definitions, select_cases
from benchmark.execution import compatible_settings, cpu_parent, run_function
from benchmark.model_profiles import profile, resolve_scheduler
from benchmark.plugin_plans import plugin_plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("sd35_medium", "flux1_dev"), default="flux1_dev")
    p.add_argument("--cases", default="all", help="all/i2p/ip/style/object or comma-separated case IDs")
    p.add_argument("--tasks", help="Optional task filter, e.g. mscoco for the nudity checkpoint")
    p.add_argument("--phase", choices=("all", "data", "train", "prepare", "sample", "evaluate"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-root", type=Path, default=METHOD_ROOT / "runs/benchmark-dit")
    p.add_argument("--method", default="EraseAnything", help="Method name within each model/case's CSV")
    p.add_argument("--base-model", help="Optional local Diffusers base model directory")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text-device", default="cpu")
    p.add_argument("--training-config", type=Path, help="Optional per-concept prompt/anchor/synonyms/irrelevant JSON")
    p.add_argument("--training-images-per-concept", type=int, default=4)
    p.add_argument("--iterations-per-concept", type=int, default=200)
    p.add_argument("--iterations", type=int, help="Total optimizer updates per case, even and at least 2 per target")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--negative-guidance", type=float, default=1.0)
    p.add_argument("--loss-weights", type=float, nargs=4, default=[1.0, .001, .1, .1])
    p.add_argument("--training-steps", type=int, default=28)
    p.add_argument("--training-resolution", type=int, default=512)
    p.add_argument("--training-sequence-length", type=int, default=256)
    p.add_argument("--training-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--training-seed", type=int, default=1234)
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
    """Read live DiT cases and record the explicit local training recipe."""
    selected = profile(args.model)
    targets = case["targets"]
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    concepts = []
    for target in targets:
        prompt = (f"an artwork painted in the style of {target}" if case["group"] == "style"
                  else f"an image depicting {target}")
        concept = {"name": target, "prompt": prompt, "anchor": anchors[target],
                   "synonyms": [target], "irrelevant": ["mountain", "bicycle", "fruit"]}
        concept.update(args.training_overrides.get(target, {}))
        concepts.append(concept)
    data_sampling = dict(selected["sampling"])
    for key in ("dtype", "steps", "height", "width"):
        if getattr(args, key) is not None:
            data_sampling[key] = getattr(args, key)
    settings = {"model_name": args.model, "base_model": args.base_model or selected["base_model"],
                "model_revision": args.model_revision, "case_id": case_id, "targets": targets,
                "concepts": concepts, "training_images_per_concept": args.training_images_per_concept,
                "data_sampling": data_sampling, "iterations": args.iterations or args.iterations_per_concept * len(targets),
                "lr": args.lr, "rank": args.rank, "negative_guidance": args.negative_guidance,
                "training_guidance": 3.5, "rollout_guidance": 3.0,
                "loss_weights": args.loss_weights, "training_steps": args.training_steps,
                "training_resolution": args.training_resolution, "training_sequence_length": args.training_sequence_length,
                "training_dtype": args.training_dtype, "training_seed": args.training_seed,
                "device": args.device, "text_device": args.text_device,
                "gradient_checkpointing": args.gradient_checkpointing,
                "sources": {str(path.relative_to(METHOD_ROOT)): file_digest(path) for path in (
                    Path(__file__), METHOD_ROOT / "benchmark_training.py", METHOD_ROOT / "utils/calc_loss.py",
                    METHOD_ROOT / "utils/infoNCE.py", METHOD_ROOT / "utils/esd_utils.py",
                    METHOD_ROOT / "tools/scheduler_process.py")}}
    settings = compatible_settings(args.run_root / args.model / case_id, settings, METHOD_ROOT.name)
    configs = []
    for task in case["tasks"]:
        if args.tasks and task not in args.tasks:
            continue
        cfg = {
            "case_id": case_id, "method": args.method, "base_model": settings["base_model"],
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
        cfg["adapter"] = {"kind": "lora",
                          "checkpoint": str(args.run_root / args.model / case_id / "checkpoint")}
        cfg["erasure_parameters"] = settings
        configs.append(configuration(task=task, model=args.model, base=cfg))
    return settings, configs


def run(args, experiments, env):
    env["PYTHONPATH"] = os.pathsep.join((str(METHOD_ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")))
    for case_id, settings, configs in experiments:
        case_root = args.run_root / args.model / case_id
        print(f"[EraseAnything:{args.model}:{case_id}] {settings['targets']}; {settings['iterations']} updates", flush=True)
        if args.phase in {"all", "data"}:
            print(f"[training data] {case_root / 'training-data'}", flush=True)
            if not args.dry_run:
                run_function("benchmark_training:prepare_images",
                             {"directory": str(case_root / "training-data"), "settings": settings,
                              "sampling_config": configs[0], "offline": args.offline},
                             env=env, cwd=WORKSPACE, cpu_threads=args.cpu_threads)
        if args.phase in {"all", "train"}:
            print(f"[train] {case_root / 'checkpoint'}", flush=True)
            if not args.dry_run:
                run_function("benchmark_training:train",
                             {"checkpoint": str(case_root / "checkpoint"), "data_directory": str(case_root / "training-data"),
                              "settings": settings, "offline": args.offline},
                             env=env, cwd=WORKSPACE, cpu_threads=args.cpu_threads)
        if args.phase in {"data", "train"}:
            continue
        if not args.dry_run and args.phase in {"prepare", "sample", "evaluate"}:
            info = read_json(case_root / "checkpoint/metadata.json")
            if (info["settings"] != settings or info["weights_sha256"] !=
                    file_digest(case_root / "checkpoint/pytorch_lora_weights.safetensors")):
                raise ValueError("Checkpoint does not match the requested EraseAnything experiment.")
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
    if args.model != "flux1_dev":
        p.error("SD3.5 is skipped: the released attention-map losses are FLUX-specific; use --model flux1_dev.")
    positive = (args.lr, args.rank, args.training_images_per_concept, args.iterations_per_concept,
                args.training_steps, args.training_resolution, args.training_sequence_length,
                args.cpu_threads, args.batch_size, args.metric_batch_size)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        p.error("Learning rate, dimensions, batch/thread counts and training budgets must be positive.")
    if any(value is not None and value <= 0 for value in (args.iterations, args.max_records)):
        p.error("Explicit iteration and record limits must be positive.")
    if (args.training_resolution % 16 or not 0 <= args.training_seed < 2**32
            or not math.isfinite(args.negative_guidance) or args.negative_guidance < 0
            or any(not math.isfinite(value) or value <= 0 for value in args.loss_weights)):
        p.error("Use a resolution divisible by 16, a seed in [0, 2**32), nonnegative guidance and four positive loss weights.")
    args.training_overrides = read_json(args.training_config) if args.training_config else {}
    args.tasks = [task.strip() for task in args.tasks.split(",")] if args.tasks else None
    cases = definitions()["cases"]
    selected = select_cases(args.cases)
    experiments = [(case_id, *case_configs(args, case_id, cases[case_id])) for case_id in selected]
    if args.tasks and set(args.tasks) - {cfg["task"] for _, _, configs in experiments for cfg in configs}:
        p.error("--tasks includes tasks absent from the selected cases.")
    experiments = [entry for entry in experiments if entry[2]]
    for _, settings, _ in experiments:
        if settings["iterations"] < 2 * len(settings["targets"]) or settings["iterations"] % 2:
            p.error("Each case needs an even update count, with an upper/lower pair for every target.")
        for concept in settings["concepts"]:
            if (concept["name"] not in concept["prompt"] or not concept["synonyms"]
                    or len(concept["irrelevant"]) != 3):
                p.error("Training prompts must contain the target; provide aliases and exactly three irrelevant concepts.")
    with cpu_parent(args, p) as env:
        run(args, experiments, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
