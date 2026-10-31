"""EraseAnything SD3.5 Medium experiments, alongside the existing FLUX entry."""
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from benchmark_dit import METHOD_ROOT, WORKSPACE, case_configs, parser
from benchmark.common import file_digest, read_json
from benchmark.dit_cases import definitions, select_cases
from benchmark.execution import compatible_settings, cpu_parent, run_function
from benchmark.model_profiles import resolve_scheduler
from benchmark.plugin_plans import plugin_plan


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
                run_function("benchmark_sd35_training:train",
                             {"checkpoint": str(case_root / "checkpoint"), "data_directory": str(case_root / "training-data"),
                              "settings": settings, "offline": args.offline},
                             env=env, cwd=WORKSPACE, cpu_threads=args.cpu_threads)
        if args.phase in {"data", "train"}:
            continue
        if not args.dry_run and args.phase in {"prepare", "sample", "evaluate"}:
            info = read_json(case_root / "checkpoint/metadata.json")
            if (info["settings"] != settings or info["weights_sha256"] !=
                    file_digest(case_root / "checkpoint/pytorch_lora_weights.safetensors")):
                raise ValueError("Checkpoint does not match the requested SD3.5 experiment.")
        for config in configs:
            if args.dry_run:
                directory = case_root / "plans" / f"{config['task']}-planned"
                path = directory / f"{config['task']}.json"
                manifest = directory / f"{config['task']}-manifest.json"
            else:
                resolve_scheduler(config)
                path, manifest = plugin_plan(case_root, "plans", config, phase=args.phase)
            commands = []
            prefix = [sys.executable, "-m", "benchmark"]
            if args.phase in {"all", "prepare"}:
                commands.append([*prefix, "prepare", "--config", str(path), "--manifest", str(manifest)])
            if args.phase in {"all", "sample"}:
                commands.append([*prefix, "sample", "--manifest", str(manifest), "--mode",
                                 "image" if config["task"] == "i2p" else "both", "--cpu-threads", str(args.cpu_threads)])
            if args.phase in {"all", "evaluate"}:
                commands.append([*prefix, "evaluate", "--manifest", str(manifest), "--cpu-threads", str(args.cpu_threads)])
            for command in commands:
                print("[command] " + shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=WORKSPACE, env=env, check=True)


def main(argv=None):
    p = parser()
    p.description = __doc__
    p.set_defaults(model="sd35_medium")
    p.add_argument("--rollout-guidance", type=float, default=3.0,
                   help="SD3 classifier-free guidance during training rollouts; upstream uses 3")
    args = p.parse_args(argv)
    # Resolve paths and validate the request once, before running any case.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.base_model:
        args.base_model = str(Path(args.base_model).expanduser().resolve())
        if not Path(args.base_model).is_dir():
            p.error("--base-model must name a local Diffusers directory.")
    if args.vlm_model and (args.vlm_model.startswith((".", "/", "~")) or Path(args.vlm_model).exists()):
        args.vlm_model = str(Path(args.vlm_model).expanduser().resolve())
    if args.model != "sd35_medium":
        p.error("Use benchmark_dit.py for FLUX; this entry implements SD3.5 Medium.")
    positive = (args.lr, args.rank, args.training_images_per_concept, args.iterations_per_concept,
                args.training_steps, args.training_resolution, args.training_sequence_length,
                args.cpu_threads, args.batch_size, args.metric_batch_size)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        p.error("Learning rate, dimensions, batch/thread counts and training budgets must be positive.")
    if any(value is not None and value <= 0 for value in (args.iterations, args.max_records)):
        p.error("Explicit iteration and record limits must be positive.")
    if (args.training_resolution % 16 or args.training_sequence_length > 512 or args.training_steps > 1000
            or not 0 <= args.training_seed < 2**32
            or any(not math.isfinite(value) or value < 0 for value in (args.negative_guidance, args.rollout_guidance))
            or any(not math.isfinite(value) or value <= 0 for value in args.loss_weights)):
        p.error("Use a resolution divisible by 16, T5 length <=512, training steps <=1000, seed in [0, 2**32), nonnegative guidance and positive loss weights.")
    args.training_overrides = read_json(args.training_config) if args.training_config else {}
    args.tasks = [task.strip() for task in args.tasks.split(",")] if args.tasks else None
    cases = definitions()["cases"]
    experiments = []
    for case_id in select_cases(args.cases):
        settings, configs = case_configs(args, case_id, cases[case_id])
        settings.pop("training_guidance")  # SD3 direct predictions have no distilled guidance input.
        settings.update(
            implementation="eraseanything-sd35-v1", rollout_guidance=args.rollout_guidance,
            attention_layer="transformer_blocks[-1].attn", attention_tokens="T5 only, after CLIP sequence",
            training_scheduler="upstream sorted logit-normal sigmas; t=1000*sigma",
            sources={str(path.relative_to(METHOD_ROOT)): file_digest(path) for path in (
                Path(__file__), METHOD_ROOT / "benchmark_sd35_training.py", METHOD_ROOT / "benchmark_dit.py",
                METHOD_ROOT / "benchmark_training.py", METHOD_ROOT / "utils/infoNCE.py",
                METHOD_ROOT / "tools/scheduler_process.py")})
        settings = compatible_settings(args.run_root / args.model / case_id, settings, METHOD_ROOT.name)
        for config in configs:
            config["erasure_parameters"] = settings
        experiments.append((case_id, settings, configs))
    if args.tasks and set(args.tasks) - {config["task"] for _, _, configs in experiments for config in configs}:
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
