"""Run native GEM erasure with the shared DiT sampling and evaluation protocol."""
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

from benchmark.common import ROOT, configuration, file_digest, read_json, write_json
from benchmark.dit_cases import definitions, select_cases
from benchmark.execution import cpu_parent, run_function, runtime_device
from benchmark.model_profiles import profile, resolve_scheduler
from benchmark.plugin_plans import plugin_plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("sd35_medium", "flux1_dev"), default="sd35_medium")
    p.add_argument("--cases", default="all", help="all/i2p/ip/style/object or comma-separated case IDs/groups")
    p.add_argument("--tasks", help="Optional task filter, e.g. mscoco")
    p.add_argument("--phase", choices=("all", "train", "prepare", "sample", "evaluate"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-root", type=Path, default=METHOD_ROOT / "runs/benchmark-dit")
    p.add_argument("--method", default="GEM")
    p.add_argument("--base-model", help="Optional local Diffusers directory or Hub model ID")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--text-device", default="cpu")
    p.add_argument("--metric-device")
    p.add_argument("--budget-per-concept", type=int, default=250, help="Native GEM velocity-prediction budget per target")
    p.add_argument("--budget", type=int, help="Override the total case budget; this is not the optimizer step count")
    p.add_argument("--lr", type=float, help="Default: 2e-4 for SD3.5, 1e-3 for FLUX")
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=100.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--trajectory-end", type=int, default=10, help="Native callback's last timestep index, inclusive")
    p.add_argument("--training-steps", type=int, help="Default: model sampling profile")
    p.add_argument("--training-resolution", type=int, default=512)
    p.add_argument("--training-sequence-length", type=int, help="Default: model sampling profile")
    p.add_argument("--training-dtype", choices=("bfloat16", "float32", "float16"), default="bfloat16")
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--rollout-guidance", type=float, default=3.0)
    p.add_argument("--max-parallel-timesteps", type=int, default=6)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--metric-batch-size", type=int, default=32)
    p.add_argument("--cpu-offload", action="store_true", help="Sampling only; training uses --device")
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
    p.add_argument("--max-records", type=int, help="Explicit reduced validation protocol")
    p.add_argument("--dtype", choices=("bfloat16", "float32", "float16"))
    p.add_argument("--steps", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    return p


def case_configs(args, case_id, case):
    selected = profile(args.model)
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    sources = [Path(__file__), *(METHOD_ROOT / "gem" / name for name in (
        "operators/erasure/erasure_gem.py", "common/stratified.py", "wrappers/factory.py",
        "wrappers/base_pipeline_wrapper.py", "wrappers/pipeline_wrapper_sd3.py",
        "wrappers/pipeline_wrapper_flux.py", "wrappers/parameter_subset_selector.py", "utils/memory.py"))]
    settings = {
        "model_name": args.model, "base_model": args.base_model or selected["base_model"],
        "model_revision": args.model_revision, "case_id": case_id, "targets": case["targets"],
        "anchors": [anchors[target] for target in case["targets"]],
        "budget": args.budget or args.budget_per_concept * len(case["targets"]),
        "lr": args.lr if args.lr is not None else (2e-4 if args.model == "sd35_medium" else 1e-3),
        "eta": args.eta, "gamma": args.gamma, "rank": args.rank, "lora_dropout": args.lora_dropout,
        "subset": "qk_dual", "trajectory_end": args.trajectory_end,
        "training_steps": args.training_steps or selected["sampling"]["steps"],
        "training_resolution": args.training_resolution,
        "training_sequence_length": args.training_sequence_length or selected["sampling"]["max_sequence_length"],
        "training_dtype": args.training_dtype, "training_seed": args.training_seed,
        "rollout_guidance": args.rollout_guidance, "max_parallel_timesteps": args.max_parallel_timesteps,
        "gradient_checkpointing": args.gradient_checkpointing, "device": args.device, "text_device": args.text_device,
        "sources": {str(path.relative_to(METHOD_ROOT)): file_digest(path) for path in sources}}
    configs = []
    for task in case["tasks"]:
        if args.tasks and task not in args.tasks:
            continue
        cfg = {
            "case_id": case_id, "method": args.method, "base_model": settings["base_model"],
            "output_dir": str(args.output_dir), "cache_dir": str(args.cache_dir),
            "original_dir": str(args.original_dir or ROOT / "original" / args.model),
            "dataset": {}, "erasure_parameters": settings,
            "adapter": {"kind": "lora", "checkpoint": str(args.run_root / args.model / case_id / "checkpoint")},
            "sampling": {"device": args.device, "batch_size": args.batch_size, "cpu_offload": args.cpu_offload,
                         "local_files_only": args.offline, "model_revision": args.model_revision},
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
        configs.append(configuration(task=task, model=args.model, base=cfg))
    return settings, configs


def train(checkpoint, settings, *, offline=False):
    """Use the author's GEM operator and PEFT adaptation; export a native DiT LoRA."""
    from dataclasses import asdict, replace
    from importlib.metadata import version
    import torch
    from diffusers import FluxPipeline, StableDiffusion3Pipeline
    from peft import get_peft_model_state_dict
    from gem.configs import AdaptationConfig, PipelineConfig
    from gem.operators.erasure import GEM, GEMConfig, GEMSD3Config
    from gem.utils.logging import setup_logger
    from gem.utils.seed import seed_everything
    from gem.wrappers.factory import PipelineFactory

    checkpoint = Path(checkpoint)
    weights = checkpoint / "pytorch_lora_weights.safetensors"
    if checkpoint.exists():
        info = read_json(checkpoint / "metadata.json")
        if info["settings"] != settings or info["weights_sha256"] != file_digest(weights):
            raise ValueError(f"Checkpoint differs from this request; use another --run-root: {checkpoint}")
        print(f"[reuse training] {checkpoint}", flush=True)
        return
    seed_everything(settings["training_seed"])
    checkpoint.mkdir(parents=True)
    setup_logger("GEM-benchmark", str(checkpoint / "train.log"))
    flux = settings["model_name"] == "flux1_dev"
    cls = FluxPipeline if flux else StableDiffusion3Pipeline
    device, text_device = runtime_device(settings["device"]), runtime_device(settings["text_device"])
    pipe = cls.from_pretrained(settings["base_model"], torch_dtype=getattr(torch, settings["training_dtype"]),
                               revision=settings["model_revision"], local_files_only=offline)
    pipe.transformer.to(device)
    pipe.vae.to(device)
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.to(text_device)
    pipe.set_progress_bar_config(disable=True)
    pipe.pipe_config = replace(PipelineConfig.default_for_model_type("flux" if flux else "sd_3_5"),
        model_name_or_path=settings["base_model"], model_dtype=settings["training_dtype"],
        image_size=settings["training_resolution"], num_inference_steps=settings["training_steps"])
    wrapper = PipelineFactory.wrap_pipeline(pipe)
    wrapper.text_device = text_device
    wrapper.max_sequence_length = settings["training_sequence_length"]
    wrapper.rollout_cfg = not flux and settings["rollout_guidance"] > 1
    adaptation = AdaptationConfig(adapter_mode="lora", subset_name=settings["subset"], lora_config={
        "r": settings["rank"], "lora_alpha": settings["rank"], "lora_dropout": settings["lora_dropout"], "bias": "none"})
    wrapper = PipelineFactory.prepare_adaptation(adaptation, wrapper)
    if settings["gradient_checkpointing"]:
        pipe.transformer.enable_gradient_checkpointing()
    config = (GEMConfig if flux else GEMSD3Config)(
        targets=settings["targets"], anchors=settings["anchors"], adaptation_config=adaptation,
        train_seed=settings["training_seed"], max_step_budget=settings["budget"], learning_rate=settings["lr"],
        eta=settings["eta"], gamma=settings["gamma"], max_parallel_timesteps=settings["max_parallel_timesteps"],
        partial_denoising_end_at_timestep=settings["trajectory_end"],
        partial_denoising_inference_timesteps=settings["training_steps"],
        partial_denoising_guidance_scale=settings["rollout_guidance"],
        use_wandb=False, initial_validation=False, run_save_path=str(checkpoint))
    operator = GEM()
    operator(wrapper, config)
    # PEFT wraps the author's whole pipeline wrapper. Strip only its transformer prefix
    # so native SD3/FLUX LoRA loaders can consume the same trained tensors directly.
    state = {key.split("transformer.", 1)[1]: value for key, value in get_peft_model_state_dict(wrapper).items()}
    cls.save_lora_weights(str(checkpoint), transformer_lora_layers=state)
    write_json(checkpoint / "metadata.json", {"settings": settings, "operator_config": asdict(config),
        "target_trajectory_counts": operator.target_counts, "weights_sha256": file_digest(weights),
        "versions": {name: version(name) for name in ("torch", "diffusers", "peft")}})
    print(f"[saved] {weights}; target trajectories: {operator.target_counts}", flush=True)


def run(args, experiments, env):
    env["PYTHONPATH"] = os.pathsep.join((str(METHOD_ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")))
    for case_id, settings, configs in experiments:
        case_root = args.run_root / args.model / case_id
        checkpoint = case_root / "checkpoint"
        print(f"[GEM:{args.model}:{case_id}] {settings['targets']} -> {settings['anchors']}; budget={settings['budget']}", flush=True)
        if args.phase in {"all", "train"}:
            print(f"[train] {checkpoint}", flush=True)
            if not args.dry_run:
                run_function("benchmark_dit:train", {"checkpoint": str(checkpoint), "settings": settings, "offline": args.offline},
                             env=env, cwd=WORKSPACE, cpu_threads=args.cpu_threads)
        if args.phase == "train":
            continue
        if not args.dry_run and args.phase != "all":
            info = read_json(checkpoint / "metadata.json")
            if info["settings"] != settings or info["weights_sha256"] != file_digest(checkpoint / "pytorch_lora_weights.safetensors"):
                raise ValueError("Checkpoint does not match the requested GEM experiment; use its original settings.")
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
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    for name in ("base_model", "vlm_model"):
        value = getattr(args, name)
        if value and (value.startswith((".", "/", "~")) or Path(value).exists()):
            setattr(args, name, str(Path(value).expanduser().resolve()))
    positive = (args.budget_per_concept, args.rank, args.gamma, args.trajectory_end, args.cpu_threads, args.batch_size,
                args.metric_batch_size, args.training_resolution, args.max_parallel_timesteps)
    optional = (args.budget, args.lr, args.training_steps, args.training_sequence_length, args.max_records)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or any(v is not None and (not math.isfinite(v) or v <= 0) for v in optional):
        p.error("Budgets, learning rate, dimensions and batch/thread counts must be positive.")
    if (not math.isfinite(args.eta) or args.eta < 0 or not math.isfinite(args.rollout_guidance) or args.rollout_guidance < 0
            or not 0 <= args.lora_dropout < 1
            or args.training_resolution % 16 or not 0 <= args.training_seed < 2**32):
        p.error("Use nonnegative guidance/eta, dropout in [0,1), resolution divisible by 16 and a uint32 seed.")
    args.tasks = [task.strip() for task in args.tasks.split(",")] if args.tasks else None
    cases = definitions()["cases"]
    experiments = [(case_id, *case_configs(args, case_id, cases[case_id])) for case_id in select_cases(args.cases)]
    if args.tasks and set(args.tasks) - {cfg["task"] for _, _, configs in experiments for cfg in configs}:
        p.error("--tasks includes tasks absent from the selected cases.")
    experiments = [entry for entry in experiments if entry[2]]
    for _, settings, _ in experiments:
        if settings["trajectory_end"] >= settings["training_steps"] or settings["training_sequence_length"] > 512:
            p.error("Trajectory end must be below training steps; T5 sequence length must be <=512.")
        if args.model == "flux1_dev" and args.training_resolution < 128:
            p.error("The native FLUX wrapper requires training resolution >=128.")
    with cpu_parent(args, p) as env:
        run(args, experiments, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
