"""CASteer on SD3.5 Medium / FLUX.1-dev with the shared DiT benchmark."""
import argparse
import copy
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
from benchmark.execution import cpu_parent, run_function
from benchmark.model_profiles import profile, resolve_scheduler
from benchmark.plugin_plans import plugin_plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("sd35_medium", "flux1_dev"), default="sd35_medium")
    p.add_argument("--cases", default="all", help="all/i2p/ip/style/object or comma-separated case IDs/groups")
    p.add_argument("--tasks", help="Optional task filter, e.g. mscoco")
    p.add_argument("--phase", choices=("all", "fit", "prepare", "sample", "evaluate"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-root", type=Path, default=METHOD_ROOT / "runs/benchmark-dit")
    p.add_argument("--method", default="CASteer")
    p.add_argument("--base-model", help="Optional local Diffusers directory or matching Hub ID")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--metric-device")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--metric-batch-size", type=int, default=32)
    p.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=None,
                   help="Default enabled on CUDA; offload inactive pipeline components to CPU.")
    p.add_argument("--seed", type=int, default=0, help="Method calibration seed; evaluation keeps its per-record seeds.")
    p.add_argument("--calibration-steps", type=int, help="Explicit method-fit override; default shared sampling steps.")
    p.add_argument("--calibration-prompts", type=int, help="Default concrete/style=50; human-related uses all 210 native pairs")
    p.add_argument("--strength", type=float, default=2.0)
    p.add_argument("--intermediate-clipping", action=argparse.BooleanOptionalAction, default=True)
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
    p.add_argument("--max-records", type=int, help="Reduced validation protocol, not full benchmark scores")
    p.add_argument("--dtype", choices=("bfloat16", "float32", "float16"))
    p.add_argument("--steps", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--max-sequence-length", type=int)
    return p


def case_configs(args, case_id, case):
    selected = profile(args.model)
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    sources = [Path(__file__), *(METHOD_ROOT / name for name in ("core/controller.py", "core/vector_dump.py", "core/diffusion_steering.py", "core/construct_prompts.py", "imagenet_classes.txt")), ROOT / "configs/dit_target_anchor.json"]
    settings = {
        "implementation": "CASteer-DiT-v1", "model_name": args.model,
        "base_model": args.base_model or selected["base_model"], "model_revision": args.model_revision,
        "case_id": case_id, "group": case["group"], "targets": case["targets"],
        "anchors": [anchors[target] for target in case["targets"]], "seed": args.seed,
        "calibration_prompts": args.calibration_prompts, "strength": args.strength,
        "intermediate_clipping": args.intermediate_clipping,
        "sources": {str(path): file_digest(path) for path in sources}}
    configs = []
    for task in case["tasks"]:
        if args.tasks and task not in args.tasks:
            continue
        cfg = {
            "model_name": args.model, "case_id": case_id, "method": args.method,
            "base_model": settings["base_model"], "output_dir": str(args.output_dir), "cache_dir": str(args.cache_dir),
            "original_dir": str(args.original_dir or ROOT / "original" / args.model),
            "erasure_parameters": settings,
            "adapter": {"kind": "plugin", "factory": "benchmark_dit:CASteerAdapter",
                        "source_files": ["core/controller.py", "core/vector_dump.py", "core/diffusion_steering.py"],
                        "checkpoint": str(args.run_root / args.model / case_id / "checkpoint")},
            "sampling": {"device": args.device, "batch_size": args.batch_size,
                         "cpu_offload": args.cpu_offload, "local_files_only": args.offline,
                         "model_revision": args.model_revision},
            "evaluation": {"device": args.metric_device or args.device, "batch_size": args.metric_batch_size,
                           "local_files_only": args.offline}}
        for key in ("dtype", "steps", "height", "width", "max_sequence_length"):
            if getattr(args, key) is not None:
                cfg["sampling"][key] = getattr(args, key)
        for key in ("vlm_model", "vlm_device", "resnet_weights_path", "q16_prompts", "q16_clip_path", "nudenet_weights"):
            if getattr(args, key) is not None:
                cfg["evaluation"][key] = str(getattr(args, key))
        if args.skip_vlm:
            cfg["evaluation"]["vlm_enabled"] = False
        if args.with_q16 and task == "i2p":
            cfg.setdefault("dataset", {})["i2p_subset"] = "all"
            cfg["evaluation"]["q16_enabled"] = True
        if args.max_records is not None:
            cfg.setdefault("dataset", {})["max_records"] = args.max_records
        configs.append(configuration(task=task, model=args.model, base=cfg))
    if configs:
        calibration = copy.deepcopy(configs[0]["sampling"])
        calibration["steps"] = args.calibration_steps or calibration["steps"]
        calibration["batch_size"] = 1
        settings["calibration_sampling"] = calibration
    return settings, configs


def run(args, experiments, env):
    env["PYTHONPATH"] = os.pathsep.join((str(METHOD_ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")))
    for case_id, settings, configs in experiments:
        case_root = args.run_root / args.model / case_id
        checkpoint = case_root / "checkpoint"
        metadata = checkpoint / "metadata.json"
        print(f"[CASteer:{args.model}:{case_id}] {settings['targets']} -> {settings['anchors']}", flush=True)
        if not args.dry_run:
            if metadata.is_file():
                info = read_json(metadata)
                if info["settings"] != settings or info["weights_sha256"] != file_digest(checkpoint / "steering.pt"):
                    raise ValueError(f"Stored method settings/weights differ; use a new --run-root: {checkpoint}")
            elif args.phase not in {"all", "fit"}:
                raise ValueError(f"Run --phase fit first: {checkpoint}")
        if args.phase in {"all", "fit"}:
            print(f"[fit] {checkpoint}", flush=True)
            if not args.dry_run and not metadata.is_file():
                run_function("benchmark_dit:fit", {"checkpoint": str(checkpoint), "settings": settings, "config": configs[0]},
                             env=env, cwd=METHOD_ROOT, cpu_threads=args.cpu_threads)
        if args.phase == "fit":
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
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    for name in ("base_model", "vlm_model"):
        value = getattr(args, name)
        if value and (value.startswith((".", "/", "~")) or Path(value).exists()):
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if args.cpu_offload is None:
        args.cpu_offload = args.device.startswith("cuda")
    positive = [args.cpu_threads, args.batch_size, args.metric_batch_size]
    optional = [args.steps, args.height, args.width, args.max_records, args.calibration_steps,
                args.max_sequence_length, args.calibration_prompts]
    if any(not math.isfinite(v) or v <= 0 for v in positive) or any(v is not None and v <= 0 for v in optional):
        p.error("Step counts, dimensions, method sizes and batch/thread counts must be positive.")
    if not 0 <= args.seed < 2**32 or any(v is not None and v % 16 for v in (args.height, args.width)):
        p.error("Use a uint32 seed and image dimensions divisible by 16.")
    if args.device == "cpu" and (args.dtype != "float32" or args.cpu_offload):
        p.error("CPU checks require --dtype float32 --no-cpu-offload.")
    if not math.isfinite(args.strength) or args.strength < 0:
        p.error("Use a finite, nonnegative steering strength.")
    if args.calibration_steps is not None and args.calibration_steps != (args.steps or profile(args.model)["sampling"]["steps"]):
        p.error("Set --steps to the same value as --calibration-steps for CASteer.")
    args.tasks = args.tasks.split(",") if args.tasks else None
    cases = definitions()["cases"]
    experiments = [(key, *case_configs(args, key, cases[key])) for key in select_cases(args.cases)]
    if args.tasks and set(args.tasks) - {cfg["task"] for _, _, configs in experiments for cfg in configs}:
        p.error("--tasks includes tasks absent from the selected cases.")
    experiments = [entry for entry in experiments if entry[2]]
    with cpu_parent(args, p) as env:
        run(args, experiments, env)
    return 0


def fit(checkpoint, settings, config):
    """Estimate paired activation differences, without optimizing model weights."""
    import torch
    from core.construct_prompts import get_prompts_concrete, get_prompts_style, get_prompts_human_related
    from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
    from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode
    from benchmark.dit_adapter import DiTAdapter
    torch.manual_seed(settings["seed"])
    cfg = copy.deepcopy(config)
    cfg["adapter"] = {"kind": "original"}
    cfg["sampling"] = copy.deepcopy(settings["calibration_sampling"])
    adapter = DiTAdapter(cfg)
    vectors, counts = [], []
    model_type = DiffusionModelType.from_model(settings["model_name"])
    for target, anchor in zip(settings["targets"], settings["anchors"]):
        if settings["group"] == "i2p":
            positive, negative = get_prompts_human_related(concept_pos=target, concept_neg=anchor)
            if settings["calibration_prompts"] is not None:
                positive, negative = positive[:settings["calibration_prompts"]], negative[:settings["calibration_prompts"]]
        else:
            construct = get_prompts_style if settings["group"] == "style" else get_prompts_concrete
            positive, negative = construct(num=settings["calibration_prompts"] or 50, concept_pos=target, concept_neg=anchor)
        means = []
        for label, prompts in (("positive", positive), ("anchor", negative)):
            stats = CrossAttentionOutputStatsCollector(token_aggregation_mode=TokenAggregationMode.AVERAGE,
                classifier_free_guidance=not adapter.flux)
            manager = diffusion_register_vector_controls_with_hooks(adapter.pipe.transformer, stats, model_type=model_type)
            try:
                for i, prompt in enumerate(prompts):
                    adapter.generate([{"prompt": prompt, "seed": settings["seed"]}], cfg["sampling"])
                    stats.reset()
                    print(f"[CASteer:{target}:{label}] {i + 1}/{len(prompts)}", flush=True)
                means.append(stats.means)
            finally:
                manager.remove_hooks()
        # Original normalized mean-difference formula, one direction per step/layer.
        steering = {}
        for step, places in means[0].items():
            steering[step] = {}
            for place, layers in places.items():
                steering[step][place] = []
                for pos, neg in zip(layers, means[1][step][place]):
                    direction = pos - neg
                    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    steering[step][place].append(direction.cpu())
        vectors.append(steering)
        counts.append({"target": target, "anchor": anchor, "pairs": len(positive),
                       "positive_prompts": positive, "anchor_prompts": negative})
    adapter.close()
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, checkpoint / "steering.pt")
    write_json(checkpoint / "metadata.json", {"settings": settings, "weights_sha256": file_digest(checkpoint / "steering.pt"),
               "calibration": counts, "calibration_sampling": adapter.config["sampling"],
               "intervention": "projected text-value contribution to image tokens; joint softmax normalization retained"})


class CASteerAdapter:
    """One benchmark adapter; delegate sampling unchanged to native DiTAdapter."""
    def __init__(self, config):
        import torch
        from benchmark.dit_adapter import DiTAdapter
        from core.controller import CrossAttentionOutputSteering
        from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
        checkpoint = Path(config["adapter"]["checkpoint"])
        settings = read_json(checkpoint / "metadata.json")["settings"]
        cfg = copy.deepcopy(config)
        cfg["adapter"] = {"kind": "original"}
        self.sampler = DiTAdapter(cfg)
        vectors = torch.load(checkpoint / "steering.pt", map_location="cpu", weights_only=True)
        # Keep compact vectors in CPU memory with offloaded models. Each layer's
        # small direction is transferred by the controller at its point of use.
        control = CrossAttentionOutputSteering(source_concepts=vectors, target_concepts=[None] * len(vectors),
            strength=settings["strength"], device="cpu", intermediate_clipping=settings["intermediate_clipping"],
            classifier_free_guidance=not self.sampler.flux, precompute_matrix=False)
        self.manager = diffusion_register_vector_controls_with_hooks(self.sampler.pipe.transformer, control,
            model_type=DiffusionModelType.from_model(config["model_name"]))

    def generate(self, records, sampling_config):
        self.manager.reset_controls()
        return self.sampler.generate(records, sampling_config)

    def close(self):
        self.manager.remove_hooks()
        self.sampler.close()


if __name__ == "__main__":
    raise SystemExit(main())
