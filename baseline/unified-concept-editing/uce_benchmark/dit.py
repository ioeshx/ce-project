"""Four DiT experiment groups: edit UCE, then reuse benchmark sampling/evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from benchmark.common import ROOT, configuration, file_digest, read_json
from benchmark.dit_cases import definitions, select_cases
from benchmark.model_profiles import profile, resolve_scheduler
from benchmark.plugin_plans import plugin_plan
from . import METHOD_ROOT, WORKSPACE


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("sd35_medium", "flux1_dev"), default="sd35_medium")
    p.add_argument("--cases", default="all", help="all/i2p/ip/style/object or comma-separated case IDs")
    p.add_argument("--tasks", help="Optional task filter, e.g. mscoco for the nudity checkpoint")
    p.add_argument("--phase", choices=("all", "erase", "prepare", "sample", "evaluate"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-root", type=Path, default=METHOD_ROOT / "runs/benchmark-dit")
    p.add_argument("--method", default="UCE", help="Method name within each model/case's CSV")
    p.add_argument("--base-model", help="Optional local Diffusers base model directory")
    p.add_argument("--model-revision")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--edit-device", help="Text encoder device; defaults to --device. Solve always uses CPU.")
    p.add_argument("--metric-device", help="Defaults to --device")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--metric-batch-size", type=int, default=32)
    p.add_argument("--cpu-offload", action="store_true", help="Offload the sampling pipeline")
    p.add_argument("--device-map", choices=("balanced", "auto"))
    p.add_argument("--max-memory", type=json.loads)
    p.add_argument("--gpu-memory-reserve", type=float)
    p.add_argument("--erase-scale", type=float, default=10.0)
    p.add_argument("--preserve-scale", type=float, default=1.0)
    p.add_argument("--lamb", type=float, default=0.1)
    p.add_argument("--anchor", help="Override all per-target guides from dit_target_anchor.json")
    p.add_argument("--preserve-concepts", default="", help="Additional retain names separated by ;")
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
    """Use the live DiT JSONs, with only UCE and execution options added."""
    selected = profile(args.model)
    targets = case["targets"]
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    guides = [args.anchor if args.anchor is not None else anchors[target] for target in targets]
    settings = {"model_name": args.model, "base_model": args.base_model or selected["base_model"],
                "model_revision": args.model_revision, "case_id": case_id, "targets": targets,
                "guides": guides, "preserve_concepts": args.preserve_concepts,
                "erase_scale": args.erase_scale, "preserve_scale": args.preserve_scale, "lamb": args.lamb,
                "embedding_dtype": args.dtype or selected["sampling"]["dtype"],
                "max_sequence_length": selected["sampling"]["max_sequence_length"],
                "edit_device": args.edit_device or args.device,
                "sources": {path.name: file_digest(path) for path in (
                    Path(__file__), Path(__file__).with_name("dit_editing.py"),
                    METHOD_ROOT / "trainscripts/uce_flux_edit.py")}}
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
        cfg["adapter"] = {"kind": "plugin", "factory": "uce_benchmark.dit_editing:create",
                          "checkpoint": str(args.run_root / args.model / case_id / "checkpoint")}
        cfg["erasure_parameters"] = settings
        if len(set(guides)) == 1:
            cfg["erasure_anchor"] = guides[0]
        configs.append(configuration(task=task, model=args.model, base=cfg))
    return settings, configs


def run(args, experiments):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(METHOD_ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")))
    for case_id, settings, configs in experiments:
        case_root = args.run_root / args.model / case_id
        print(f"[UCE:{args.model}:{case_id}] {settings['targets']} -> {settings['guides']}", flush=True)
        if args.phase in {"all", "erase"}:
            print(f"[edit] {case_root / 'checkpoint'}", flush=True)
            if not args.dry_run:
                import torch
                from .dit_editing import edit
                torch.set_num_threads(args.cpu_threads)
                edit(case_root / "checkpoint", settings, offline=args.offline)
        if args.phase == "erase":
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
    positive = (args.erase_scale, args.lamb, args.cpu_threads, args.batch_size, args.metric_batch_size)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not math.isfinite(args.preserve_scale) or args.preserve_scale < 0:
        p.error("Scales, thread/batch counts and lambda must be positive; preserve-scale may be zero.")
    args.preserve_concepts = [c.strip() for c in args.preserve_concepts.split(";") if c.strip()]
    args.tasks = [task.strip() for task in args.tasks.split(",")] if args.tasks else None
    cases = definitions()["cases"]
    selected = select_cases(args.cases)
    experiments = [(case_id, *case_configs(args, case_id, cases[case_id])) for case_id in selected]
    if args.tasks and set(args.tasks) - {cfg["task"] for _, _, configs in experiments for cfg in configs}:
        p.error("--tasks includes tasks absent from the selected cases.")
    experiments = [entry for entry in experiments if entry[2]]
    for _, settings, _ in experiments:
        if set(map(str.casefold, settings["targets"])) & set(map(str.casefold, args.preserve_concepts)):
            p.error("An erased target cannot also be a preserve concept.")
        if any(target.casefold() == guide.casefold() for target, guide in zip(settings["targets"], settings["guides"])):
            p.error("A target cannot be its own anchor.")
    run(args, experiments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
