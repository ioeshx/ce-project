"""Read the actual benchmark JSONs instead of duplicating experiment targets."""
from pathlib import Path

from benchmark.common import ROOT, configuration, file_digest
from benchmark.experiment_cases import CASE_CONFIGS, INSTANCE_CASES, STYLE_CASES
from . import METHOD_ROOT

EXPERIMENTS = ("instance", "style", "i2p_q16", "i2p_nudenet", "mscoco")
I2P_CASES = ("i2p_q16", "i2p_nudenet")


def source_files():
    paths = [METHOD_ROOT / "esd_sd.py", METHOD_ROOT / "utils/esd_trainer.py",
             METHOD_ROOT / "utils/sd_utils.py", METHOD_ROOT / "utils/esd_checkpoint.py",
             *sorted(Path(__file__).parent.glob("*.py")),
             ROOT / "sampling_profile.py", ROOT / "configs/sd14_sampling.json"]
    return {str(p): file_digest(p) for p in paths}


def cases(experiments):
    groups = {"instance": INSTANCE_CASES, "style": STYLE_CASES,
              "mscoco": ("mscoco_cifar50",), **{name: (name,) for name in I2P_CASES}}
    names = [v.strip() for v in experiments.split(",")]
    if not names or len(set(names)) != len(names) or any(n not in groups for n in names):
        raise ValueError(f"Choose unique experiments from {', '.join(EXPERIMENTS)}")
    return [case for group in names for case in groups[group]]


def case_config(case, args):
    if case in I2P_CASES:
        path = ROOT / "configs/i2p.json"
        cfg = configuration(path)
        values = args.q16_targets if case == "i2p_q16" else args.nudenet_targets
        cfg["targets"] = [v.strip() for v in values.split(";")]
        cfg["erasure_task"] = "object"
        cfg["erasure_anchor"] = " "
    else:
        if case not in CASE_CONFIGS:
            raise ValueError(f"Unknown case: {case}")
        path = CASE_CONFIGS[case]
        cfg = configuration(path)
    if not cfg["targets"] or any(not t for t in cfg["targets"]) or len(set(cfg["targets"])) != len(cfg["targets"]):
        raise ValueError("Erasure targets must be nonempty and unique.")
    cfg["configuration_source"] = str(path)
    return cfg


def training_settings(case, args):
    cfg = case_config(case, args)
    from benchmark.sampling_profile import defaults
    method = args.train_method or ("esd-u" if case in I2P_CASES else "esd-x")
    iterations = args.iterations or args.iterations_per_concept * len(cfg["targets"])
    if iterations < len(cfg["targets"]):
        raise ValueError("Training iterations must cover every target at least once.")
    rollout = defaults()
    rollout.update(steps=args.training_steps, guidance_scale=args.training_guidance_scale,
                   height=args.training_resolution, width=args.training_resolution, dtype=args.training_dtype)
    if "execution_backend" in rollout:
        rollout["execution_backend"] = str(args.device).split(":", 1)[0]
    return {"base_model": cfg["base_model"], "model_revision": args.model_revision,
            "case": case, "task": cfg["task"], "targets": cfg["targets"],
            "train_method": method, "iterations": iterations, "lr": args.lr,
            "negative_guidance": args.negative_guidance, "training_seed": args.training_seed,
            "batch_size": 1, "erase_from": None, "target_selection": "shuffled cycles; one target per step",
            "training_protocol": rollout,
            "training_exception": "ESD optimization uses float32 by default for Adam stability; deployment uses shared float16 profile.",
            "device": args.device, "sources": source_files()}
