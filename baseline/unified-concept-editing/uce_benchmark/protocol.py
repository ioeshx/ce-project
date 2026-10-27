"""Read the actual benchmark JSONs instead of duplicating experiment targets."""
from pathlib import Path

from benchmark.common import ROOT, configuration, file_digest
from benchmark.experiment_cases import CASE_CONFIGS, INSTANCE_CASES, STYLE_CASES
from . import METHOD_ROOT

EXPERIMENTS = ("instance", "style", "i2p_q16", "i2p_nudenet", "mscoco")
I2P_CASES = ("i2p_q16", "i2p_nudenet")


def source_files():
    paths = [METHOD_ROOT / "trainscripts/uce_sd_erase.py",
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


def edit_settings(case, args):
    cfg = case_config(case, args)
    guide = args.guide if args.guide is not None else cfg.get("erasure_anchor", " ")
    preserve = [v.strip() for v in args.preserve_concepts.split(";") if v.strip()]
    if any(t.casefold() == p.casefold() for t in cfg["targets"] for p in preserve):
        raise ValueError("Do not preserve a concept that is being erased.")
    if any(t.casefold() == guide.casefold() for t in cfg["targets"]):
        raise ValueError("The guide cannot equal an erased concept.")
    return {"base_model": cfg["base_model"], "model_revision": args.model_revision,
            "case": case, "task": cfg["task"], "targets": cfg["targets"],
            "guide_concepts": [guide] * len(cfg["targets"]), "preserve_concepts": preserve,
            "erase_scale": args.erase_scale, "preserve_scale": args.preserve_scale,
            "lamb": args.lamb, "expand_prompts": False, "edited_modules": "attn2.to_k + attn2.to_v",
            "weight_dtype": "float32", "device": args.device, "sources": source_files()}
