"""Both method suites read their erasure cases from executable benchmark JSONs."""

from .common import ROOT, concept_key, configuration

INSTANCE_CASES = ("instance_snoopy", "instance_snoopy_mickey", "instance_snoopy_mickey_spongebob")
STYLE_CASES = ("style_van_gogh", "style_monet", "style_picasso")
MSCOCO_CASES = ("mscoco_cifar50",)
DEFAULT_CASES = (*INSTANCE_CASES, *STYLE_CASES, *MSCOCO_CASES)
CASE_CONFIGS = {
    "instance_snoopy": ROOT / "configs/instance.json",
    "instance_snoopy_mickey": ROOT / "configs/instance_snoopy_mickey.json",
    "instance_snoopy_mickey_spongebob": ROOT / "configs/instance_snoopy_mickey_spongebob.json",
    "style_van_gogh": ROOT / "configs/style.json",
    "style_monet": ROOT / "configs/style_monet.json",
    "style_picasso": ROOT / "configs/style_picasso.json",
    "mscoco_cifar50": ROOT / "configs/mscoco.json",
}
_CONFIGS = {case: configuration(path) for case, path in CASE_CONFIGS.items()}
CASE_TARGETS = {case: (config.get("erasure_task", config["task"]), tuple(config["targets"]))
                for case, config in _CONFIGS.items()}
CASE_TARGETS.update({
    # Explicit extra cases remain available for older commands.
    "instance_mickey": ("instance", ("Mickey",)),
    "instance_spongebob": ("instance", ("SpongeBob",)),
})
DEFAULT_ANCHOR = " "
CASE_ANCHORS = {case: _CONFIGS.get(case, {}).get("erasure_anchor", DEFAULT_ANCHOR)
                for case in CASE_TARGETS}


def evaluation_configuration(task, targets):
    """Read task/case defaults; the actual checkpoint remains target authority."""
    wanted = {concept_key(t) for t in targets}
    path = ROOT / "configs" / f"{task}.json"
    for case, candidate in CASE_CONFIGS.items():
        if _CONFIGS[case]["task"] == task and {concept_key(t) for t in CASE_TARGETS[case][1]} == wanted:
            path = candidate
            break
    config = configuration(path)
    if {concept_key(t) for t in config["targets"]} != wanted:
        # A historical/custom checkpoint may use the task defaults while its
        # actual target set differs from the named default case.
        config.pop("targets_from", None)
        config.pop("description", None)
    config["targets"] = list(targets)
    config["configuration_source"] = str(path)
    return config
