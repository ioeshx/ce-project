"""Text concepts for the independent CIFAR-50 -> MS-COCO retention experiment.

CIFAR names are erasure/calibration concepts; no CIFAR images are evaluated.
"""
import re

from .dataset.prompt_templates import CIFAR50_classes, CIFAR100_classes

CIFAR50_TARGETS = tuple(CIFAR50_classes)
CONCEPT_SOURCE = "benchmark/dataset/prompt_templates.py"
if len(CIFAR50_TARGETS) != 50 or len(set(CIFAR50_TARGETS)) != 50:
    raise ValueError("CIFAR50_classes must contain exactly 50 unique concepts.")


def evaluation_tasks(erasure_task):
    if erasure_task == "object":
        return ("mscoco",)
    if erasure_task in {"instance", "style"}:
        return (erasure_task,)
    raise ValueError(f"Unsupported erasure task: {erasure_task}")


def target_label(targets):
    if len(targets) == 50 and {t.casefold() for t in targets} == {t.casefold() for t in CIFAR50_TARGETS}:
        return "CIFAR-50"
    if len(targets) > 3:
        return f"{len(targets)}-targets"
    return "-".join(targets) or "all"


def object_retain_concepts(targets):
    """CIFAR-100 concepts disjoint from the erased names, in source order."""
    def overlaps(a, b):
        return re.search(r"(?<!\w)" + re.escape(a.casefold()) + r"(?!\w)", b.casefold())
    return [c for c in CIFAR100_classes if not any(overlaps(t, c) or overlaps(c, t) for t in targets)]


def erasure_protocol(task, targets):
    result = {"erasure_task": task, "targets": list(targets),
              "evaluation_tasks": list(evaluation_tasks(task))}
    if task == "object":
        canonical = {t.casefold() for t in targets} == {t.casefold() for t in CIFAR50_TARGETS}
        result.update(name="cifar50-to-mscoco1k" if canonical else "custom-objects-to-mscoco",
                      target_source=CONCEPT_SOURCE if canonical else "explicit --targets",
                      target_variable="CIFAR50_classes" if canonical else None,
                      retain_source=CONCEPT_SOURCE, retain_variable="CIFAR100_classes",
                      retain_selection="exclude erased names and overlapping aliases",
                      evaluation_dataset="mscoco.csv", evaluation_limit=1000,
                      evaluation_seed="caption row index (zero-based)")
    return result
