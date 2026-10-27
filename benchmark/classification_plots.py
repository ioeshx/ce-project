"""Confusion matrices rebuilt from aggregate CSV counts alone."""
from __future__ import annotations

import numpy as np

from .common import concept_key


def figures(row, task):
    import matplotlib.pyplot as plt
    kind = "resnet" if task == "object" else "vlm"
    if kind == "vlm":
        from .vlm import LABELS
        labels = LABELS[task]
        concepts = [concept_key(s) for s in labels[:-1]]
        predicted = [concept_key(s) for s in labels]
    else:
        from .object_classifier import IMAGENET10
        concepts = list(IMAGENET10)
        predicted = [str(IMAGENET10[c]) for c in concepts] + ["other"]
        labels = concepts + ["other 990 classes"]
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 1.6), max(4, len(concepts) * .5 + 2)))
    for ax, prefix, title in zip(axes, ("reference_", ""), ("Original model", "Method")):
        matrix = np.zeros((len(concepts), len(predicted)))
        for i, concept in enumerate(concepts):
            start = f"{prefix}{kind}.confusion.{concept}."
            for key, value in row.items():
                if key.startswith(start) and key.endswith(".count") and value not in (None, ""):
                    label = key[len(start):-6]
                    column = predicted.index(label) if label in predicted else len(predicted) - 1
                    matrix[i, column] += float(value)
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(concepts)):
            for j in range(len(predicted)):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7,
                        color="white" if normalized[i, j] > .55 else "black")
        ax.set_xticks(range(len(predicted)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(concepts)), concepts)
        ax.set(xlabel="Predicted label", ylabel="Requested concept", title=title)
        fig.colorbar(im, ax=ax, fraction=.035, label="Row fraction; annotations are counts")
    fig.suptitle(f"{row['method']} | {row.get('case_id', task)} | {kind} classification")
    fig.tight_layout()
    return fig
