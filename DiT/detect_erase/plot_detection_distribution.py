"""Plot smoothed prompt-score percentages from a similarity_plot/records.csv file."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import ndtr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def select_threshold(target, non_target):
    """Maximize balanced accuracy on the raw scores, then center the widest optimum."""
    edges = np.unique(np.concatenate(([-1.0], target, non_target, [1.0])))
    thresholds = np.concatenate(([-1.0], (edges[:-1] + edges[1:]) / 2, [1.0]))
    widths = np.concatenate(([0.0], np.diff(edges), [0.0]))
    recall = 1 - np.searchsorted(np.sort(target), thresholds) / len(target)
    false_positive = 1 - np.searchsorted(np.sort(non_target), thresholds) / len(non_target)
    separation = recall - false_positive
    best = np.flatnonzero(np.isclose(separation, separation.max(), rtol=0, atol=1e-12))
    return float(thresholds[best[np.argmax(widths[best])]])


def smoothed_percentages(grid, scores, bandwidth, window):
    """Average the Gaussian probability mass in a window centered at each x."""
    lower = np.maximum(grid[:, None] - window / 2, -1)
    upper = np.minimum(grid[:, None] + window / 2, 1)
    mass = ndtr((upper - scores) / bandwidth) - ndtr((lower - scores) / bandwidth)
    support = ndtr((1 - scores) / bandwidth) - ndtr((-1 - scores) / bandwidth)
    return 100 * (mass / support).mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="Per-prompt scores exported by detect_erase")
    parser.add_argument("output", type=Path, help="Output path, e.g. score-distribution.pdf")
    parser.add_argument("--detector", default="Pikachu")
    parser.add_argument("--bandwidth", type=float, default=0.02, help="Gaussian standard deviation in score units")
    parser.add_argument("--threshold", type=float, help="Mark this cosine threshold on the plot")
    args = parser.parse_args()

    with args.records.open(encoding="utf-8-sig", newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if row["detector"].casefold() == args.detector.casefold() and row["similarity"] == "cosine"]
    target = np.array([float(row["score"]) for row in rows if row["group"] == "target"])
    non_target_rows = [row for row in rows if row["group"] == "unrelated"]
    non_target = np.array([float(row["score"]) for row in non_target_rows])
    scores = np.concatenate((target, non_target))
    if (not len(scores) or not np.all(np.abs(scores) <= 1)
            or not (np.isfinite(args.bandwidth) and args.bandwidth > 0)
            or (args.threshold is not None and not -1 <= args.threshold <= 1)):
        parser.error("Expected cosine scores and any threshold in [-1, 1], and a positive bandwidth.")

    use_recorded_threshold = args.threshold is None and not (len(target) and len(non_target))
    threshold = args.threshold
    if threshold is None:
        threshold = float(rows[0]["threshold"]) if use_recorded_threshold else select_threshold(target, non_target)
    threshold_name = r"\tau_{\mathrm{run}}" if use_recorded_threshold else r"\tau"
    threshold_text = str(args.threshold) if args.threshold is not None else f"{threshold:.3f}"
    window = 0.02
    padding = 3 * args.bandwidth + window / 2
    grid = np.linspace(max(-1, min(scores.min() - padding, threshold)),
                       min(1, max(scores.max() + padding, threshold)), 480)
    non_target_names = list(dict.fromkeys(row["concept"] for row in non_target_rows if row["concept"]))
    non_target_label = " + ".join(non_target_names) if 0 < len(non_target_names) <= 2 else "Non-target"
    series = (
        (non_target_label, non_target, "#A34E2E"),
        (args.detector, target, "#1D5C88"),
    )

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral"],
        "font.size": 8.5,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.labelcolor": "#253341",
        "axes.edgecolor": "#86929C",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(3.25, 1.94))
        figure.subplots_adjust(left=0.176, right=0.975, bottom=0.23, top=0.80)
        handles = []
        peak = 0
        for label, group_scores, color in series:
            if not len(group_scores):
                continue
            curve = smoothed_percentages(grid, group_scores, args.bandwidth, window)
            axis.fill_between(grid, curve, color=color, alpha=0.13, linewidth=0)
            handle, = axis.plot(grid, curve, color=color, linewidth=1.45, label=label)
            handles.append(handle)
            peak = max(peak, curve.max())

        axis.axvline(threshold, color="#344150", linewidth=0.95, linestyle=(0, (3, 2)))
        axis.text(threshold, 0.86, rf"${threshold_name}={threshold_text}$", transform=axis.get_xaxis_transform(),
                  color="#253341", ha="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
        axis.set(xlim=(grid[0], grid[-1]), ylim=(0, 1.16 * peak),
                 xlabel="Cosine similarity", ylabel="Prompt percentage")
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DCE2E7", linewidth=0.45)
        axis.xaxis.set_major_locator(MaxNLocator(6))
        axis.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        axis.tick_params(length=2.2, width=0.55, colors="#4B5966", labelsize=8)
        axis.legend(handles=handles[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.02),
                    ncol=len(handles), frameon=False, fontsize=8.5, handlelength=1.35, columnspacing=1.1)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=300)
        plt.close(figure)

    rates = [f"{name}={(group_scores >= threshold).mean():.1%}"
             for name, group_scores in (("recall", target), ("FPR", non_target)) if len(group_scores)]
    source = " (recorded)" if use_recorded_threshold else ""
    print(f"tau={threshold:.6f}{source} | target={len(target)}, non-target={len(non_target)} | {', '.join(rates)}")


if __name__ == "__main__":
    main()
