"""Per-detector similarity records and target-versus-unrelated distributions."""
from collections import Counter
import json
from pathlib import Path

from benchmark.common import concept_key, slug, write_csv, write_json


def read_gates(path):
    """Use the latest decision for a regenerated sample."""
    decisions = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            decision = json.loads(line)
            decisions[decision["sample_id"], int(decision["seed"])] = decision
    return decisions


def prompt_concept(manifest, record, nude_keys):
    """Use evaluation labels, independently of detector predictions/case targets."""
    if manifest["task"] == "mscoco":
        return ""
    if manifest["task"] == "i2p":
        return "nudity" if (record["sample_id"], int(record["seed"])) in nude_keys else ""
    return concept_key(record["concept"])


def detector_thresholds(options, detectors, decisions):
    """Prefer the actual logged thresholds; also support existing gate logs."""
    thresholds = {}
    for decision in decisions.values():
        thresholds.update(decision.get("thresholds", {}))
    missing = [target for target in detectors if target not in thresholds]
    if options["similarity"] == "cosine":
        override = options.get("tau")
        if override is not None:
            thresholds.update({target: override for target in missing})
        elif missing:
            import torch

            checkpoints = {item["target"]: item["checkpoint"] for item in options["checkpoints"]
                           if item["stage"] == "detect"}
            for target in missing:
                entry = torch.load(checkpoints[target], map_location="cpu", weights_only=True)
                thresholds[target] = entry["tau"]
    else:
        cutoff = None if options.get("all_detect_prompts", False) else options["l2_threshold"]
        thresholds.update({target: cutoff for target in missing})
    return {target: thresholds[target] for target in detectors}


def publish_similarity(manifest, gate_log, nude_keys, directory):
    """Save one figure per detector; each group is normalized independently."""
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    options = manifest["config"]["adapter"]
    metric = options["similarity"]
    detectors = (options.get("detect_targets", manifest["targets"]) if options.get("all_detect_prompts", False)
                 else manifest["targets"])
    decisions = read_gates(gate_log)
    thresholds = detector_thresholds(options, detectors, decisions)
    unique, sample_counts = {}, Counter()
    applied = []
    for record in manifest["records"]:
        decision = decisions.get((record["sample_id"], int(record["seed"])))
        if decision is None:
            continue
        unique[record["prompt"]] = record, decision
        sample_counts[record["prompt"]] += 1
        # Old all-detector logs predate cosine threshold gating.
        applied.append(decision.get("threshold_applied", not options.get("all_detect_prompts", False)))
    threshold_applied = all(applied) if applied else None
    rows = []
    for record, decision in unique.values():
        scores = decision["scores"]
        top1 = (max if metric == "cosine" else min)(scores, key=scores.get)
        label = prompt_concept(manifest, record, nude_keys)
        for target in detectors:
            if target not in scores:
                continue
            score, threshold = scores[target], thresholds[target]
            passed = None if threshold is None else (score >= threshold if metric == "cosine" else score <= threshold)
            rows.append(dict(case_id=manifest["case_id"], sample_id=record["sample_id"], seed=record["seed"],
                             prompt=record["prompt"], concept=record["concept"], label_concept=label,
                             sample_count=sample_counts[record["prompt"]], detector=target, similarity=metric,
                             group="target" if label == concept_key(target) else "unrelated", score=score,
                             threshold=threshold, passed_threshold=passed, top1=top1 == target,
                             predicted=target in decision.get("predicted", decision["selected"]),
                             selected=target in decision["selected"]))

    if metric == "cosine":
        edges = np.linspace(-1, 1, 101)
    else:
        maximum = max([row["score"] for row in rows] + [value for value in thresholds.values() if value is not None] + [1e-6])
        edges = np.linspace(0, maximum * 1.05, 51)
    summaries, histograms = [], []
    for target in detectors:
        for group in ("target", "unrelated"):
            members = [row for row in rows if row["detector"] == target and row["group"] == group]
            count = len(members)
            passed = sum(row["passed_threshold"] is True for row in members) if thresholds[target] is not None else None
            summaries.append(dict(detector=target, group=group, count=count, threshold=thresholds[target],
                                  threshold_count=passed,
                                  threshold_percentage=100 * passed / count if count and passed is not None else None,
                                  top1_count=sum(row["top1"] for row in members),
                                  predicted_count=sum(row["predicted"] for row in members),
                                  selected_count=sum(row["selected"] for row in members)))
            # Clip only roundoff outside cosine's mathematical range, preserving raw scores in records.csv.
            values = np.clip([row["score"] for row in members], edges[0], edges[-1])
            counts, _ = np.histogram(values, bins=edges)
            histograms.extend(dict(detector=target, group=group, bin_left=float(left), bin_right=float(right),
                                   count=int(number), percentage=100 * int(number) / count if count else 0.0)
                              for left, right, number in zip(edges[:-1], edges[1:], counts))

    report = dict(case_id=manifest["case_id"], similarity=metric, all_detect_prompts=options.get("all_detect_prompts", False),
                  threshold_applied=threshold_applied, thresholds=thresholds,
                  total_sample_count=len(manifest["records"]), recorded_sample_count=sum(sample_counts.values()),
                  missing_sample_count=len(manifest["records"]) - sum(sample_counts.values()),
                  unique_prompt_count=len(unique), deduplication="exact prompt text within this case; latest recorded sample",
                  normalization="bin count / unique prompt count in this detector's group * 100",
                  label_source="real_nude_mapping" if manifest["task"] == "i2p" else
                               "MSCOCO negative controls" if manifest["task"] == "mscoco" else "benchmark concept",
                  bin_edges=edges.tolist(), groups=summaries)
    write_csv(directory / "records.csv", rows)
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "histogram.csv", histograms)
    write_json(directory / "summary.json", report)
    plot_similarity(report, histograms, directory)
    return report


def plot_similarity(report, histograms, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    edges = report["bin_edges"]
    peak = max((row["percentage"] for row in histograms), default=0)
    ylim = (0, max(5, min(105, peak * 1.25)))
    for target, threshold in report["thresholds"].items():
        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        notes = []
        for group, color in (("target", "#2563eb"), ("unrelated", "#e07824")):
            summary = next(row for row in report["groups"] if row["detector"] == target and row["group"] == group)
            distribution = [row["percentage"] for row in histograms if row["detector"] == target and row["group"] == group]
            label = f"{group.capitalize()} (n={summary['count']})"
            if threshold is not None:
                relation = ">=" if report["similarity"] == "cosine" else "<="
                percentage = summary["threshold_percentage"]
                rate = f"{percentage:.1f}%" if percentage is not None else "N/A"
                label += f"; {relation} threshold: {summary['threshold_count']}/{summary['count']} ({rate})"
            axis.stairs(distribution, edges, label=label, color=color, linewidth=1.8)
            notes.append(f"{group.capitalize()}: top1={summary['top1_count']}, selected={summary['selected_count']}")
        if threshold is not None:
            label = (rf"$\tau$ = {threshold:.4g}" if report["similarity"] == "cosine" else f"Threshold = {threshold:.4g}")
            if report["threshold_applied"] is False:
                label += " (reference; not applied in saved decisions)"
            axis.axvline(threshold, color="#333333", linestyle="--", linewidth=1.4, label=label)
        axis.set(xlabel="Cosine similarity" if report["similarity"] == "cosine" else "RMS distance",
                 ylabel="Prompts in bin (%)", xlim=(edges[0], edges[-1]), ylim=ylim,
                 title=f"{report['case_id']} / {target}")
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=9, loc="upper left")
        fig.text(.12, .025, " | ".join(notes) + "\nOne vote per unique prompt; each nonempty group sums to 100%.", fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, 1))
        for suffix in ("png", "pdf"):
            fig.savefig(directory / f"{slug(concept_key(target))}.{suffix}", dpi=160)
        plt.close(fig)
