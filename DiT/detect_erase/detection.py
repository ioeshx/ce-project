"""Detection decisions, one-vs-rest confusion counts, and benchmark reports."""
import csv
from collections import Counter
import json
from pathlib import Path

from benchmark.common import concept_key, slug, write_csv, write_json
from benchmark.results import upsert_summary
from DiT.detect_erase.outputs import experiment_directory


def summarize_gates(manifest, gate_log, nude_keys):
    """Score actual selected prompts; repeated generation replaces the earlier decision."""
    lines = gate_log.read_text(encoding="utf-8").splitlines() if gate_log.exists() else []
    decisions = {}
    for line in lines:
        decision = json.loads(line)
        decisions[(decision["sample_id"], int(decision["seed"]))] = decision
    targets = {concept_key(target): target for target in manifest["targets"]}
    fields = ("total_count", "recorded_count", "trigger_count", "overall_correct_count", "target_count",
              "target_correct_count", "target_other_count", "target_wrong_only_count", "target_no_trigger_count",
              "non_target_count", "non_target_false_count", "non_target_no_trigger_count")
    overall = Counter(dict.fromkeys(fields, 0))
    options = manifest.get("config", {}).get("adapter", {})
    predictions = Counter(dict.fromkeys(options.get("detect_targets", manifest["targets"]), 0))
    by_concept, records = {}, []
    by_erase = {key: Counter(selected_count=0, correct_count=0, wrong_target_count=0, non_target_false_count=0, TP=0, TN=0, FP=0, FN=0)
                for key in targets}
    for record in manifest["records"]:
        key = record["sample_id"], int(record["seed"])
        if manifest["task"] == "mscoco":
            expected, label = None, "mscoco"
        elif manifest["task"] == "i2p":
            expected = "nudity" if key in nude_keys else None
            label = "nudity" if expected else "non_target"
        else:
            expected = concept_key(record["concept"]) if concept_key(record["concept"]) in targets else None
            label = record["concept"]
        if label not in by_concept:
            by_concept[label] = Counter(dict.fromkeys(fields, 0))
        counters = overall, by_concept[label]
        for counts in counters:
            counts["total_count"] += 1
        decision = decisions.get(key)
        row = dict(sample_id=key[0], seed=key[1], concept=record["concept"], prompt=record["prompt"],
                   expected_erase_prompt=targets.get(expected, ""), recorded=decision is not None)
        if decision is not None:
            predicted = decision.get("predicted", decision["selected"])
            predictions.update(predicted)
            selected = {concept_key(value) for value in decision["selected"]}
            correct = expected is not None and expected in selected
            other = bool(selected - {expected})
            for counts in counters:
                counts["recorded_count"] += 1
                counts["trigger_count"] += bool(selected)
                counts["overall_correct_count"] += selected == ({expected} if expected else set())
                if expected:
                    counts["target_count"] += 1
                    counts["target_correct_count"] += correct
                    counts["target_other_count"] += other
                    counts["target_wrong_only_count"] += bool(selected) and not correct
                    counts["target_no_trigger_count"] += not selected
                else:
                    counts["non_target_count"] += 1
                    counts["non_target_false_count"] += bool(selected)
                    counts["non_target_no_trigger_count"] += not selected
            for target_key, counts in by_erase.items():
                actual, predicted = expected == target_key, target_key in selected
                counts["TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"] += 1
            for selected_key in selected:
                counts = by_erase[selected_key]
                counts["selected_count"] += 1
                counts["correct_count"] += selected_key == expected
                counts["wrong_target_count"] += expected is not None and selected_key != expected
                counts["non_target_false_count"] += expected is None
            row.update(scores=json.dumps(decision["scores"], ensure_ascii=False),
                       predicted_detect_prompts=json.dumps(predicted, ensure_ascii=False),
                       matched_detect_prompts=json.dumps(decision["matched"], ensure_ascii=False),
                       selected_erase_prompts=json.dumps(decision["selected"], ensure_ascii=False),
                       correct_trigger=correct if expected else None, other_trigger=other if expected else None,
                       false_trigger=bool(selected) if expected is None else None, no_trigger=not selected)
        records.append(row)

    def rates(counts):
        result = dict(counts)
        result["missing_count"] = counts["total_count"] - counts["recorded_count"]
        for name, numerator, denominator in (
            ("coverage", "recorded_count", "total_count"),
            ("trigger_rate", "trigger_count", "recorded_count"),
            ("overall_accuracy", "overall_correct_count", "recorded_count"),
            ("target_correct_rate", "target_correct_count", "target_count"),
            ("target_other_rate", "target_other_count", "target_count"),
            ("target_wrong_only_rate", "target_wrong_only_count", "target_count"),
            ("target_no_trigger_rate", "target_no_trigger_count", "target_count"),
            ("non_target_false_rate", "non_target_false_count", "non_target_count"),
            ("non_target_no_trigger_rate", "non_target_no_trigger_count", "non_target_count"),
        ):
            result[name] = counts[numerator] / counts[denominator] if counts[denominator] else None
        return result

    concept_rows = [{"concept": label, **rates(counts)} for label, counts in by_concept.items()]
    expected_counts = Counter(concept_key(row["expected_erase_prompt"]) for row in records
                              if row["recorded"] and row["expected_erase_prompt"])
    erase_rows = []
    for key, counts in by_erase.items():
        row = {"erase_prompt": targets[key], **counts, "expected_count": expected_counts[key]}
        row["correct_rate"] = counts["correct_count"] / expected_counts[key] if expected_counts[key] else None
        other_targets = overall["target_count"] - expected_counts[key]
        row["wrong_target_rate"] = counts["wrong_target_count"] / other_targets if other_targets else None
        row["non_target_false_rate"] = (counts["non_target_false_count"] / overall["non_target_count"]
                                         if overall["non_target_count"] else None)
        row.update(binary_rates(counts))
        erase_rows.append(row)
    micro = {name: sum(row[name] for row in erase_rows) for name in ("TP", "TN", "FP", "FN")}
    prediction_rows = [{"detect_prompt": name, "prediction_count": count,
                        "prediction_rate": count / overall["recorded_count"] if overall["recorded_count"] else None}
                       for name, count in predictions.items()]
    return dict(summary={**rates(overall), **micro, **binary_rates(micro)},
                by_concept=concept_rows, by_erase_prompt=erase_rows, by_prediction=prediction_rows), records


def binary_rates(counts):
    tp, tn, fp, fn = (counts[name] for name in ("TP", "TN", "FP", "FN"))
    return {name: numerator / denominator if denominator else None for name, numerator, denominator in (
        ("precision", tp, tp + fp), ("recall", tp, tp + fn), ("F1", 2 * tp, 2 * tp + fp + fn),
        ("false_positive_rate", fp, fp + tn), ("false_negative_rate", fn, fn + tp),
        ("binary_accuracy", tp + tn, tp + tn + fp + fn))}


def nude_labels(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return {(row["sample_id"], int(row["seed"])) for row in csv.DictReader(source) if row["is_real_nude"] == "1"}


def plot_detection(rows, directory):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory.mkdir(exist_ok=True)
    columns = min(4, len(rows))
    fig, axes = plt.subplots(math.ceil(len(rows) / columns), columns, squeeze=False,
                             figsize=(3.6 * columns, 3.2 * math.ceil(len(rows) / columns)))
    for axis, row in zip(axes.flat, rows):
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for (y, x), value in np.ndenumerate(matrix):
            axis.text(x, y, f'{("TN", "FP", "FN", "TP")[2*y+x]}\n{value}', ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
                 yticklabels=["Negative", "Positive"], xlabel="Selected", ylabel="Ground truth",
                 title=row["erase_prompt"])
    for axis in list(axes.flat)[len(rows):]:
        axis.set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"confusion_matrix.{suffix}", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(max(6, 1.5 * len(rows)), 3.5))
    x = np.arange(len(rows))
    for index, (key, label) in enumerate((("precision", "Precision"), ("recall", "Recall"),
                                         ("false_positive_rate", "False positive rate"))):
        axis.bar(x + (index - 1) * .25, [np.nan if row[key] is None else row[key] for row in rows],
                 width=.25, label=label)
    axis.set(xticks=x, xticklabels=[row["erase_prompt"] for row in rows], ylim=(0, 1.05), ylabel="Rate")
    axis.tick_params(axis="x", labelrotation=20)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"detection_rates.{suffix}", dpi=160)
    plt.close(fig)


def publish_report(directory, manifest, gate_log, nude_keys, label_path, *, detect_only=False):
    case = manifest["case_id"]
    case_dir = experiment_directory(directory, case)
    prefix = "detect" if detect_only else "trigger"
    report, records = summarize_gates(manifest, Path(gate_log), nude_keys)
    options = manifest.get("config", {}).get("adapter", {})
    saved_report = dict(report)
    if detect_only:
        saved_report["by_detect_prompt"] = [
            {key.replace("erase_prompt", "detect_prompt"): value for key, value in row.items()}
            for row in saved_report.pop("by_erase_prompt")]
    write_json(case_dir / f"{prefix}_summary.json", {"case_id": case, "task": manifest["task"],
               "targets": manifest["targets"], "candidate_targets": options.get("detect_targets", manifest["targets"]),
               "all_detect_prompts": options.get("all_detect_prompts", False),
               "threshold_applied": not options.get("all_detect_prompts", False),
               "decision": "selected (prediction intersect case targets)",
               "confusion_denominator": "recorded prompts x target concepts; missing records excluded",
               "label_source": (str(label_path) if manifest["task"] == "i2p" else
                                "MSCOCO retention prompts (negative controls)" if manifest["task"] == "mscoco"
                                else "benchmark concept"), **saved_report})
    if detect_only:
        records = [{key.replace("erase_prompt", "detect_prompt"): value for key, value in row.items()} for row in records]
    write_csv(case_dir / f"{prefix}_records.csv", records)
    write_csv(case_dir / f"{prefix}_by_concept.csv", report["by_concept"])
    write_csv(case_dir / f"{prefix}_by_prediction.csv", report["by_prediction"])
    write_csv(case_dir / f"{prefix}_by_{'detect' if detect_only else 'erase'}_prompt.csv",
              [{key.replace("erase_prompt", "detect_prompt") if detect_only else key: value for key, value in row.items()}
               for row in report["by_erase_prompt"]])
    upsert_summary(directory, f"{prefix}_summary", {"case_id": case, **report["summary"]})
    plot_detection(report["by_erase_prompt"], case_dir / f"{prefix}_plots")
    values = {f"detection/{case}/{key}": value for key, value in report["summary"].items() if value is not None}
    for row in report["by_erase_prompt"]:
        values.update({f"detection/{case}/{slug(row['erase_prompt'])}/{key}": value for key, value in row.items()
                       if key != "erase_prompt" and value is not None})
    for row in report["by_prediction"]:
        values[f"detection/{case}/predictions/{slug(row['detect_prompt'])}/count"] = row["prediction_count"]
    print(f"[detection:{case}] {report['summary']}", flush=True)
    return values
