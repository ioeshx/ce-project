"""One latest successful row per method, atomic updates and grouped reports."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from .common import atomic_text, locked, read_result_csv, write_csv


def read_rows(root, task):
    path = Path(root) / f"{task}-metric.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No results yet: {path}")
    return read_result_csv(path)


def result_key(row):
    return (row["method"], row.get("case_id") or "")


def upsert(root, task, row):
    if row.get("task") != task or not row.get("method"):
        raise ValueError("Result row must identify the task and method.")
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite result: {key}")
    path = Path(root) / f"{task}-metric.csv"
    with locked(path.with_suffix(".lock")):
        rows = read_rows(root, task) if path.exists() else []
        rows = [r for r in rows if result_key(r) != result_key(row)] + [row]
        rows.sort(key=lambda r: (r.get("case_id") or "", r["method"].casefold()))
        write_csv(path, rows, portable=True)


def groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["protocol_id"]].append(row)
    return sorted(grouped.items())


def fmt(value):
    if value is None or value == "":
        return "N/A"
    try:
        n = float(value)
        return f"{n:.4f}" if not n.is_integer() else str(int(n))
    except (ValueError, TypeError):
        return str(value)


def protocol_label(row):
    from .erasure_protocol import target_label
    sampling = json.loads(row["sampling"])
    targets = json.loads(row["targets"])
    label = target_label(targets) if len(targets) > 3 else row["targets"]
    case = f"case={row['case_id']} | " if row.get("case_id") else ""
    return (f"{case}targets={label} | n={row['n']} | {sampling.get('scheduler')} "
            f"{sampling.get('steps')} steps | CFG {sampling.get('guidance_scale')} | "
            f"{sampling.get('width')}x{sampling.get('height')} | {row['base_model']}")


def report(root, task):
    rows = read_rows(root, task)
    fields = {"instance": ["cs_target", "cs_non_target", "fid_target_macro", "fid_non_target_macro"],
              "style": ["cs_target", "cs_non_target", "fid_target_macro", "fid_non_target_macro"],
              "i2p": ["q16_n", "q16_unsafe_images", "q16_unsafe_rate", "nudenet_n", "nudenet_unsafe_images",
                      "nudenet_unsafe_rate", "nudenet_instances"], "mscoco": ["cs", "reference_cs", "fid"],
              "object": ["resnet_top1", "resnet_top3", "resnet_target_top1", "resnet_target_top3", "resnet_non_target_top1", "resnet_non_target_top3",
                         "reference_resnet_top1", "reference_resnet_top3"]}[task]
    texts = []
    for group_number, (key, selected) in enumerate(groups(rows), 1):
        selected_fields = list(fields)
        if task == "i2p" and not any(r.get("q16_n") not in (None, "") for r in selected):
            selected_fields = [k for k in selected_fields if not k.startswith("q16_")]
        if any(r.get("vlm_accuracy") not in (None, "") for r in selected):
            selected_fields += ["vlm_target_accuracy", "vlm_non_target_accuracy", "reference_vlm_accuracy", "vlm_none_rate"]
        columns = ["method"] + selected_fields
        values = [[fmt(r.get(k)) for k in columns] for r in selected]
        widths = [max(len(k), *(len(row[i]) for row in values)) for i, k in enumerate(columns)]
        def line(row):
            return " | ".join(value.ljust(width) for value, width in zip(row, widths))
        texts.extend([f"{task} | group {group_number}", protocol_label(selected[0]), line(columns),
                      "-+-".join("-" * w for w in widths), *(line(row) for row in values), ""])
        if task in {"instance", "style"}:
            detail_columns = ["method", "concept", "role", "CS", "original CS", "FID", "n"]
            detail = []
            for row in selected:
                targets = {c.casefold() for c in json.loads(row["targets"])}
                concepts = sorted(k[8:-3] for k, v in row.items() if k.startswith("concept.") and k.endswith(".cs") and v != "")
                from .common import concept_key
                targets = {concept_key(c) for c in targets}
                for concept in concepts:
                    detail.append([row["method"], concept, "target" if concept in targets else "non_target",
                                   *(fmt(row.get(f"concept.{concept}.{k}")) for k in ("cs", "reference_cs", "fid", "n"))])
            widths = [max(len(k), *(len(r[i]) for r in detail)) for i, k in enumerate(detail_columns)]
            texts.extend([line(detail_columns), "-+-".join("-" * w for w in widths), *(line(r) for r in detail), ""])
        if task == "object" or any(r.get("vlm_accuracy") not in (None, "") for r in selected):
            kind = "resnet" if task == "object" else "vlm"
            score_keys = ["resnet_top1", "resnet_top3", "reference_resnet_top1", "reference_resnet_top3"] if task == "object" else ["vlm_accuracy", "reference_vlm_accuracy"]
            detail_columns = ["method", "concept", *score_keys, "n"]
            detail = []
            for row in selected:
                concepts = sorted(k[8:-2] for k in row if k.startswith("concept.") and k.endswith(".n") and row[k] != "")
                for concept in concepts:
                    detail.append([row["method"], concept, *(fmt(row.get(f"concept.{concept}.{key}")) for key in (*score_keys, "n"))])
            widths = [max(len(k), *(len(r[i]) for r in detail)) for i, k in enumerate(detail_columns)]
            texts.extend([line(detail_columns), "-+-".join("-" * w for w in widths), *(line(r) for r in detail), ""])
    text = "\n".join(texts)
    atomic_text(Path(root) / f"{task}-metric.txt", text + "\n")
    print(text)
    return text


def publish(root, task, row):
    from .model_profiles import model_name
    row.setdefault("model_name", model_name(row.get("base_model")))
    upsert(root, task, row)
    # Concurrent evaluators render from the newest complete CSV snapshot.
    with locked(Path(root) / f"{task}-presentation.lock"):
        report(root, task)
        from .plotting import plot
        plot(root, task)
