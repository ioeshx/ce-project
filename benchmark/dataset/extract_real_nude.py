"""Export the 125 SD3.5 baseline positives and their mapping to all 931 I2P rows.

Uses saved NudeNet predictions only; no generation or detector inference.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / (
    "results/sd35_medium/i2p/i2p_nudity/original/evaluations/"
    "evaluation-20261026-230536"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(evaluation_dir, output_dir):
    manifest_path = evaluation_dir / "manifest.json"
    predictions_path = evaluation_dir / "nudenet-predictions.json"
    result_path = evaluation_dir / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (manifest["model_name"], manifest["method"], manifest["task"]) != (
        "sd35_medium", "original", "i2p"
    ):
        raise ValueError("Expected the original SD3.5 Medium I2P evaluation.")

    source_path = ROOT / manifest["dataset"]["path"]
    sexual_path = ROOT / "dataset/i2p_sexual.csv"
    fields, source_rows = read_csv(source_path)
    _, sexual_rows = read_csv(sexual_path)
    indexed = [(i, row) for i, row in enumerate(source_rows)
               if "sexual" in {c.strip() for c in row["categories"].split(",")}]
    if sha256(source_path) != manifest["dataset"]["sha256"]:
        raise ValueError("The source I2P CSV differs from the evaluated dataset.")
    if [row for _, row in indexed] != sexual_rows:
        raise ValueError("i2p_sexual.csv differs from the evaluated sexual subset.")
    records = manifest["records"]
    by_id = {prediction["sample_id"]: prediction for prediction in predictions}
    if not (len(indexed) == len(records) == len(predictions) == len(by_id) == 931):
        raise ValueError("Expected exactly 931 source rows, records and predictions.")
    if len({r["sample_id"] for r in records}) != 931 or set(by_id) != {
        r["sample_id"] for r in records
    }:
        raise ValueError("Manifest and prediction sample IDs do not match uniquely.")

    definitions = result["protocol"]["evaluator"]["definitions"]
    classes = set(definitions["nudenet_classes"])
    threshold = definitions["nudenet_threshold"]
    selected, mapping, class_counts = [], [], Counter()
    for sexual_index, ((source_index, row), record) in enumerate(zip(indexed, records)):
        if (record["source_index"] != source_index
                or record["sample_id"] != f"i2p-{source_index:06d}-000"
                or record["prompt"] != row["prompt"]
                or record["seed"] != int(row["sd_seed"])):
            raise ValueError(f"Source prompt/seed/index mismatch: {record['sample_id']}")
        prediction = by_id[record["sample_id"]]
        accepted = [d for d in prediction["detections"]
                    if d["class"] in classes and d["score"] >= threshold]
        if (prediction["image"] != record["image"]
                or accepted != prediction["accepted_detections"]
                or bool(accepted) != prediction["unsafe"]
                or len(accepted) != prediction["accepted_instances"]):
            raise ValueError(f"Inconsistent saved detection: {record['sample_id']}")
        identity = {
            "sample_id": record["sample_id"], "source_index": source_index,
            "source_id": record["source_id"], "sexual_index": sexual_index,
            "seed": record["seed"],
        }
        mapping.append({**identity, "is_real_nude": int(bool(accepted)),
                        "real_nude_index": len(selected) if accepted else ""})
        if accepted:
            class_counts.update(d["class"] for d in accepted)
            selected.append({
                **row, **identity, "image": record["image"],
                "reference": record["reference"], "nudenet_unsafe": 1,
                "nudenet_accepted_instances": len(accepted),
                "nudenet_max_selected_confidence": prediction["max_selected_confidence"],
                "nudenet_classes": ",".join(sorted({d["class"] for d in accepted})),
                "nudenet_accepted_detections": json.dumps(accepted, ensure_ascii=False),
            })

    if not (len(selected) == result["row"]["nudenet_unsafe_images"] == 125
            and result["row"]["nudenet_n"] == 931
            and sum(class_counts.values()) == result["row"]["nudenet_instances"] == 219):
        raise ValueError("Saved results do not match the expected 125/931 images and 219 boxes.")

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "real_nude.csv"
    mapping_path = output_dir / "real_nude_mapping.csv"
    write_csv(subset_path, fields + [k for k in selected[0] if k not in fields], selected)
    write_csv(mapping_path, list(mapping[0]), mapping)

    def artifact(path):
        return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    metadata = {
        "description": "125 baseline SD3.5 Medium I2P images marked unsafe by the saved NudeNet evaluation.",
        "path_base": "benchmark",
        "counts": {"i2p_full": len(source_rows), "i2p_sexual": len(mapping),
                   "selected_rows": len(selected),
                   "unique_selected_prompt_texts": len({r["prompt"] for r in selected}),
                   "accepted_detections": sum(class_counts.values())},
        "accepted_instances_by_class": {c: class_counts[c] for c in sorted(classes)},
        "indexing": {
            "source_index": "Zero-based data row in dataset/i2p_benchmark.csv (4703 rows).",
            "source_id": "Original manifest source_id; the original full I2P row index as a string.",
            "sexual_index": "Zero-based data row in dataset/i2p_sexual.csv and the original 931-record manifest.",
            "real_nude_index": "Zero-based data row in real_nude.csv; empty for unselected mapping rows.",
            "sample_id": "Stable baseline ID retained from the original 931-record manifest.",
            "csv_rows": "Indices count parsed CSV data records, excluding the header; embedded newlines do not start new records.",
        },
        "selection": {"field": "unsafe", "value": True,
                      "rule": "At least one detection in the saved selected classes with score >= threshold.",
                      "deduplicate_prompt_text": False},
        "sources": {"i2p_benchmark": artifact(source_path),
                    "i2p_sexual": artifact(sexual_path),
                    "manifest": artifact(manifest_path),
                    "predictions": artifact(predictions_path),
                    "result": artifact(result_path)},
        "outputs": {"real_nude.csv": {"sha256": sha256(subset_path)},
                    "real_nude_mapping.csv": {"sha256": sha256(mapping_path)}},
        "protocol_id": result["row"]["protocol_id"],
        "protocol": result["protocol"],
    }
    (output_dir / "real_nude.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(selected)}/931 rows and {sum(class_counts.values())} accepted detections to {subset_path}")
    print(f"Full 931-row mapping: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=EVALUATION)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    export(args.evaluation_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
