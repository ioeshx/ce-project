"""Independent detector entry points using benchmark's unmodified detectors.

An unrequested detector is N/A, never a zero-unsafe result. Publication and
figures use benchmark's standard locked CSV/report/plot implementation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from benchmark import __version__
from benchmark.common import concept_key, digest, file_digest, write_csv, write_json
from benchmark.data import load_manifest, validate_manifest
from benchmark.detectors import Q16Detector, nudenet_predict
from benchmark.evaluate import release, versions
from benchmark.layout import method_directory, timestamp, unique_directory
from benchmark.results import publish


def detect(manifest, detector, directory):
    opts = manifest["config"]["evaluation"]
    records = manifest["records"]
    metrics = dict.fromkeys(["q16_n", "q16_unsafe_images", "q16_unsafe_rate", "nudenet_n",
                             "nudenet_unsafe_images", "nudenet_unsafe_rate", "nudenet_instances"])
    # Keep the category panels well-defined even in a Q16-only report.
    metrics.update({f"nudenet.{c}.instances": None for c in opts["nudenet_classes"]})
    if detector == "q16":
        model = Q16Detector(opts)
        try:
            predictions = model.predict(records, opts["batch_size"])
            provenance = {"q16": model.provenance}
        finally:
            release(model)
        if ([r["sample_id"] for r in predictions] != [r["sample_id"] for r in records]
                or any(r.get("unsafe") not in (True, False) for r in predictions)):
            raise ValueError("Q16 predictions are incomplete or out of order.")
        unsafe = sum(r["unsafe"] for r in predictions)
        metrics.update(q16_n=len(records), q16_unsafe_images=unsafe, q16_unsafe_rate=unsafe / len(records))
        write_csv(directory / "q16-predictions.csv", predictions)
    elif detector == "nudenet":
        predictions, counts, source = nudenet_predict(records, opts)
        selected = [r for r in records if opts["nudenet_subset"] == "all" or "sexual" in r["categories"]]
        if not selected or [r["sample_id"] for r in predictions] != [r["sample_id"] for r in selected]:
            raise ValueError("NudeNet predictions are incomplete or out of order.")
        unsafe = sum(r["unsafe"] for r in predictions)
        metrics.update(nudenet_n=len(predictions), nudenet_unsafe_images=unsafe,
                       nudenet_unsafe_rate=unsafe / len(predictions), nudenet_instances=sum(counts.values()))
        metrics.update({f"nudenet.{c}.instances": n for c, n in counts.items()})
        provenance = {"nudenet": source, "nudenet_subset": opts["nudenet_subset"],
                      "nudenet_threshold": opts["nudenet_threshold"], "nudenet_classes": sorted(opts["nudenet_classes"])}
        write_json(directory / "nudenet-predictions.json", predictions)
        write_csv(directory / "nudenet-predictions.csv", [
            {k: v for k, v in r.items() if k not in {"detections", "accepted_detections"}} for r in predictions])
        write_csv(directory / "nudenet-categories.csv", [{"category": c, "instances": n} for c, n in counts.items()])
    else:
        raise ValueError("Choose q16 or nudenet.")
    provenance.update(selected_detector=detector, integration_version="uce-independent-i2p-v1",
                      integration_sha256=file_digest(__file__))
    return metrics, provenance


def evaluate(path, detector):
    manifest = load_manifest(path)
    if manifest["task"] != "i2p":
        raise ValueError("Independent detectors require an I2P manifest.")
    validate_manifest(manifest, check_images=True)
    root = Path(manifest["config"]["output_dir"])
    owner = method_directory(root, "i2p", manifest["method"])
    pending = unique_directory(owner, f".evaluation-pending-{timestamp()}")
    try:
        metrics, provenance = detect(manifest, detector, pending)
        evaluator = {"version": __version__, "definitions": provenance, "packages": versions("i2p")}
        data = [{k: r[k] for k in ("prompt", "concept", "categories", "seed")} for r in manifest["records"]]
        protocol = {"task": "i2p", "model": manifest["base_model"], "sampling": manifest["sampling"],
                    "targets": sorted(concept_key(t) for t in manifest["targets"]),
                    "dataset_sha256": digest(data), "evaluator": evaluator}
        destination = unique_directory(owner / "evaluations", f"evaluation-{timestamp()}")
        row = {"method": manifest["method"], "task": "i2p", "targets": json.dumps(manifest["targets"]),
               "base_model": manifest["base_model"], "n": len(data), "protocol_id": digest(protocol),
               "dataset_sha256": protocol["dataset_sha256"], "sampling": json.dumps(manifest["sampling"], sort_keys=True),
               "evaluator_version": __version__, "evaluation": json.dumps(evaluator, sort_keys=True),
               "adapter": json.dumps(manifest["adapter_identity"], sort_keys=True),
               "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "manifest": str(Path(path).resolve()), "details_dir": str(destination), **metrics}
        write_json(pending / "result.json", {"row": row, "protocol": protocol,
                                           "evaluation_options": manifest["config"]["evaluation"]})
        write_json(pending / "manifest.json", manifest)
        pending.replace(destination)
        publish(root, "i2p", row)
        return row
    finally:
        if pending.exists():
            shutil.rmtree(pending)
