"""Strict, staged evaluation: publish a result only after all metrics succeed."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import shutil
import tempfile
from zipfile import BadZipFile

from . import __version__
from .common import configuration, concept_key, digest, slug, write_csv, write_json
from .data import load_manifest, validate_manifest
from .layout import feature_directory, method_directory, unique_directory, timestamp


def versions(task):
    names = ["torch", "numpy", "Pillow"]
    names += ["clip", "nudenet", "onnxruntime"] if task == "i2p" else ["transformers", "torch-fidelity", "torchvision"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def release(backend):
    if backend is not None:
        backend.model = None
    from .adapters import release_memory
    release_memory()


def clip_fid(manifest, opts, directory, cache_dir):
    import numpy as np
    from .metrics import CLIPScorer, FIDExtractor, cached_features, feature_fingerprint, fid_from_features

    records = [dict(r) for r in manifest["records"]]
    groups = defaultdict(list)
    for row in records:
        groups[concept_key(row["concept"]) if manifest["task"] != "mscoco" else "mscoco"].append(row)
    if any(len(rows) < 2 for rows in groups.values()):
        raise ValueError("FID requires at least two images per concept (or in the COCO set).")
    scorer = CLIPScorer(opts["clip_model"], opts["device"], opts["local_files_only"], opts["clip_long_prompt"])
    provenance = {"clip_model": opts["clip_model"], "clip_revision": scorer.revision,
                  "clip_long_prompt": opts["clip_long_prompt"], "cs_definition": "mean(100*cos); no zero clipping",
                  "fid_features": "torch-fidelity inception-v3-compat 2048", "fid_method": opts["fid_method"]}
    try:
        for field in ("image", "reference"):
            for start in range(0, len(records), opts["batch_size"]):
                batch = records[start:start + opts["batch_size"]]
                scores = scorer.score([r[field] for r in batch], [r["prompt"] for r in batch])
                if len(scores) != len(batch) or not np.isfinite(scores).all():
                    raise ValueError("CLIP did not return one finite score for every image.")
                for row, score in zip(batch, scores):
                    row[f"cs_{field}"] = float(score)
                print(f"[CLIP:{field}] {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    finally:
        release(scorer)
    extractor = None
    try:
        def extract(paths):
            nonlocal extractor
            if extractor is None:
                extractor = FIDExtractor(opts["device"], opts["local_files_only"])
            return extractor.extract([str(p) for p in paths], opts["batch_size"], opts["workers"])

        targets = {concept_key(t) for t in manifest["targets"]}
        per_concept = []
        for key, rows in groups.items():
            features = []
            for field in ("reference", "image"):
                paths = [Path(r[field]) for r in rows]
                cache = feature_directory(cache_dir, manifest, rows, field, feature_fingerprint(paths))
                # Also accepts an existing casu-2 feature file: same schema and fingerprint.
                legacy = opts.get("fid_cache_files", {}).get(key, {}).get(field)
                if legacy and not opts["refresh_cache"]:
                    legacy_path = Path(legacy)
                    if legacy_path.is_file():
                        expected = {feature_fingerprint(paths)}
                        aliases = [Path(r.get(f"{field}_alias", r[field])) for r in rows]
                        if all(alias.resolve() == path for alias, path in zip(aliases, paths)):
                            expected.add(feature_fingerprint(aliases))
                        try:
                            with np.load(legacy_path, allow_pickle=False) as saved:
                                if str(saved["fingerprint"].item()) in expected:
                                    values = saved["features"]
                                    if values.shape == (len(paths), 2048) and np.isfinite(values).all():
                                        features.append(values.copy())
                                        continue
                        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                            pass
                features.append(cached_features(paths, cache, extract, refresh=opts["refresh_cache"]))
            fid, algorithm = fid_from_features(*features, method=opts["fid_method"])
            per_concept.append({"concept": rows[0]["concept"] or key, "concept_key": key,
                                "role": "target" if key in targets else "non_target", "n": len(rows),
                                "cs": float(np.mean([r["cs_image"] for r in rows])),
                                "reference_cs": float(np.mean([r["cs_reference"] for r in rows])),
                                "fid": fid, "fid_algorithm": algorithm})
    finally:
        release(extractor)
    write_csv(directory / "per-image.csv", records)
    write_csv(directory / "per-concept.csv", per_concept)
    metrics = {"cs": float(np.mean([r["cs_image"] for r in records])),
               "reference_cs": float(np.mean([r["cs_reference"] for r in records]))}
    if manifest["task"] == "mscoco":
        metrics["fid"] = per_concept[0]["fid"]
    else:
        for group in ("target", "non_target", "all"):
            selected = [r for r in per_concept if group == "all" or r["role"] == group]
            metrics[f"fid_{group}_macro"] = float(np.mean([r["fid"] for r in selected])) if selected else None
            metrics[f"cs_{group}_macro"] = float(np.mean([r["cs"] for r in selected])) if selected else None
            metrics[f"cs_{group}"] = float(np.average([r["cs"] for r in selected], weights=[r["n"] for r in selected])) if selected else None
        for row in per_concept:
            for field in ("cs", "reference_cs", "fid", "n"):
                metrics[f"concept.{row['concept_key']}.{field}"] = row[field]
    return metrics, provenance


def i2p(manifest, opts, directory):
    from .detectors import Q16Detector, nudenet_predict

    detector = Q16Detector(opts)
    provenance = {"q16": detector.provenance}
    try:
        predictions = detector.predict(manifest["records"], opts["batch_size"])
    finally:
        release(detector)
    if len(predictions) != len(manifest["records"]):
        raise ValueError("Q16 predictions are incomplete.")
    write_csv(directory / "q16-predictions.csv", predictions)
    nude, categories, nude_provenance = nudenet_predict(manifest["records"], opts)
    write_json(directory / "nudenet-predictions.json", nude)
    write_csv(directory / "nudenet-predictions.csv", [
        {k: v for k, v in r.items() if k not in {"detections", "accepted_detections"}} for r in nude])
    write_csv(directory / "nudenet-categories.csv", [{"category": k, "instances": v} for k, v in categories.items()])
    q_unsafe = sum(r["unsafe"] for r in predictions)
    n_unsafe = sum(r["unsafe"] for r in nude)
    result = {"q16_n": len(predictions), "q16_unsafe_images": q_unsafe, "q16_unsafe_rate": q_unsafe / len(predictions),
              "nudenet_n": len(nude), "nudenet_unsafe_images": n_unsafe, "nudenet_unsafe_rate": n_unsafe / len(nude),
              "nudenet_instances": sum(categories.values())}
    result.update({f"nudenet.{k}.instances": v for k, v in categories.items()})
    provenance.update(nudenet=nude_provenance, nudenet_subset=opts["nudenet_subset"],
                      nudenet_threshold=opts["nudenet_threshold"], nudenet_classes=sorted(opts["nudenet_classes"]))
    return result, provenance


def evaluate(path, config=None, method=None, targets=None):
    manifest = load_manifest(path)
    if method is not None:
        manifest["method"] = method
    if targets is not None:
        manifest["targets"] = targets
    config = config or configuration(task=manifest["task"], base=manifest.get("config"))
    opts = config["evaluation"]
    # Validate the entire requested image set, including references, before model loads.
    validate_manifest(manifest, check_images=True)
    import torch
    if str(opts["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; set evaluation.device=cpu or --device cpu.")
    if opts["batch_size"] <= 0:
        raise ValueError("Evaluation batch size must be positive.")
    task = manifest["task"]
    root = Path(config["output_dir"])
    method_dir = method_directory(root, task, manifest["method"])
    method_dir.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    pending = unique_directory(method_dir, f".evaluation-pending-{started}")
    try:
        if task == "i2p":
            metrics, provenance = i2p(manifest, opts, pending)
        else:
            metrics, provenance = clip_fid(manifest, opts, pending, config["cache_dir"])
        import json
        data = [{k: r[k] for k in ("prompt", "concept", "categories", "seed")} for r in manifest["records"]]
        evaluator = {"version": __version__, "definitions": provenance, "packages": versions(task)}
        protocol = {"task": task, "model": manifest["base_model"], "sampling": manifest["sampling"],
                    "targets": sorted(concept_key(t) for t in manifest["targets"]),
                    "dataset_sha256": digest(data), "evaluator": evaluator}
        updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        destination = unique_directory(method_dir / "evaluations", f"evaluation-{started}")
        row = {"method": manifest["method"], "task": task, "targets": json.dumps(manifest["targets"], ensure_ascii=False),
               "base_model": manifest["base_model"], "n": len(data), "protocol_id": digest(protocol),
               "dataset_sha256": protocol["dataset_sha256"], "sampling": json.dumps(manifest["sampling"], sort_keys=True),
               "evaluator_version": __version__, "evaluation": json.dumps(evaluator, sort_keys=True),
               "adapter": json.dumps(manifest.get("adapter_identity", {}), sort_keys=True),
               "updated_at": updated, "manifest": str(Path(path).resolve()), "details_dir": str(destination), **metrics}
        write_json(pending / "result.json", {"row": row, "protocol": protocol, "evaluation_options": opts})
        # Save the exact evaluated manifest even if a future prepare replaces its input file.
        write_json(pending / "manifest.json", manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending.replace(destination)
        from .results import publish
        publish(root, task, row)
        return row
    finally:
        if pending.exists():
            shutil.rmtree(pending)
