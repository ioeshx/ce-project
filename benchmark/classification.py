"""Staged classifier inference, reusable prediction caches and auditable summaries."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import concept_key, read_json, write_csv, write_json
from .layout import registered_directory
from .paths import relative_path


def resource_settings(opts, kind, task):
    import importlib.metadata
    result = {k: v for k, v in opts.items() if k.startswith(kind + "_")}
    result.update(task=task, device=opts.get(kind + "_device") or opts["device"],
                  package=importlib.metadata.version("transformers" if kind == "vlm" else "torchvision"))
    if kind == "vlm":
        from .vlm import LABELS, PROMPT_VERSION
        result.update(labels=LABELS[task], instruction_version=PROMPT_VERSION)
        source = Path(opts["vlm_model"])
        if not source.is_dir():
            if source.is_absolute() or opts["vlm_model"].startswith(("./", "../")):
                raise FileNotFoundError("Qwen3-VL local weights missing; run python -m benchmark download --resource vlm.")
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(opts["vlm_model"], "config.json", revision=opts.get("vlm_revision"))
            source = Path(cached).parent if isinstance(cached, str) else None
    else:
        from .object_classifier import weight_path
        source = weight_path(opts)
    if source and source.exists():
        files = sorted(p for p in source.rglob("*") if p.is_file() and ".cache" not in p.relative_to(source).parts) if source.is_dir() else [source]
        result["resource_files"] = [[str(p.relative_to(source)) if source.is_dir() else p.name,
                                      p.stat().st_size, p.stat().st_mtime_ns] for p in files]
    return result


def predict_field(manifest, opts, kind, field, cache_root, backend):
    paths = [Path(r[field]) for r in manifest["records"]]
    settings = resource_settings(opts, kind, manifest["task"])
    images = [[relative_path(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in paths]
    identity = {"settings": settings, "images": images}
    directory = registered_directory(Path(cache_root) / "predictions" / kind / manifest.get("model_name", "custom") / manifest["task"],
                                     f"{kind}-{field}-{len(paths)}-images", identity)
    cached = directory / "predictions.json"
    if cached.is_file() and not opts.get("refresh_cache"):
        saved = read_json(cached)
        if len(saved["predictions"]) != len(paths):
            raise ValueError("Classifier prediction cache is incomplete.")
        print(f"[{kind} cache:{field}] {len(paths)} images", flush=True)
        return saved["predictions"], saved["provenance"]
    model = backend()
    batch_size = opts.get("vlm_batch_size", 1) if kind == "vlm" else opts["batch_size"]
    predictions = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        output = model.predict(batch)
        if len(output) != len(batch):
            raise ValueError(f"{kind} must return exactly one prediction per image.")
        predictions.extend(output)
        print(f"[{kind}:{field}] {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    # Loading a resource may have populated its on-disk cache; register that
    # resolved identity so subsequent cases can reuse the original predictions.
    identity["settings"] = resource_settings(opts, kind, manifest["task"])
    directory = registered_directory(directory.parent, f"{kind}-{field}-{len(paths)}-images", identity)
    write_json(directory / "predictions.json", {"schema": "benchmark-classifier-predictions-v1",
               "predictions": predictions, "provenance": model.provenance}, portable=True)
    return predictions, model.provenance


def summarize(records, targets, kind):
    """Classification ground truth is the requested concept, never the erasure label."""
    targets = {concept_key(t) for t in targets}
    metrics = {}
    concepts = list(dict.fromkeys(concept_key(r["concept"]) for r in records))
    for field, prefix in (("image", ""), ("reference", "reference_")):
        names = ("accuracy",) if kind == "vlm" else ("top1", "top3")
        for name in names:
            def correct(row):
                prediction = row[field + "_prediction"]
                if kind == "vlm":
                    return concept_key(prediction["label"]) == concept_key(row["concept"])
                return row["imagenet_id"] in (prediction["top3_indices"][:1] if name == "top1" else prediction["top3_indices"])
            for role in ("all", "target", "non_target"):
                selected = [r for r in records if role == "all" or (concept_key(r["concept"]) in targets) == (role == "target")]
                key = f"{prefix}{kind}_{name}" if role == "all" else f"{prefix}{kind}_{role}_{name}"
                metrics[key] = sum(correct(r) for r in selected) / len(selected) if selected else None
                metrics[f"{prefix}{kind}_{role}_n"] = len(selected)
                metrics[key + "_correct"] = sum(correct(r) for r in selected)
            for concept in concepts:
                selected = [r for r in records if concept_key(r["concept"]) == concept]
                metrics[f"concept.{concept}.{prefix}{kind}_{name}"] = sum(correct(r) for r in selected) / len(selected)
                metrics[f"concept.{concept}.n"] = len(selected)
        if kind == "vlm":
            from .vlm import LABELS
            metrics[f"{prefix}vlm_none_rate"] = sum(r[field + "_prediction"]["label"] == "NONE" for r in records) / len(records)
            # Use the task's full label vocabulary, even in balanced smoke subsets.
            from .vlm import LABELS
            vocabulary = LABELS["instance"] if all(c in {concept_key(s) for s in LABELS["instance"]} for c in concepts) else LABELS["style"]
            for concept in concepts:
                counts = Counter(r[field + "_prediction"]["label"] for r in records if concept_key(r["concept"]) == concept)
                for label in vocabulary:
                    metrics[f"{prefix}vlm.confusion.{concept}.{concept_key(label)}.count"] = counts[label]
        else:
            for concept in concepts:
                selected = [r for r in records if concept_key(r["concept"]) == concept]
                counts = Counter(r[field + "_prediction"]["prediction_index"] for r in selected)
                for index, count in counts.items():
                    metrics[f"{prefix}resnet.confusion.{concept}.{index}.count"] = count
    return metrics


def evaluate_classifier(manifest, opts, directory, cache_root, kind):
    model = None
    def backend():
        nonlocal model
        if model is None:
            if kind == "vlm":
                from .vlm import VLMClassifier
                model = VLMClassifier(opts, manifest["task"])
            else:
                from .object_classifier import ResNetClassifier
                model = ResNetClassifier(opts)
        return model
    records = [dict(r) for r in manifest["records"]]
    provenance = {}
    try:
        for field in ("reference", "image"):
            predictions, evidence = predict_field(manifest, opts, kind, field, cache_root, backend)
            provenance[field] = evidence
            for row, prediction in zip(records, predictions):
                row[field + "_prediction"] = prediction
    finally:
        if model is not None:
            from .evaluate import release
            release(model)
    write_csv(directory / f"{kind}-predictions.csv", records, portable=True)
    return summarize(records, manifest["targets"], kind), provenance
