"""Readable paths, JSON identities, and concurrency-safe run allocation.

Digests are permitted in metadata only. Directory selection always compares the
complete declared identity; a readable name is never evidence of compatibility.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import ROOT, concept_key, digest, locked, read_json, slug, write_json


def timestamp(epoch=None):
    return datetime.fromtimestamp(epoch, ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S") if epoch is not None else datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")


def content_label(task, records):
    concepts = list(dict.fromkeys(slug(r.get("concept", "")).lower() for r in records if r.get("concept")))
    if task in {"instance", "style", "object"}:
        content = "-".join(concepts) if len(concepts) <= 3 else f"{len(concepts)}-concepts"
        return f"{content or 'concepts'}-{len(records)}-images"
    if task == "mscoco":
        return f"{len(records)}-captions"
    return f"{len(records)}-prompts"


def sampling_label(sampling):
    scheduler = sampling.get("scheduler", "unknown")
    scheduler = {"DPMSolverMultistepScheduler": "dpm", "dpm_solver_multistep": "dpm",
                 "DDIMScheduler": "ddim", "FlowMatchEulerDiscreteScheduler": "flow-euler"}.get(scheduler, slug(scheduler).lower())
    dtype = sampling.get("dtype", "auto")
    return f"{scheduler}{sampling.get('steps', 'unknown')}-{sampling.get('width', 'unknown')}x{sampling.get('height', 'unknown')}-{dtype}"


def run_label(task, records, sampling):
    return f"{task}-{content_label(task, records)}-{sampling_label(sampling)}"


def unique_directory(parent, name):
    """Reserve a new directory atomically; same-second collisions use run-2, etc."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = parent / (name if index == 1 else f"{name}-run-{index}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def registered_directory(parent, label, identity, *, started_at=None, stable=False):
    """Resume by full JSON identity, allocate readable versioned names otherwise."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    with locked(parent / ".layout.lock"):
        for child in sorted(parent.iterdir()):
            metadata = child / "run.json"
            if not child.is_dir() or not metadata.is_file():
                continue
            # A corrupt registry is a hard error: silently allocating another
            # reference set could trigger thousands of unnecessary generations.
            if read_json(metadata).get("identity") == identity:
                return child
        name = slug(label) if stable else f"{slug(label)}-{started_at or timestamp()}"
        directory = unique_directory(parent, name)
        write_json(directory / "run.json", {"schema": "benchmark-readable-run-v1", "label": label,
                   "started_at": started_at or timestamp(), "identity": identity}, portable=True)
        return directory


def method_directory(root, task, method, case_id=None):
    parent = Path(root) / task
    if case_id:
        parent = parent / case_id
    return registered_directory(parent, method, {"method": method}, stable=True)


def sampling_identity(records, model, sampling):
    # Old cache keys included these exact raw prompt records. Keep this definition
    # byte-for-byte compatible with the shared SD1.4 protocol migration.
    raw = [{k: v for k, v in r.items() if k not in {"image", "reference", "image_alias", "reference_alias", "cs_image", "cs_reference"}}
           for r in records]
    return {"records": raw, "model": model, "sampling": sampling}


def original_directory(config, records, protocol):
    identity = sampling_identity(records, config["base_model"], protocol)
    return registered_directory(Path(config.get("original_dir", ROOT / "original")) / config["task"],
                                run_label(config["task"], records, protocol), identity)


def image_directory(config, records, protocol, adapter):
    image_root = config["output_dir"]
    if config.get("image_output_dir"):
        image_root = Path(config["image_output_dir"]) / config["model_name"]
    root = method_directory(image_root, config["task"], config["method"], config.get("case_id"))
    identity = {"sampling": sampling_identity(records, config["base_model"], protocol), "adapter": adapter}
    if config.get("case_id"):
        identity["erasure"] = {"targets": sorted(concept_key(t) for t in config["targets"])}
        if "erasure_anchor" in config:
            identity["erasure"]["anchor"] = config["erasure_anchor"]
        if "erasure_parameters" in config:
            identity["erasure"]["parameters"] = config["erasure_parameters"]
    return registered_directory(root / "images", run_label(config["task"], records, protocol), identity)


def feature_directory(cache_root, manifest, rows, field, fingerprint):
    task = manifest["task"]
    owner = "original" if field == "reference" else slug(manifest["method"])
    concept = rows[0].get("concept") or task
    label = f"{slug(concept)}-{field}-{len(rows)}-images-{sampling_label(manifest['sampling'])}"
    directory = registered_directory(Path(cache_root) / "fid" / task / owner, label,
                                     {"feature_fingerprint": fingerprint})
    return directory / f"{slug(concept)}-{field}.npz"


def plot_name(task, row, group_number):
    import json
    from .erasure_protocol import target_label
    sampling = json.loads(row["sampling"])
    targets = slug(target_label(json.loads(row.get("targets", "[]")))).lower()
    case = f"{row['case_id']}-" if row.get("case_id") else ""
    return f"{task}-{case}{targets}-{sampling_label(sampling)}-n{row['n']}-group-{group_number:02d}"
