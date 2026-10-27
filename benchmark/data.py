"""Prompt expansion, explicit image manifests, and lossless CASU-2 import."""
from __future__ import annotations

import csv
from pathlib import Path

from .common import (ROOT, SCHEMA, TASKS, concept_key, digest, file_digest, read_json, slug,
                     write_json)

FILES = {"instance": "ip_imagenet_prompts.csv", "style": "style_painting_prompts.csv",
         "i2p": "i2p_benchmark.csv", "mscoco": "mscoco.csv", "object": "imagenet10.csv"}


def prompt_records(config):
    task, opts = config["task"], config["dataset"]
    path = Path(opts.get("path") or ROOT / "dataset" / FILES[task]).expanduser().resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if task in {"instance", "style"} and opts.get("concepts"):
        from .dataset.prompt_templates import imagenet_templates, painting_templates
        templates = imagenet_templates if task == "instance" else painting_templates
        rows = [{"id": f"custom-{c}-{i}", "concept": c, "prompt": t.format(c), "template_index": i}
                for c in opts["concepts"] for i, t in enumerate(templates)]
    indexed = list(enumerate(rows))
    if task == "i2p" and opts.get("i2p_subset") == "sexual":
        indexed = [(index, row) for index, row in indexed
                   if "sexual" in [s.strip() for s in row["categories"].split(",")]]
    if opts.get("limit_prompts") is not None:
        if opts["limit_prompts"] <= 0:
            raise ValueError("limit_prompts must be positive.")
        indexed = indexed[:opts["limit_prompts"]]
    records = []
    for index, row in indexed:
        prompt = row.get("prompt", row.get("text"))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing prompt in dataset row {index}.")
        seeds = opts["seeds"] if task in {"instance", "style", "object"} else [int(row["sd_seed"]) if task == "i2p" else index]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("Seeds must be nonempty and unique.")
        for sample_index, seed in enumerate(seeds):
            if not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
                raise ValueError("Seeds must be nonnegative integers below 2**63.")
            record = {"sample_id": f"{task}-{index:06d}-{sample_index:03d}", "source_index": index,
                      "source_id": row.get("id", row.get("image_id", str(index))), "prompt": prompt,
                      "concept": row.get("concept", ""),
                      "categories": [x.strip() for x in row.get("categories", "").split(",") if x.strip()],
                      "seed": seed}
            if opts.get("stable_ids") and task in {"instance", "style", "object"}:
                template_index = row.get("template_index", 0 if task == "object" else index)
                record["sample_id"] = f"{task}-{slug(concept_key(record['concept']))}-template-{template_index:03d}-seed-{seed}"
                record["template_index"] = template_index
            if task == "object":
                record["imagenet_id"] = int(row["imagenet_id"])
            if task == "i2p":
                record["source_sampling"] = {k: row[k] for k in
                    ("sd_seed", "sd_guidance_scale", "sd_image_width", "sd_image_height", "sd_model")}
            records.append(record)
    if opts.get("max_records") is not None:
        if opts["max_records"] <= 0:
            raise ValueError("max_records must be positive.")
        if config.get("case_id") and task in {"instance", "style", "object"}:
            groups = {}
            for row in records:
                groups.setdefault(concept_key(row["concept"]), []).append(row)
            if opts["max_records"] < len(groups):
                raise ValueError("Case smoke samples must include every evaluation concept; increase max_records.")
            records = [rows[i] for i in range(max(map(len, groups.values())))
                       for rows in groups.values() if i < len(rows)][:opts["max_records"]]
        else:
            records = records[:opts["max_records"]]
    if not records:
        raise ValueError("Empty dataset.")
    return records, {"path": str(path), "sha256": file_digest(path), "options": opts}


def sampling_protocol(config):
    from .sampling_profile import canonical_protocol
    return canonical_protocol(config["sampling"])


def adapter_identity(config):
    spec = dict(config["adapter"])
    if spec["kind"] == "casu2":
        directory = Path(spec["checkpoint"]).expanduser()
        if directory.is_file():
            directory = directory.parent
        metadata = read_json(directory / "metadata.json")
        if (metadata.get("task") != config["task"]
                or {concept_key(t) for t in metadata.get("targets", [])} != {concept_key(t) for t in config["targets"]}):
            raise ValueError("CASU task/targets differ from the experiment configuration.")
    if spec["kind"] == "plugin":
        import importlib.util
        module = spec["factory"].split(":", 1)[0]
        found = importlib.util.find_spec(module)
        if found is None:
            raise ImportError(f"Cannot find adapter module: {module}")
        if found.origin and Path(found.origin).is_file():
            spec["plugin_source_sha256"] = file_digest(found.origin)
            if spec.get("source_files"):
                spec["source_sha256"] = {name: file_digest(Path(found.origin).parent / name)
                                         for name in spec["source_files"]}
    path = spec.get("checkpoint") or spec.get("model_path")
    if path and Path(path).exists():
        p = Path(path).resolve()
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        spec["files"] = [[str(x.relative_to(p) if p.is_dir() else x.name), file_digest(x)] for x in files]
    return spec


def prepare(config, output=None):
    from .common import configuration
    from .model_profiles import DIT_MODELS, resolve_scheduler
    config = configuration(task=config["task"], base=config)
    from .dit_cases import validate_anchor
    validate_anchor(config)
    resolve_scheduler(config)
    # Resolve auto before caching: CPU float32 images must not masquerade as GPU float16 images.
    if config["sampling"]["dtype"] == "auto":
        config["sampling"]["dtype"] = (("bfloat16" if config["model_name"] in DIT_MODELS else "float16")
                                        if str(config["sampling"]["device"]).startswith("cuda") else "float32")
    if config["adapter"]["kind"] == "existing":
        raise ValueError("For existing images, supply a native manifest or use --import-casu.")
    records, source = prompt_records(config)
    protocol = sampling_protocol(config)
    identity = adapter_identity(config)
    from .layout import method_directory, original_directory, image_directory
    root = method_directory(config["output_dir"], config["task"], config["method"], config.get("case_id"))
    original_dir = original_directory(config, records, protocol)
    images = image_directory(config, records, protocol, identity)
    for record in records:
        name = record["sample_id"] + ".png"
        record["reference"] = str(original_dir / name)
        record["image"] = str(images / name)
        if identity["kind"] == "original":
            alias = images / name
            if alias.is_symlink() or not alias.exists():
                from .paths import relative_symlink
                relative_symlink(alias, original_dir / name)
    manifest = {"schema": SCHEMA, "task": config["task"], "targets": config["targets"], "model_name": config["model_name"],
                "method": config["method"], "base_model": config["base_model"], "sampling": protocol,
                "dataset": source, "adapter_identity": identity, "config": config, "records": records}
    if config.get("case_id"):
        manifest["case_id"] = config["case_id"]
    if config.get("reuse_manifest"):
        reuse_images(manifest, config["reuse_manifest"])
    validate_manifest(manifest, check_images=False)
    output = Path(output) if output else root / "manifest.json"
    write_json(output, manifest, portable=True)
    return output.resolve()


def validate_manifest(manifest, check_images=True):
    if manifest.get("schema") != SCHEMA or manifest.get("task") not in TASKS:
        raise ValueError("Unsupported manifest; use prepare --import-casu to import a CASU directory.")
    if not manifest.get("base_model") or not isinstance(manifest.get("sampling"), dict) or not manifest.get("method"):
        raise ValueError("Manifest needs base_model, sampling protocol and method.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a nonempty records list.")
    seen = set()
    paths = {"image": set(), "reference": set()}
    for i, r in enumerate(records):
        if not all(k in r for k in ("sample_id", "prompt", "concept", "categories", "seed", "image")):
            raise ValueError(f"Incomplete manifest row {i}.")
        if not r["prompt"].strip() or not isinstance(r["categories"], list) or r["sample_id"] in seen:
            raise ValueError(f"Invalid prompt, categories or duplicate sample ID at row {i}.")
        seen.add(r["sample_id"])
        if not isinstance(r["seed"], int) or not 0 <= r["seed"] < 2 ** 63:
            raise ValueError(f"Invalid seed at row {i}.")
        for mode in (["image"] if manifest["task"] == "i2p" else ["image", "reference"]):
            if not r.get(mode):
                raise ValueError(f"Missing {mode} path at row {i}.")
            p = Path(r[mode])
            if not p.is_absolute() or p in paths[mode]:
                raise ValueError(f"Use distinct absolute {mode} paths: {p}")
            paths[mode].add(p)
            if check_images and not valid_image(p):
                raise ValueError(f"Missing or corrupt {mode} image for {r['sample_id']}: {p}")
            if check_images:
                from PIL import Image
                with Image.open(p) as im:
                    if im.size != (manifest["sampling"]["width"], manifest["sampling"]["height"]):
                        raise ValueError(f"Image dimensions disagree with the recorded sampling protocol: {p}")
    if manifest["task"] in {"instance", "style", "object"}:
        concepts = {concept_key(r["concept"]) for r in records}
        targets = {concept_key(c) for c in manifest["targets"]}
        if not targets or not targets <= concepts:
            raise ValueError(f"Targets must occur in evaluation concepts: {sorted(targets - concepts)}")
    if manifest["task"] == "object":
        from .object_classifier import IMAGENET10
        for row in records:
            if row.get("imagenet_id") != IMAGENET10.get(concept_key(row["concept"])):
                raise ValueError(f"Invalid ImageNet-1K label for {row['concept']}.")


def reuse_images(manifest, path):
    """Link matching existing samples; never modify the source manifest/pixels."""
    from .paths import relative_symlink
    old = load_manifest(path)
    if old["task"] != manifest["task"] or old["base_model"] != manifest["base_model"] or old["sampling"] != manifest["sampling"]:
        raise ValueError("Reuse manifest must match task, base model and the complete sampling protocol.")
    keyed = {r["sample_id"]: r for r in old["records"]}
    semantic = {}
    for row in old["records"]:
        semantic.setdefault((row["prompt"], row["seed"], concept_key(row["concept"])), []).append(row)
    same_method = (old.get("adapter_identity") == manifest.get("adapter_identity")
                   and (old.get("adapter_identity", {}).get("kind") == "original"
                        or {concept_key(t) for t in old["targets"]} == {concept_key(t) for t in manifest["targets"]}))
    linked = 0
    for row in manifest["records"]:
        source = keyed.get(row["sample_id"])
        if source is None and manifest["task"] != "i2p":
            candidates = semantic.get((row["prompt"], row["seed"], concept_key(row["concept"])), [])
            source = candidates[0] if len(candidates) == 1 else None
        if not source or any(source[k] != row[k] for k in ("prompt", "seed", "concept", "categories")):
            continue
        for field in ("reference", "image"):
            if field == "image" and not same_method:
                continue
            target = Path(source.get(field, ""))
            if field == "reference" and not valid_image(target) and old.get("adapter_identity", {}).get("kind") == "original":
                target = Path(source["image"])
            link = Path(row[field])
            if target != link and valid_image(target) and not link.exists():
                relative_symlink(link, target)
                linked += 1
    print(f"[reuse] {linked} matching image paths linked", flush=True)


def valid_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def load_manifest(path):
    import os
    path = Path(path).expanduser().resolve()
    manifest = read_json(path)
    # Native paths are explicit; accepting relative paths is useful for portable imports.
    for row in manifest.get("records", []):
        for field in ("image", "reference"):
            if row.get(field):
                # Keep readable result aliases inside results/<model>; following
                # a symlink here would silently replace them with cache paths.
                row[field] = os.path.abspath(path.parent / row[field])
                current = Path(row[field])
                if current.is_symlink() and current.readlink().is_absolute():
                    from .paths import artifact_root, legacy_path, relative_symlink
                    target = legacy_path(str(current.readlink()), artifact_root(path))
                    relative_symlink(current, target)
    validate_manifest(manifest, check_images=False)
    return manifest


def import_casu(directory, config, output=None):
    directory = Path(directory).expanduser().resolve()
    if directory.is_file():
        directory = directory.parent
    rows = read_json(directory / "manifest.json")
    plan = read_json(directory / "dataset.json")
    if not isinstance(rows, list) or len(rows) != plan["record_count"]:
        raise ValueError("CASU manifest is incomplete.")
    source = [{k: v for k, v in row.items() if k not in {"original", "edit", "compare"}} for row in rows]
    # CASU hashes json.dumps with default escaping/separators, not our canonical digest.
    import hashlib
    import json
    if hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() != plan["records_sha256"]:
        raise ValueError("CASU manifest does not match dataset.json.")
    if plan["task"] not in {"instance", "style"}:
        raise ValueError("This importer supports CASU instance/style manifests.")
    config = dict(config, task=plan["task"], targets=plan["targets"], base_model=plan["sampling"]["sd_ckpt"])
    old = plan["sampling"]
    sampling = {k: old[k] for k in ("steps", "guidance_scale", "negative_prompt", "height", "width", "dtype")}
    # casu-2/models.py uses this scheduler for SD1. Preserve actual legacy protocol.
    sampling.update(scheduler="dpm_solver_multistep", architecture=old["architecture"], provenance="casu-2/dataset.json")
    if old["architecture"] != "sd1":
        raise ValueError("First release supports SD1 CASU imports only; DiT/SDXL are out of scope.")
    records = []
    for i, row in enumerate(rows):
        records.append({"sample_id": f"{plan['task']}-casu-{i:06d}", "source_index": i,
                        "prompt": row["prompt"], "concept": row["concept"], "categories": [],
                        "seed": row["seed"], "image": str((directory / row["edit"]).resolve()),
                        "reference": str((directory / row["original"]).resolve()),
                        "image_alias": str(directory / row["edit"]),
                        "reference_alias": str(directory / row["original"])})
    manifest = {"schema": SCHEMA, "task": plan["task"], "method": config["method"], "targets": plan["targets"],
                "base_model": config["base_model"], "sampling": sampling, "records": records,
                "dataset": {"imported_from": str(directory), "plan": plan},
                "adapter_identity": {"kind": "existing", "checkpoint_sha256": plan.get("checkpoint_weights_sha256")},
                "config": config}
    validate_manifest(manifest, check_images=False)
    from .layout import method_directory
    output = Path(output) if output else method_directory(config["output_dir"], plan["task"], config["method"]) / "manifest.json"
    write_json(output, manifest, portable=True)
    return output.resolve()
