"""Convert active benchmark artifacts and image links to relative paths.

Preview: python -m benchmark.migrate_paths
Apply:   python -m benchmark.migrate_paths --apply
Image pixels, checkpoints, source datasets and metric values stay unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil

from .common import ROOT, concept_key, read_result_csv, write_csv, write_json
from .layout import timestamp, unique_directory
from .paths import BASE_FIELD, BASE_VALUE, benchmark_document, legacy_path, old_prefixes, portable, relative_path, relative_symlink, restore


def metadata_files(root, excluded_models=()):
    local = [root / name for name in ("results", "original", "cache")]
    files = {p for directory in local if directory.exists() for p in directory.rglob("*")
             if p.is_file() and not p.is_symlink() and p.suffix in {".json", ".csv"}}
    external = list((root.parent / "baseline").glob("*/runs")) + [root.parent / "casu-2/runs"]
    for directory in external:
        if directory.exists():
            for p in directory.rglob("*.json"):
                if "plans" in p.relative_to(directory).parts or p.name.endswith("manifest.json"):
                    files.add(p)
    excluded = [root / base / model for base in ("results", "original") for model in excluded_models]
    return sorted(p for p in files if not any(p.is_relative_to(directory) for directory in excluded))


def fingerprint_map(files):
    from .metrics import feature_fingerprint
    result = {}
    for path in files:
        if not path.name.endswith("manifest.json"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema") != "concept-benchmark-v1":
            continue
        current = restore(raw, path)
        grouped = defaultdict(list)
        for old, new in zip(raw["records"], current["records"]):
            grouped[concept_key(new.get("concept") or raw["task"])].append((old, new))
        for rows in grouped.values():
            for field in ("reference", "image"):
                if not all(field in old and field in new for old, new in rows):
                    continue
                actual = [Path(new[field]) if Path(new[field]).is_absolute() else path.parent / new[field]
                          for old, new in rows]
                if not all(p.is_file() for p in actual):
                    continue
                expected = feature_fingerprint(actual)
                old_paths = [Path(old[field]) for old, new in rows]
                result[feature_fingerprint(old_paths, portable=False, stat_paths=actual)] = expected
                result[feature_fingerprint(actual, portable=False)] = expected
    return result


def remap_fingerprints(value, fingerprints):
    if isinstance(value, dict):
        return {key: fingerprints.get(child, child) if key == "feature_fingerprint" and isinstance(child, str)
                else remap_fingerprints(child, fingerprints) for key, child in value.items()}
    if isinstance(value, list):
        return [remap_fingerprints(child, fingerprints) for child in value]
    return value


def migrate(root=ROOT, *, apply=False, excluded_models=()):
    root = Path(root).resolve()
    files = metadata_files(root, excluded_models)
    excluded = [root / base / model for base in ("results", "original") for model in excluded_models]
    links = []
    for directory in (root / "results", root / "original", root / "cache"):
        if directory.exists():
            for path in directory.rglob("*"):
                if any(path.is_relative_to(parent) for parent in excluded):
                    continue
                if path.is_symlink() and path.readlink().is_absolute():
                    links.append((path, str(path.readlink())))
    if not apply:
        print(f"[relative paths] Inspect {len(files)} metadata files; convert {len(links)} absolute links. "
              "Use --apply to convert paths without sampling or evaluation.")
        return None
    journal = unique_directory(root / "migrations", f"relative-paths-{timestamp()}")
    log = {"status": "started", "files": [], "links_changed": 0, "feature_caches_rebased": 0,
           "excluded_models": list(excluded_models)}
    write_json(journal / "migration.json", log, portable=True)
    write_json(journal / "old-link-targets.json", {relative_path(p, root): old for p, old in links})
    for index, (path, old) in enumerate(links):
        relative_symlink(path, legacy_path(old, root))
        log["links_changed"] += 1
        if (index + 1) % 2000 == 0:
            print(f"[relative links] {index + 1}/{len(links)}", flush=True)
    fingerprints = fingerprint_map(files)
    for path in files:
        original = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            raw = json.loads(original)
            if not (benchmark_document(raw) or old_prefixes(raw)
                    or isinstance(raw, dict) and raw.get(BASE_FIELD) == BASE_VALUE
                    or isinstance(raw, list) and any(isinstance(row, dict) and ("image" in row or row.get(BASE_FIELD) == BASE_VALUE) for row in raw)):
                continue
            current = remap_fingerprints(restore(raw, path, csv_row=True), fingerprints)
            encoded = portable(current, path)
            if encoded == raw:
                continue
            # Avoid changing unrelated historical serialization (e.g. Infinity).
            json.dumps(encoded, allow_nan=False)
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                raw = list(csv.DictReader(handle))
            if not raw or not any(set(row) & {"image", "reference", "manifest", "details_dir", "details_file"} for row in raw):
                continue
            current = read_result_csv(path)
            encoded = portable(current, path)
            if encoded == raw:
                continue
        backup = journal / "metadata" / path.relative_to(root.parent)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        if path.suffix == ".json":
            write_json(path, current, portable=True)
        else:
            write_csv(path, current, portable=True)
        log["files"].append(relative_path(path, root))
        if len(log["files"]) % 100 == 0:
            print(f"[relative metadata] {len(log['files'])} files converted", flush=True)
    import numpy as np
    for path in (root / "cache/fid").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as saved:
            old = str(saved["fingerprint"].item())
            new = fingerprints.get(old, old)
            if old == new:
                continue
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["fingerprint"] = new
        temporary = path.with_name(path.stem + "-relative-paths.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        log["feature_caches_rebased"] += 1
    log["status"] = "complete"
    write_json(journal / "migration.json", log, portable=True)
    print(f"[relative paths] {len(log['files'])} metadata files; {log['links_changed']} links; "
          f"{log['feature_caches_rebased']} feature caches. Journal: {relative_path(journal, root)}", flush=True)
    return log


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    from .model_profiles import MODEL_IDS
    parser.add_argument("--exclude-model", action="append", choices=tuple(MODEL_IDS), default=[],
                        help="避开该模型正在运行的采样目录；可重复指定")
    args = parser.parse_args()
    migrate(args.root, apply=args.apply, excluded_models=args.exclude_model)


if __name__ == "__main__":
    main()
