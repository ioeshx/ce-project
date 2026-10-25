"""Migrate legacy benchmark paths without resampling or changing protocols.

Preview: python -m benchmark.migrate_layout
Apply:   python -m benchmark.migrate_layout --apply --remove-test-results
Only explicit smoke/test results are removed. All checkpoints stay untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil

from .common import ROOT, digest, locked, read_json, slug, write_json
from .layout import run_label, sampling_identity, sampling_label, timestamp


def is_test(value):
    return bool(re.search(r"(?:^|[/_.-])(smoke|smoketest|test-results)(?:$|[/_.-])", str(value).lower()))


def manifests(workspace):
    roots = [workspace / "benchmark/results", workspace / "casu-2/runs",
             workspace / "retention_repair/runs", workspace / "output_response/runs"]
    result = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*manifest.json"):
            try:
                document = read_json(path)
            except (ValueError, OSError):
                continue
            if isinstance(document, dict) and document.get("schema") == "concept-benchmark-v1" and document.get("records"):
                result.append((path, document))
    return result


class Migration:
    def __init__(self, benchmark_root=ROOT):
        self.root = Path(benchmark_root).resolve()
        self.workspace = self.root.parent
        self.moves = {}
        self.metadata = {}
        self.deletions = set()
        self.reserved = set()
        self.feature_sources = {}
        self.kept = []
        self.tests = []

    def destination(self, parent, label, *, stable=False, epoch=None):
        name = slug(label) if stable else f"{slug(label)}-{timestamp(epoch)}"
        path = Path(parent) / name
        index = 2
        while path in self.reserved or path.exists():
            path = Path(parent) / f"{name}-run-{index}"
            index += 1
        self.reserved.add(path)
        return path

    def register(self, old, new, identity, label):
        old, new = Path(old), Path(new)
        if old != new:
            previous = self.moves.setdefault(str(old), str(new))
            if previous != str(new):
                raise ValueError(f"Ambiguous migration source: {old}")
        self.metadata[new / "run.json"] = {"schema": "benchmark-readable-run-v1", "label": label,
            "started_at": timestamp(self.start_time(old)), "identity": identity}
        return new

    @staticmethod
    def start_time(path):
        path = Path(path)
        if path.is_file():
            return path.stat().st_mtime
        if path.is_dir():
            images = list(path.glob("*.png"))
            if images:
                return min(p.stat().st_mtime for p in images)
            return path.stat().st_mtime
        return None

    def translated(self, value):
        value = str(value)
        # The most specific mapping already describes its final location.
        for old, new in sorted(self.moves.items(), key=lambda pair: len(pair[0]), reverse=True):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value

    def plan(self):
        from .model_profiles import MODEL_IDS
        results = self.root / "results"
        if (any((results / name).is_dir() for name in MODEL_IDS)
                and not any((results / task).is_dir() and not (results / task).is_symlink()
                            for task in ("instance", "style", "mscoco", "i2p"))
                and not list(results.glob("*-metric.csv"))):
            return self  # Model-scoped results already use the readable layout.
        import numpy as np
        from .metrics import feature_fingerprint
        all_manifests = manifests(self.workspace)
        self.tests = [(p, m) for p, m in all_manifests if is_test(p) or is_test(m.get("method", ""))]
        self.kept = [(p, m) for p, m in all_manifests if not (is_test(p) or is_test(m.get("method", "")))]
        test_refs = {str(Path(r["reference"]).parent) for _, m in self.tests for r in m["records"] if r.get("reference")}
        formal_refs = defaultdict(list)
        images = defaultdict(list)
        for path, manifest in self.kept:
            for ref in {str(Path(r["reference"]).parent) for r in manifest["records"] if r.get("reference")}:
                formal_refs[ref].append(manifest)
            for image in {str(Path(r["image"]).parent) for r in manifest["records"]}:
                images[image].append(manifest)
        smoke = self.root / "results/smoke"
        if smoke.exists():
            self.deletions.add(smoke)
        # Prepared smoke configs outside the main smoke directory are tests too.
        for plan_root in (self.root / "results/cfrr-plans", self.root / "results/output-response-plans"):
            if not plan_root.exists():
                continue
            for directory in plan_root.iterdir():
                if not directory.is_dir():
                    continue
                configs = [read_json(p) for p in directory.glob("*.json")]
                labels = [c.get("method") for c in configs if isinstance(c, dict) and c.get("method")]
                if labels and all(is_test(label) for label in labels):
                    self.deletions.add(directory)
                    continue
                if labels:
                    method = labels[0]
                    if (directory / "run.json").is_file() and read_json(directory / "run.json").get("identity") == {"method": method}:
                        continue
                    target = self.destination(plan_root, method, stable=True)
                    self.register(directory, target, {"method": method}, method)
        # Move original PNGs and attach the exact old identity before any rename.
        originals = self.root / "cache/originals"
        reference_dirs = set(formal_refs)
        if originals.exists():
            reference_dirs.update(str(p) for p in originals.iterdir() if p.is_dir())
        for old in sorted(reference_dirs):
            path = Path(old)
            if not path.is_relative_to(originals):
                continue
            usages = formal_refs.get(old, [])
            if not usages and (old in test_refs or self.small_test_cache(path)):
                if path.exists():
                    self.deletions.add(path)
                continue
            if not usages:
                raise ValueError(f"Cannot identify original cache safely: {path}")
            manifest = max(usages, key=lambda m: len(m["records"]))
            identity = sampling_identity(manifest["records"], manifest["base_model"], manifest["sampling"])
            if re.fullmatch(r"[a-f0-9]{64}", path.name) and digest(identity) != path.name:
                raise ValueError(f"Original protocol identity mismatch: {path}")
            label = run_label(manifest["task"], manifest["records"], manifest["sampling"])
            target = self.destination(self.root / "original" / manifest["task"], label, epoch=self.start_time(path))
            self.register(path, target, identity, label)
        # Method and sampling image directories, including planned-but-not-yet-generated sets.
        methods = {}
        for _, manifest in self.kept:
            key = (manifest["task"], manifest["method"])
            if key not in methods:
                existing = self.root / "results" / key[0] / slug(key[1])
                if (existing / "run.json").is_file() and read_json(existing / "run.json").get("identity") == {"method": key[1]}:
                    methods[key] = existing
                    continue
                target = self.destination(self.root / "results" / key[0], key[1], stable=True)
                methods[key] = target
                self.metadata[target / "run.json"] = {"schema": "benchmark-readable-run-v1", "label": key[1],
                    "started_at": timestamp(), "identity": {"method": key[1]}}
                # Locate old slug directory without relying on a particular hash function.
                parent = self.root / "results" / key[0]
                if parent.exists():
                    for child in parent.iterdir():
                        if child.is_dir() and (child.name == key[1] or re.fullmatch(re.escape(slug(key[1])) + r"-[a-f0-9]{8}", child.name)):
                            self.moves[str(child)] = str(target)
        for old, usages in sorted(images.items()):
            path = Path(old)
            if old in formal_refs:
                continue  # original adapter images point to the reference set
            manifest = usages[0]
            key = (manifest["task"], manifest["method"])
            identity = {"sampling": sampling_identity(manifest["records"], manifest["base_model"], manifest["sampling"]),
                        "adapter": manifest.get("adapter_identity", {"kind": "existing"})}
            if (path / "run.json").is_file() and read_json(path / "run.json").get("identity") == identity:
                continue
            label = run_label(manifest["task"], manifest["records"], manifest["sampling"])
            target = self.destination(methods[key] / "images", label, epoch=self.start_time(path))
            self.register(path, target, identity, label)
        # Evaluation timestamps become local, readable start times with ordinal collisions.
        for task_dir in (self.root / "results").iterdir():
            if not task_dir.is_dir() or task_dir.name not in {"instance", "style", "mscoco", "i2p"}:
                continue
            for directory in task_dir.glob("*/evaluations/*"):
                if not directory.is_dir():
                    continue
                if directory.name.startswith("evaluation-"):
                    continue
                try:
                    epoch = datetime.fromisoformat(directory.name.rsplit("-", 1)[0]).timestamp()
                except ValueError:
                    epoch = directory.stat().st_mtime
                target = self.destination(Path(self.translated(directory.parent)), "evaluation", epoch=epoch)
                self.moves[str(directory)] = str(target)
        # Identify feature arrays by their validated input fingerprint before paths move.
        claims = {}
        test_fingerprints = set()
        for test, collection in ((False, self.kept), (True, self.tests)):
            for _, manifest in collection:
                if manifest["task"] == "i2p":
                    continue
                grouped = defaultdict(list)
                for row in manifest["records"]:
                    from .common import concept_key
                    grouped[concept_key(row["concept"]) if manifest["task"] != "mscoco" else "mscoco"].append(row)
                for rows in grouped.values():
                    for field in ("reference", "image"):
                        paths = [Path(r[field]) for r in rows]
                        if not all(p.is_file() for p in paths):
                            continue
                        fp = feature_fingerprint(paths)
                        if test:
                            test_fingerprints.add(fp)
                        else:
                            claims[fp] = (manifest, rows, field, paths)
        for path in sorted((self.root / "cache/fid").glob("*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                fp = str(saved["fingerprint"].item())
                n = len(saved["features"])
            if fp not in claims and (fp in test_fingerprints or n < 50):
                self.deletions.add(path)
                continue
            if fp in claims:
                manifest, rows, field, paths = claims[fp]
                owner = "original" if field == "reference" else slug(manifest["method"])
                concept = rows[0].get("concept") or manifest["task"]
                label = f"{slug(concept)}-{field}-{len(rows)}-images-{sampling_label(manifest['sampling'])}"
                parent = self.root / "cache/fid" / manifest["task"] / owner
                target_dir = self.destination(parent, label, epoch=path.stat().st_mtime)
                target = target_dir / f"{slug(concept)}-{field}.npz"
                self.feature_sources[str(target)] = [str(p) for p in paths]
            else:
                label = f"unassigned-{n}-images"
                target_dir = self.destination(self.root / "cache/fid/unassigned", label, epoch=path.stat().st_mtime)
                target = target_dir / "features.npz"
            self.moves[str(path)] = str(target)
            self.metadata[target_dir / "run.json"] = {"schema": "benchmark-readable-run-v1", "label": label,
                "started_at": timestamp(path.stat().st_mtime), "identity": {"feature_fingerprint": fp}}
        for path in (self.root / "results").rglob(".plot-cache"):
            self.deletions.add(path)
        return self

    @staticmethod
    def small_test_cache(path):
        # The earliest failed smoke invocation has no persisted manifest. Its
        # three 64x64 PNGs are identifiable; never infer a full-resolution test.
        from PIL import Image
        paths = list(path.glob("*.png")) if path.is_dir() else []
        if not paths or len(paths) > 20:
            return False
        for p in paths:
            with Image.open(p) as image:
                if max(image.size) > 128:
                    return False
        return True

    def summary(self):
        return {"moves": self.moves, "delete_test_or_disposable_paths": [str(p) for p in sorted(self.deletions)],
                "retained_manifests": len(self.kept), "test_manifests": len(self.tests)}

    def apply(self, remove_tests=False):
        import numpy as np
        from .metrics import feature_fingerprint
        from .common import atomic_text
        report_dir = self.root / "migrations" / f"readable-layout-{timestamp()}"
        report_dir.mkdir(parents=True, exist_ok=False)
        report = self.summary()
        report.update(status="started", sampling_profiles_changed=False)
        write_json(report_dir / "migration.json", report)
        # Read-only snapshots protect historical manifests/configuration. Names
        # are ordinal, while their old paths are kept inside the JSON index.
        text_files = []
        for path in self.workspace.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.suffix not in {".json", ".csv", ".jsonl", ".md", ".txt", ".py", ".sh"}:
                continue
            if any(part in {".git", ".codex", ".agents", "migrations"} for part in path.parts):
                continue
            if any(path == d or d in path.parents for d in self.deletions):
                continue
            # Skip binary/model HF metadata outside this repository by not following symlinks.
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                continue
            replacements = list(self.moves.items())
            replacements += [(str(Path(a).relative_to(self.workspace)), str(Path(b).relative_to(self.workspace)))
                             for a, b in self.moves.items() if Path(a).is_relative_to(self.workspace) and Path(b).is_relative_to(self.workspace)]
            # One pass prevents a parent replacement from hiding a child mapping.
            lookup = dict(replacements)
            pattern = re.compile("|".join(re.escape(k) for k in sorted(lookup, key=len, reverse=True))) if lookup else None
            replaced = pattern.sub(lambda match: lookup[match.group()], text) if pattern else text
            if replaced != text:
                backup = report_dir / "documents" / f"document-{len(text_files)+1:04d}{path.suffix}"
                atomic_text(backup, text)
                text_files.append((path, replaced, backup))
        # Check every cache writer lock before moving anything. An active writer
        # must finish first; never race it by creating a second sampling cache.
        import contextlib
        import fcntl
        with contextlib.ExitStack() as stack:
            lock_paths = sorted(self.root.rglob(".sampling.lock"))
            for path in lock_paths:
                f = stack.enter_context(path.open("a"))
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            applied = {}
            for old, new in sorted(self.moves.items(), key=lambda pair: len(Path(pair[0]).parts)):
                source = Path(old)
                for before, after in sorted(applied.items(), key=lambda pair: len(pair[0]), reverse=True):
                    if str(source).startswith(before + "/"):
                        source = Path(after + str(source)[len(before):])
                        break
                target = Path(new)
                if source.exists() and source != target:
                    if target.exists():
                        raise FileExistsError(f"Refusing to merge existing directories: {target}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.rename(target)
                applied[old] = new
            for path, replaced, backup in text_files:
                atomic_text(self.translated(path), replaced)
            for path, metadata in self.metadata.items():
                write_json(path, metadata)
            # Rebase cache fingerprints, preserving feature values byte-for-byte.
            for target, old_paths in self.feature_sources.items():
                path = Path(target)
                paths = [Path(self.translated(p)) for p in old_paths]
                with np.load(path, allow_pickle=False) as data:
                    values = data["features"].copy()
                fp = feature_fingerprint(paths)
                temporary = path.with_name(path.stem + ".migrating.npz")
                np.savez_compressed(temporary, features=values, fingerprint=fp)
                temporary.replace(path)
                metadata = read_json(path.parent / "run.json")
                metadata["identity"] = {"feature_fingerprint": fp}
                metadata["source_paths"] = [str(p) for p in paths]
                write_json(path.parent / "run.json", metadata)
            # Repair relative and absolute symlinks that refer to moved originals.
            for parent, dirs, files in os.walk(self.workspace, followlinks=False):
                dirs[:] = [d for d in dirs if d not in {".git", ".codex", ".agents", "migrations"}]
                for name in dirs + files:
                    path = Path(parent) / name
                    if not path.is_symlink():
                        continue
                    link = os.readlink(path)
                    absolute = os.path.normpath(path.parent / link) if not os.path.isabs(link) else link
                    replaced = self.translated(absolute)
                    if replaced != absolute:
                        path.unlink()
                        path.symlink_to(os.path.relpath(replaced, path.parent) if not os.path.isabs(link) else replaced)
            if remove_tests:
                for path in sorted(self.deletions, key=lambda p: len(p.parts), reverse=True):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
            if (self.root / "cache/originals").exists() and not any((self.root / "cache/originals").iterdir()):
                (self.root / "cache/originals").rmdir()
        report.update(status="complete", documents=[{"old": str(p), "new": self.translated(p),
                      "backup": str(b)} for p, _, b in text_files], removed_test_results=remove_tests)
        write_json(report_dir / "migration.json", report)
        return report_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--remove-test-results", action="store_true")
    args = parser.parse_args(argv)
    migration = Migration().plan()
    if not args.apply:
        print(json.dumps(migration.summary(), ensure_ascii=False, indent=2))
        return
    with locked(ROOT / ".migration.lock"):
        print(migration.apply(args.remove_test_results))


if __name__ == "__main__":
    main()
