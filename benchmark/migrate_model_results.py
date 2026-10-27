"""Move legacy SD1.4 results under results/sd14, preserving scores and PNGs.

Preview: python -m benchmark.migrate_model_results
Apply:   python -m benchmark.migrate_model_results --apply
Metadata backups and a migration journal are kept under benchmark/migrations.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re

from .common import ROOT, atomic_text, digest, read_json, write_csv, write_json
from .layout import timestamp
from .model_profiles import MODEL_IDS


class ModelResultsMigration:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.results = self.root / "results"
        self.destination = self.results / "sd14"
        self.moves = {str(p): str(self.destination / p.name) for p in sorted(self.results.iterdir())
                      if p.name not in MODEL_IDS and not p.name.startswith(".")
                      and not (p.is_symlink() and p.resolve() == self.destination / p.name)}
        alternatives = "|".join(re.escape(p) for p in sorted(self.moves, key=len, reverse=True))
        self.pattern = re.compile(f"(?:{alternatives})(?=$|/|[\"'\\s,}}])") if alternatives else None
        for p in self.results.glob("*-metric.csv"):
            with p.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("base_model") != MODEL_IDS["sd14"]:
                        raise ValueError(f"Legacy results contain another model; classify it before migration: {p}")
        self.files = self.metadata_files()
        self.features = {}

    def translate(self, text):
        if text == str(self.results):
            return str(self.destination)
        return self.pattern.sub(lambda m: self.moves[m.group()], text) if self.pattern else text

    def metadata_files(self):
        roots = [self.results, self.root.parent / "casu-2/runs"]
        roots.extend((self.root.parent / "baseline").glob("*/runs"))
        roots.extend((self.root.parent / "methods").glob("**/runs"))
        return sorted({p for root in roots if root.exists() for p in root.rglob("*")
                       if p.is_file() and p.suffix in {".json", ".csv", ".txt"}})

    def rewrite(self, value):
        if isinstance(value, str):
            return self.translate(value)
        if isinstance(value, list):
            return [self.rewrite(v) for v in value]
        if isinstance(value, dict):
            result = {k: self.rewrite(v) for k, v in value.items()}
            if all(k in result for k in ("task", "base_model", "sampling")):
                result["model_name"] = "sd14"
            return result
        return value

    def prepare_feature_rebases(self):
        """Re-key existing Inception features when only image paths change."""
        from .metrics import feature_fingerprint
        from .paths import relative_path
        version = importlib.metadata.version("torch-fidelity")
        seen = set()
        for path in self.files:
            if not path.name.endswith("manifest.json"):
                continue
            document = read_json(path)
            if not isinstance(document, dict) or document.get("schema") != "concept-benchmark-v1":
                continue
            grouped = defaultdict(list)
            for row in document["records"]:
                grouped[row.get("concept") or document["task"]].append(row)
            for rows in grouped.values():
                for field in ("image", "reference"):
                    if field not in rows[0]:
                        continue
                    paths = tuple(r[field] for r in rows)
                    if paths in seen or not any(self.translate(p) != p for p in paths):
                        continue
                    seen.add(paths)
                    if not all(Path(p).is_file() for p in paths):
                        continue
                    entries = [(relative_path(self.translate(p)), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
                    old = feature_fingerprint([Path(p) for p in paths])
                    new = hashlib.sha256(json.dumps(["torch-fidelity-inception-v3-compat-2048", version, entries]).encode()).hexdigest()
                    self.features[old] = new
                    self.features[feature_fingerprint([Path(p) for p in paths], portable=False)] = new

    def apply(self):
        if not self.moves:
            print("[migration] Results already use model directories; nothing to move.")
            return None
        for old, new in self.moves.items():
            if Path(new).exists():
                raise ValueError(f"Migration destination already exists: {new}")
        self.prepare_feature_rebases()
        journal = self.root / "migrations" / f"model-results-{timestamp()}"
        journal.mkdir(parents=True, exist_ok=False)
        log = {"status": "started", "moves": self.moves, "metadata_backups": [], "feature_rebases": self.features}
        write_json(journal / "migration.json", log)
        # Back up only metadata that actually changes. Checkpoints and pixels are
        # never rewritten. Cache feature arrays stay identical; only keys change.
        changes = []
        for path in self.files:
            original = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                before = json.loads(original)
                after = self.rewrite(before)
                changed = before != after
                if not changed:
                    continue  # Leave unrelated historical config serialization intact.
                text = json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            elif path.suffix == ".csv":
                rows = list(csv.DictReader(io.StringIO(original)))
                after = [self.rewrite(row) for row in rows]
                changed = rows != after
                text = None
            else:
                text = self.translate(original)
                changed = text != original
            if not changed:
                continue
            backup = journal / "metadata" / path.relative_to(self.root.parent)
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(path.read_bytes())
            log["metadata_backups"].append({"path": str(path), "backup": str(backup)})
            changes.append((Path(self.translate(str(path))), text, after if path.suffix == ".csv" else None))
        write_json(journal / "migration.json", log)
        self.destination.mkdir(parents=True, exist_ok=True)
        for old, new in self.moves.items():
            Path(old).rename(new)
        for path, text, rows in changes:
            if rows is not None:
                write_csv(path, rows)
            else:
                atomic_text(path, text)
        rebased = 0
        import numpy as np
        for path in (self.root / "cache/fid").rglob("*.npz"):
            with np.load(path, allow_pickle=False) as saved:
                fingerprint = str(saved["fingerprint"].item())
                if fingerprint not in self.features:
                    continue
                arrays = {k: saved[k].copy() for k in saved.files}
            arrays["fingerprint"] = self.features[fingerprint]
            temporary = path.with_name(path.stem + "-model-migration.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(path)
            rebased += 1
        for path in (self.root / "cache/fid").rglob("run.json"):
            info = read_json(path)
            identity = info.get("identity", {})
            old = identity.get("feature_fingerprint")
            if old in self.features:
                identity["feature_fingerprint"] = self.features[old]
                write_json(path, info)
        log.update(status="complete", metadata_changed=len(changes), feature_caches_rebased=rebased)
        write_json(journal / "migration.json", log)
        print(f"[migration] {len(self.moves)} entries -> {self.destination}; {len(changes)} metadata files; {rebased} feature caches; journal {journal}")
        return journal


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    migration = ModelResultsMigration(args.root)
    if args.apply:
        migration.apply()
    else:
        for old, new in migration.moves.items():
            print(f"{old} -> {new}")


if __name__ == "__main__":
    main()
