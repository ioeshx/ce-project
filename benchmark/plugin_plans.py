"""Immutable readable experiment plans shared by method integration entry points."""
from __future__ import annotations

from pathlib import Path

from .common import file_digest, read_json, slug, write_csv, write_json
from .layout import registered_directory, sampling_label, timestamp


def plugin_plan(result_root, namespace, config, *, phase, subset=None, dry_run=False):
    """Return config/manifest paths, resuming only a matching immutable plan.

    Each task is independent, so a later COCO-only invocation finds the same plan
    created by an instance+COCO suite. Full settings and weight checksums live in
    run.json rather than in filenames. Dry runs never create result directories.
    """
    import copy
    config = copy.deepcopy(config)
    task, method = config["task"], config["method"]
    checkpoint = Path(config["adapter"]["checkpoint"])
    identities = {p.name: file_digest(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
    # A selected subset's location depends on this directory. Compare its rows,
    # not the eventual location, when selecting an existing plan.
    if subset is not None:
        config["dataset"]["path"] = f"<plan>/{task}-subset.csv"
    identity = {"config": config, "subset": subset, "checkpoint_files": identities}
    label = f"{task}-{sampling_label(config['sampling'])}"
    base = Path(result_root) / namespace
    if dry_run:
        directory = base / slug(method) / f"{label}-planned-{timestamp()}"
    else:
        if phase in {"sample", "evaluate"}:
            owner = next((p for p in sorted(base.iterdir()) if p.is_dir() and (p / "run.json").is_file()
                          and read_json(p / "run.json").get("identity") == {"method": method}), None) if base.exists() else None
            if owner is None:
                raise ValueError(f"No prepared method plan for {method}; run --phase prepare first.")
        else:
            owner = registered_directory(base, method, {"method": method}, stable=True)
        directory = None
        for candidate in sorted(owner.iterdir()):
            if candidate.is_dir() and (candidate / "run.json").is_file():
                if read_json(candidate / "run.json").get("identity") == identity:
                    directory = candidate
                    break
        if directory is None:
            if phase in {"sample", "evaluate"}:
                raise ValueError(f"No matching prepared {task} plan for {method}; run --phase prepare with these settings first.")
            directory = registered_directory(owner, label, identity)
    if subset is not None:
        config["dataset"]["path"] = str(directory / f"{task}-subset.csv")
    path = directory / f"{task}.json"
    manifest = directory / f"{task}-manifest.json"
    if not dry_run:
        if path.exists():
            if read_json(path) != config:
                raise ValueError(f"Stored plan changed unexpectedly: {path}")
        elif phase in {"sample", "evaluate"}:
            raise ValueError(f"Prepared configuration is missing: {path}")
        else:
            if subset is not None:
                write_csv(directory / f"{task}-subset.csv", subset)
            write_json(path, config)
        if subset is not None:
            import csv
            with Path(config["dataset"]["path"]).open(encoding="utf-8", newline="") as handle:
                actual = list(csv.DictReader(handle))
            if actual != subset:
                raise ValueError("Prepared prompt subset changed; refusing to modify the historical plan.")
        if phase in {"sample", "evaluate"} and not manifest.is_file():
            raise ValueError(f"Prepared image manifest is missing: {manifest}; rerun --phase prepare.")
    return path, manifest
