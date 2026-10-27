"""Portable benchmark-relative artifact paths; absolute paths stay in memory."""
from __future__ import annotations

import json
import os
from pathlib import Path

BASE_FIELD = "path_base"
BASE_VALUE = "benchmark"
PATH_KEYS = {"output_dir", "image_output_dir", "cache_dir", "original_dir", "path", "checkpoint", "model_path",
             "q16_prompts", "q16_clip_path", "nudenet_weights", "manifest", "manifest_path",
             "details_dir", "details_file", "image", "reference", "image_alias", "reference_alias",
             "image_dir", "imported_from", "configuration_source", "config_path", "sampling_profile",
             "calibration_report", "report", "reuse_manifest", "resnet_weights_path"}
MODEL_KEYS = {"base_model", "model", "sd_ckpt", "clip_model", "vlm_model"}
JSON_COLUMNS = {"adapter", "sampling", "sampling_options", "evaluation"}
WORKSPACE_DIRS = {"baseline", "casu-2", "methods", "DiT", "retention_repair", "output_response"}


def benchmark_root():
    from .common import ROOT
    return Path(ROOT)


def artifact_root(path):
    """An artifact can belong to a copied checkout rather than the imported one."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    for parent in path.parents:
        if parent.name == "benchmark":
            return parent.resolve()
        if (parent / "benchmark").is_dir():
            return (parent / "benchmark").resolve()
    return benchmark_root()


def absolute_path(value, root=None):
    root = Path(root) if root is not None else benchmark_root()
    value = Path(value).expanduser()
    if not value.is_absolute():
        # Also accept familiar project-root spellings such as benchmark/results.
        value = root.parent / value if value.parts and value.parts[0] == "benchmark" else root / value
    return Path(os.path.abspath(value))  # Do not follow readable image aliases.


def relative_path(value, root=None):
    root = Path(root) if root is not None else artifact_root(value)
    return Path(os.path.relpath(absolute_path(value, root), root)).as_posix()


def local_model(value, root):
    return (Path(value).is_absolute() or value.startswith(("./", "../", "~/"))
            or (root / value).exists())


def path_value(key, value, root):
    if not isinstance(value, str) or not value or "://" in value or value.startswith(("data:", "<")):
        return False
    if key in MODEL_KEYS or key == "checkpoint":
        return local_model(value, root)
    return key in PATH_KEYS


def legacy_path(value, root, prefixes=()):
    """Rebase old checkout prefixes, even if the old checkout still exists."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return value
    for old in prefixes:
        if path.is_relative_to(old):
            return str(root.parent / path.relative_to(old))
    parts = path.parts
    if "benchmark" in parts:
        index = parts.index("benchmark")
        return str(root.joinpath(*parts[index + 1:]))
    for name in WORKSPACE_DIRS:
        if name in parts:
            return str(root.parent.joinpath(*parts[parts.index(name):]))
    return str(path)


def old_prefixes(value):
    found = set()
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, str) and key in PATH_KEYS | MODEL_KEYS and Path(child).is_absolute():
                    parts = Path(child).parts
                    if "benchmark" in parts:
                        found.add(Path(*parts[:parts.index("benchmark")]))
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return sorted(found, key=lambda p: len(str(p)), reverse=True)


def walk(value, root, convert, *, json_columns=False):
    if isinstance(value, list):
        return [walk(item, root, convert, json_columns=json_columns) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key == BASE_FIELD:
            continue
        if json_columns and key in JSON_COLUMNS and isinstance(child, str) and child[:1] in {"{", "["}:
            result[key] = json.dumps(walk(json.loads(child), root, convert), ensure_ascii=False, sort_keys=True)
        elif path_value(key, child, root):
            result[key] = convert(key, child)
        else:
            result[key] = walk(child, root, convert, json_columns=json_columns)
    return result


def portable(value, path):
    root = artifact_root(path)
    def convert(key, child):
        result = relative_path(child, root)
        return "./" + result if key in MODEL_KEYS | {"checkpoint"} and not result.startswith(".") else result
    result = walk(value, root, convert, json_columns=True)
    if isinstance(result, dict):
        result[BASE_FIELD] = BASE_VALUE
    elif isinstance(result, list):
        for row in result:
            if isinstance(row, dict):
                row[BASE_FIELD] = BASE_VALUE
    return result


def benchmark_document(value):
    return (isinstance(value, dict) and (
        str(value.get("schema", "")).startswith(("concept-benchmark-", "benchmark-"))
        or {"task", "base_model", "sampling"} <= value.keys()
        or {"task", "base_model", "settings"} <= value.keys()
        or isinstance(value.get("row"), dict) and "task" in value["row"]))


def restore(value, path, *, csv_row=False):
    root = artifact_root(path)
    if isinstance(value, list):
        return [restore(row, path, csv_row=csv_row) for row in value]
    if not isinstance(value, dict):
        return value
    marked = value.get(BASE_FIELD) == BASE_VALUE
    if not (marked or csv_row or benchmark_document(value)):
        return value
    prefixes = old_prefixes(value)
    def convert(key, child):
        return str(absolute_path(child, root)) if marked else legacy_path(child, root, prefixes)
    return walk(value, root, convert, json_columns=True)


def config_paths(value):
    root = benchmark_root()
    return walk(value, root, lambda key, child: str(absolute_path(child, root)))


def relative_symlink(link, target):
    """Atomically replace a legacy absolute link without touching image pixels."""
    import tempfile
    link, target = Path(link), Path(target)
    wanted = os.path.relpath(target, link.parent)
    if link.is_symlink() and os.readlink(link) == wanted:
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".relative-link-", dir=link.parent)
    os.close(fd)
    os.unlink(name)
    try:
        Path(name).symlink_to(wanted)
        os.replace(name, link)
    finally:
        if os.path.lexists(name):
            os.unlink(name)
