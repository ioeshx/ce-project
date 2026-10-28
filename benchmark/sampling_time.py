"""Sampling wall-clock records, kept separate from image-quality metrics."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import atomic_text, locked, read_result_csv, write_csv, write_json
from .layout import method_directory, unique_directory
from .model_profiles import model_name, results_directory
from .execution import runtime_device


def local_now():
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def cuda_devices(adapter, options):
    """Find native model devices; plugins may also declare sampling_devices."""
    import torch
    if not torch.cuda.is_initialized():
        return []  # CPU runs must not initialize CUDA just to take a timestamp.
    devices = set()

    def add(value):
        if value is None or str(value) in {"cpu", "disk", "meta"}:
            return
        value = f"cuda:{value}" if str(value).isdigit() else value
        device = torch.device(runtime_device(value))
        if device.type == "cuda":
            devices.add(device.index if device.index is not None else torch.cuda.current_device())

    add(options.get("device"))
    for device in getattr(adapter, "sampling_devices", []):
        add(device)
    pipe = getattr(adapter, "pipe", None)
    for module in [pipe, *getattr(pipe, "components", {}).values()]:
        for device in (getattr(module, "hf_device_map", None) or {}).values():
            add(device)
        if isinstance(module, torch.nn.Module):
            for parameter in module.parameters():
                add(parameter.device)
    return sorted(devices)


def synchronize(devices):
    import torch
    for device in devices:
        torch.cuda.synchronize(device)


def publish(manifest, mode, started_at, total_seconds, phases, validation_seconds, manifest_path=None):
    """Publish only after all requested sampling phases completed successfully."""
    config = manifest["config"]
    model = model_name(manifest["base_model"], config.get("model_name"))
    root = results_directory(config["output_dir"], model)
    method = method_directory(root, manifest["task"], manifest["method"], manifest.get("case_id"))
    directory = unique_directory(method / "samplings", "sampling-" + started_at.strftime("%Y%m%d-%H%M%S"))
    generated = sum(p["generated_images"] for p in phases)
    reused = sum(p["reused_images"] for p in phases)
    row = {"method": manifest["method"], "task": manifest["task"], "model_name": model,
           "base_model": manifest["base_model"], "targets": manifest["targets"], "mode": mode,
           "status": "completed" if generated else "cached", "n": len(manifest["records"]),
           "generated_images": generated, "reused_images": reused,
           "total_seconds": total_seconds, "validation_seconds": validation_seconds}
    if manifest.get("case_id"):
        row["case_id"] = manifest["case_id"]
    for key in ("model_load_seconds", "generation_seconds", "image_save_seconds", "cleanup_seconds",
                "lock_wait_seconds", "image_check_seconds"):
        row[key] = sum(p[key] for p in phases)
    row["seconds_per_image"] = row["generation_seconds"] / generated if generated else None
    row["images_per_second"] = generated / row["generation_seconds"] if generated and row["generation_seconds"] else None
    for field in ("reference", "image"):
        selected = next((p for p in phases if p["field"] == field), None)
        for key in ("generated_images", "reused_images", "model_load_seconds", "generation_seconds"):
            row[f"{field}_{key}"] = selected[key] if selected else None
    row.update(started_at=started_at.isoformat(timespec="seconds"),
               finished_at=local_now().isoformat(timespec="seconds"),
               sampling=manifest["sampling"], sampling_options=config["sampling"],
               manifest=str(Path(manifest_path).resolve()) if manifest_path else None,
               details_file=str(directory / "timing.json"))
    document = {"schema": "benchmark-sampling-time-v1", "row": row, "phases": phases,
                "definitions": {"total_seconds": "This sample_manifest call before timing publication, including cache checks, lock waits, model load, generation, image saves and cleanup; excludes prepare/evaluate.",
                                "generation_seconds": "Sum of adapter.generate wall time, including native text encoding, denoising and decoding; synchronized on participating CUDA devices; excludes model loading and PNG writes.",
                                "seconds_per_image": "generation_seconds / newly generated images; null when all requested images are cached.",
                                "resume": "Only this invocation is timed; previous invocation records are retained under samplings/."}}
    write_json(directory / "timing.json", document, portable=True)
    path = root / "time" / f"{manifest['task']}-sampling-time.csv"
    with locked(path.with_suffix(".lock")):
        rows = []
        if path.exists():
            rows = read_result_csv(path)
        from .results import result_key
        rows = sorted([r for r in rows if result_key(r) != result_key(row)] + [row], key=result_key)
        write_csv(path, rows, portable=True)
        write_json(method / "sampling-time.json", document, portable=True)
        text = table(rows)
        atomic_text(path.with_suffix(".txt"), text + "\n")
    print(text, flush=True)
    print(f"[sampling time] {path}\n[sampling details] {directory / 'timing.json'}", flush=True)
    return document


def table(rows):
    columns = ["method", "status", "generated_images", "reused_images", "model_load_seconds",
               "generation_seconds", "image_save_seconds", "total_seconds", "seconds_per_image"]
    if any(r.get("case_id") for r in rows):
        columns.insert(1, "case_id")
    values = []
    for row in rows:
        values.append(["N/A" if row.get(key) is None or row.get(key) == "" else
                       f"{float(row[key]):.4f}" if key.endswith("seconds") or key == "seconds_per_image" else str(row[key])
                       for key in columns])
    widths = [max(len(key), *(len(row[i]) for row in values)) for i, key in enumerate(columns)]
    def line(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))
    return "\n".join(["Sampling time (seconds; generation speed uses newly generated images)", line(columns),
                      "-+-".join("-" * width for width in widths), *(line(row) for row in values)])
