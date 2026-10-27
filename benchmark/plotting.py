"""Standalone CSV-only scientific figures, grouped by comparison protocol."""
from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from .layout import plot_name
from .common import slug, write_json

from .results import groups, protocol_label, read_rows


def number(row, key):
    try:
        return float(row[key]) if row.get(key) not in (None, "") else math.nan
    except (ValueError, TypeError):
        return math.nan


def plot(root, task):
    from .common import atomic_text
    config_dir = Path(tempfile.gettempdir()) / f"benchmark-plot-cache-user-{os.getuid()}"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    # A writable fontconfig cache also avoids warnings inside workspace sandboxes.
    font_config = config_dir / "fonts.conf"
    if not font_config.exists():
        from xml.sax.saxutils import escape
        atomic_text(font_config, '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
                    '<fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                    f'<cachedir>{escape(str(config_dir))}</cachedir></fontconfig>')
    os.environ.setdefault("FONTCONFIG_FILE", str(font_config))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(root) / "plots" / task
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    descriptions = []
    def save(fig, name):
        try:
            for extension in ("png", "pdf"):
                dest = directory / f"{name}.{extension}"
                fd, tmp = tempfile.mkstemp(suffix="." + extension, dir=directory)
                os.close(fd)
                try:
                    fig.savefig(tmp, dpi=180, bbox_inches="tight")
                    os.replace(tmp, dest)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                outputs.append(dest)
        finally:
            plt.close(fig)
    for group_number, (protocol, rows) in enumerate(groups(read_rows(root, task)), 1):
        name = plot_name(task, rows[0], group_number)
        descriptions.append({"name": name, "protocol_id": protocol, "rows": rows})
        methods = [r["method"] for r in rows]
        size = max(11, min(22, len(methods) * 1.8))
        count = 3 if task in {"instance", "style"} else 2
        has_vlm = any(r.get("vlm_accuracy") not in (None, "") for r in rows)
        if has_vlm:
            count += 1
        fig, axes = plt.subplots(1, count, figsize=(size, 4.9), squeeze=False)
        axes = axes[0]
        def bars(ax, keys, labels, ylabel):
            x = np.arange(len(methods))
            width = 0.8 / len(keys)
            for i, (key, label) in enumerate(zip(keys, labels)):
                values = [number(r, key) for r in rows]
                ax.bar(x + (i - (len(keys) - 1) / 2) * width, values, width, label=label)
            ax.set_xticks(x, methods, rotation=30, ha="right")
            ax.set_ylabel(ylabel)
            if "FID" in ylabel or "instances" in ylabel:
                ax.set_ylim(bottom=0)
            if len(keys) > 1:
                ax.legend(fontsize=7)
            if not any(math.isfinite(number(r, k)) for r in rows for k in keys):
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            ax.grid(axis="y", alpha=0.2)

        if task in {"instance", "style"}:
            concepts = sorted({key for row in rows for key, value in row.items()
                               if key.startswith("concept.") and key.endswith(".cs") and value not in (None, "")})
            bars(axes[0], concepts, [k[8:-3] for k in concepts], "CLIP Score (100 x cosine)")
            axes[0].set_title("Per-concept CS")
            bars(axes[1], ["fid_non_target_macro"], ["non-target"], "Macro FID (lower is better)")
            axes[1].set_title("Non-target preservation")
            for row in rows:
                x, y = number(row, "cs_target"), number(row, "fid_non_target_macro")
                if math.isfinite(x) and math.isfinite(y):
                    axes[2].scatter(x, y, s=55)
                    axes[2].annotate(row["method"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
            if all(not math.isfinite(number(r, "fid_non_target_macro")) for r in rows):
                axes[2].text(.5, .5, "N/A: all concepts targeted", transform=axes[2].transAxes, ha="center")
            axes[2].set(xlabel="Target CS", ylabel="Non-target macro FID", title="Erasure / preservation")
            axes[2].grid(alpha=0.2)
            if has_vlm:
                bars(axes[3], ["vlm_target_accuracy", "vlm_non_target_accuracy", "reference_vlm_accuracy"],
                     ["target", "non-target", "original (all)"], "VLM classification accuracy")
                axes[3].set_ylim(0, 1)
                axes[3].set_title("Single-label classification")
        elif task == "i2p":
            keys = ["nudenet_unsafe_rate"]
            labels = ["NudeNet (configured subset)"]
            if any(r.get("q16_n") not in (None, "") for r in rows):
                keys.insert(0, "q16_unsafe_rate")
                labels.insert(0, "Q16 (all)")
            bars(axes[0], keys, labels, "Unsafe image fraction")
            axes[0].set_ylim(0, 1)
            axes[0].set_title("Unsafe images / evaluated images")
            keys = sorted({key for r in rows for key in r if key.startswith("nudenet.") and key.endswith(".instances")})
            bars(axes[1], keys, [k.split(".")[1] for k in keys], "Detection instances (boxes)")
            axes[1].set_title("NudeNet accepted detections")
        elif task == "object":
            for ax, role in zip(axes, ("target", "non_target")):
                bars(ax, [f"resnet_{role}_top1", f"resnet_{role}_top3",
                          f"reference_resnet_{role}_top1", f"reference_resnet_{role}_top3"],
                     ["Method Top-1", "Method Top-3", "Original Top-1", "Original Top-3"], "Accuracy (full 1000 classes)")
                ax.set_ylim(0, 1)
                ax.set_title(role.replace("_", " "))
        else:
            bars(axes[0], ["cs", "reference_cs"], ["Method", "Original model"], "CLIP Score (100 x cosine)")
            bars(axes[1], ["fid"], ["FID"], "FID (lower is better)")
            axes[0].set_title("Prompt alignment")
            axes[1].set_title("FID vs original-model generations")
        fig.suptitle(f"{task} | group {group_number}\n{protocol_label(rows[0])}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .90))
        save(fig, name)
        if has_vlm or task == "object":
            from .classification_plots import figures
            for row in rows:
                save(figures(row, task), f"{name}-{slug(row['method'])}-confusion")
    # Remove obsolete protocol figures after all new figures were saved successfully.
    keep = set(outputs)
    for old in directory.glob(f"{task}-*.*"):
        if old.suffix in {".png", ".pdf"} and old not in keep:
            old.unlink()
    print(f"[plots] {directory} ({len(outputs)} files)")
    write_json(directory / "protocols.json", descriptions, portable=True)
    return outputs
