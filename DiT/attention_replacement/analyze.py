"""Stream saved distance/mask traces into experiment-level statistics and figures."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import html
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from safetensors.numpy import load as load_tensors


DIRECTORY = Path(__file__).resolve().parent
METRICS = ("d_target", "d_anchor", "margin", "mask_frequency", "selected_frequency")
LABELS = ("Mean d_target", "Mean d_anchor", "Mean margin",
          "Mask frequency", "Selected by winner")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def slug(value):
    return re.sub(r"[^\w.-]+", "-", value).strip("-.") or "item"


def concept_key(value):
    value = " ".join(value.casefold().split())
    return {"mickey mouse": "mickey", "claude monet": "monet",
            "pablo picasso": "picasso", "vincent van gogh": "van gogh"}.get(value, value)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(directory):
    """Discover metadata parents, never recurse through millions of tensor files."""
    directory = Path(directory).resolve()
    runs = []
    for base, dirs, files in os.walk(directory):
        if "metadata.json" in files:
            metadata = read_json(Path(base) / "metadata.json")
            if metadata.get("prototype_means"):
                runs.append(Path(base).parent)
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if not d.startswith("step_") and
                       d not in {"analysis", "cache", "images", "__pycache__"}]
    return sorted(set(runs))


class Reservoir:
    """Bounded random-priority sample; spatial means use every pixel, not this sample."""

    def __init__(self, capacity=100_000, seed=42):
        self.capacity = capacity
        self.values = np.empty((2 * capacity, 3), dtype=np.float32)
        self.priorities = np.empty(2 * capacity)
        self.used = 0
        self.pixels = 0
        self.rng = np.random.default_rng(seed)

    def add(self, target, anchor, margin):
        self.pixels += margin.size
        size = min(8, margin.size, self.capacity)
        positions = self.rng.choice(margin.size, size, replace=False)
        if self.used + size > len(self.values):
            self.compact()
        end = self.used + size
        self.values[self.used:end] = np.column_stack(
            [x.reshape(-1)[positions] for x in (target, anchor, margin)])
        self.priorities[self.used:end] = self.rng.random(size)
        self.used = end

    def compact(self):
        if self.used > self.capacity:
            indices = np.argpartition(self.priorities[:self.used], -self.capacity)[-self.capacity:]
            self.values[:self.capacity] = self.values[indices]
            self.priorities[:self.capacity] = self.priorities[indices]
            self.used = self.capacity

    def finish(self):
        self.compact()
        return self.values[:self.used].copy()


def reduce_maps(sums, counts):
    """sums: [layer,step,metric,h,w]; each observed map contributes one count."""
    def mean(total, count):
        count = np.asarray(count)
        divisor = count.reshape(count.shape + (1,) * (total.ndim - count.ndim))
        return np.divide(total, divisor, out=np.full(total.shape, np.nan), where=divisor > 0)

    return {
        "layer_maps": mean(sums.sum(axis=1), counts.sum(axis=1)),
        "step_maps": mean(sums.sum(axis=0), counts.sum(axis=0)),
        "global_maps": mean(sums.sum(axis=(0, 1)), counts.sum()),
        "layer_step": mean(sums.mean(axis=(-2, -1)), counts),
        "counts": counts,
    }


def collect_statistics(run, output, reservoir_size=100_000):
    samples = sorted(p for p in run.iterdir() if p.is_dir() and (p / "metadata.json").exists())
    first = read_json(samples[0] / "metadata.json")
    sites = sorted(first["sites"], key=lambda s: (s["layer"], s["branch"]))
    pairs = first["settings"]["pairs"]
    steps = list(range(first["settings"]["step_start"], first["settings"]["step_end"]))
    shape = tuple(sites[0]["grid"])
    if any(tuple(site["grid"]) != shape for site in sites):
        raise ValueError(f"Mixed spatial grids in {run}")
    site_index = {site["id"]: i for i, site in enumerate(sites)}
    step_index = {step: i for i, step in enumerate(steps)}
    branches = sorted({site["branch"] for site in sites})
    # Cohort 0 = all prompts; cohort 1 = prompt concept matching this target.
    sums = np.zeros((2, len(pairs), len(sites), len(steps), len(METRICS), *shape))
    counts = np.zeros((2, len(pairs), len(sites), len(steps)), dtype=np.int64)
    reservoirs = {(c, branch, group): Reservoir(reservoir_size, 42 + c)
                  for c in range(len(pairs)) for branch in branches for group in ("target", "other")}
    validation = {"mask_vs_margin_mismatches": 0, "winner_outside_mask": 0,
                  "union_vs_winner_mismatches": 0, "max_margin_distance_error": 0.0,
                  "missing_files": 0, "files_read": 0}
    inventory = []
    started = time.monotonic()
    for sample_i, sample in enumerate(samples):
        meta = read_json(sample / "metadata.json")
        if meta["settings"]["pairs"] != pairs or meta["sites"] != first["sites"]:
            raise ValueError(f"Mixed concept pairs/sites in {sample}")
        if (meta["settings"]["tau"], meta["settings"]["tau_by_concept"], meta["schedule"]) != (
                first["settings"]["tau"], first["settings"]["tau_by_concept"], first["schedule"]):
            raise ValueError(f"Mixed thresholds/schedules in {sample}")
        label = concept_key(meta["record"].get("concept", ""))
        matches = [bool(label) and label == concept_key(p["target"]) for p in pairs]
        observed = 0
        for step in steps:
            folder = sample / f"step_{step:04d}"
            if not folder.is_dir():
                validation["missing_files"] += len(sites)
                continue
            # Reading the small file as bytes avoids repeated mmap/pread overhead.
            for path in folder.iterdir():
                if path.suffix != ".safetensors" or path.stem not in site_index:
                    continue
                si, ti = site_index[path.stem], step_index[step]
                site = sites[si]
                tensors = load_tensors(path.read_bytes())
                winner = tensors["winner"]
                validation["union_vs_winner_mismatches"] += int(np.count_nonzero(
                    tensors["mask_union"] != (winner >= 0)))
                for c, pair in enumerate(pairs):
                    tar, anc = tensors[f"d_tar_{c}"], tensors[f"d_anc_{c}"]
                    margin, mask = tensors[f"margin_{c}"], tensors[f"mask_{c}"]
                    if tar.shape != shape or not all(np.isfinite(x).all() for x in (tar, anc, margin)):
                        raise ValueError(f"Invalid distance map in {path}")
                    selected = winner == c
                    fields = np.stack((tar, anc, margin, mask, selected))
                    sums[0, c, si, ti] += fields
                    counts[0, c, si, ti] += 1
                    if matches[c]:
                        sums[1, c, si, ti] += fields
                        counts[1, c, si, ti] += 1
                    tau = meta["settings"]["tau_by_concept"].get(pair["target"], meta["settings"]["tau"])
                    validation["mask_vs_margin_mismatches"] += int(np.count_nonzero(mask != (margin > tau)))
                    validation["winner_outside_mask"] += int(np.count_nonzero(selected & ~mask))
                    validation["max_margin_distance_error"] = max(validation["max_margin_distance_error"],
                        float(np.max(np.abs(margin - (anc - tar)))))
                    reservoirs[c, site["branch"], "target" if matches[c] else "other"].add(tar, anc, margin)
                observed += 1
        validation["files_read"] += observed
        # Whole missing step directories were already counted above.
        present_steps = sum((sample / f"step_{step:04d}").is_dir() for step in steps)
        validation["missing_files"] += present_steps * len(sites) - observed
        inventory.append({"sample": sample.name, "prompt_concept": meta["record"].get("concept", ""),
                          "seed": meta["record"]["seed"], "files_read": observed,
                          "expected_files": len(sites) * len(steps), "prompt": meta["record"]["prompt"]})
        if (sample_i + 1) % 25 == 0 or sample_i + 1 == len(samples):
            elapsed = time.monotonic() - started
            print(f"[read {run.parent.parent.parent.parent.name}/{run.parent.parent.parent.name}] "
                  f"{sample_i + 1}/{len(samples)} samples, {validation['files_read']:,} files, "
                  f"{elapsed:.0f}s", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "samples.csv", inventory)
    manifest = {"source_run": str(run), "samples": len(samples), "pairs": pairs,
                "settings": first["settings"], "sampling": first["sampling"], "sites": sites,
                "steps": steps, "timesteps": [first["schedule"]["timesteps"][t] for t in steps],
                "metrics": METRICS, "validation": validation, "groups": [],
                "statistics_seconds": time.monotonic() - started,
                "distribution_sampling": "8 uniform pixels per file, then random-priority reservoir; "
                    "spatial means and saved-mask frequencies use all pixels in all observed files",
                "threshold_scope": "Re-thresholding saved margins on the existing trajectory, not a new generation",
                "cohort_scope": "target_prompt matches record.concept to the erased target; other_prompts "
                    "are other prompt labels, not image-level safety ground truth"}
    for c, pair in enumerate(pairs):
        for branch in branches:
            indices = [i for i, site in enumerate(sites) if site["branch"] == branch]
            folder = output / f"concept_{c:02d}_{slug(pair['target'])}" / branch
            folder.mkdir(parents=True, exist_ok=True)
            arrays = {"layers": np.array([sites[i]["layer"] for i in indices]), "steps": np.array(steps),
                      "timesteps": np.array(manifest["timesteps"])}
            cohorts = ["all"]
            if counts[1, c, indices].sum() > 0:
                cohorts.append("target_prompt")
            if counts[1, c, indices].sum() > 0 and (counts[0, c, indices] - counts[1, c, indices]).sum() > 0:
                cohorts.append("other_prompts")
            for cohort in cohorts:
                if cohort == "all":
                    cs, cn = sums[0, c, indices], counts[0, c, indices]
                elif cohort == "target_prompt":
                    cs, cn = sums[1, c, indices], counts[1, c, indices]
                else:
                    cs = sums[0, c, indices] - sums[1, c, indices]
                    cn = counts[0, c, indices] - counts[1, c, indices]
                arrays.update({f"{cohort}__{key}": value for key, value in reduce_maps(cs, cn).items()})
            for label in ("target", "other"):
                reservoir = reservoirs[c, branch, label]
                arrays[f"distribution_{label}"] = reservoir.finish()
                arrays[f"pixels_{label}"] = np.array(reservoir.pixels)
            np.savez_compressed(folder / "statistics.npz", **arrays)
            manifest["groups"].append({"concept_index": c, "target": pair["target"], "anchor": pair["anchor"],
                "branch": branch, "directory": str(folder.relative_to(output)), "cohorts": cohorts,
                "tau": first["settings"]["tau_by_concept"].get(pair["target"], first["settings"]["tau"])})
    write_json(output / "manifest.json", manifest)
    return manifest


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.titlesize": 12,
                         "savefig.dpi": 120})
    return plt


def color_settings(layer_maps, step_maps, global_maps):
    from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm
    maps = [layer_maps, step_maps, global_maps[None]]
    distance = np.concatenate([x[:, :2].reshape(-1) for x in maps])
    distance = distance[np.isfinite(distance)]
    lo, hi = float(distance.min()), float(distance.max())
    distance_norm = LogNorm(lo, hi) if lo > 0 and hi / lo > 100 else Normalize(lo, max(hi, lo + 1e-9))
    extent = max(float(np.nanmax(np.abs(x[:, 2]))) for x in maps)
    extent = max(extent, 1e-9)
    return [("viridis", distance_norm), ("viridis", distance_norm),
            ("RdBu_r", TwoSlopeNorm(0, -extent, extent)),
            ("magma", Normalize(0, 1)), ("magma", Normalize(0, 1))]


def spatial_pages(folder, prefix, maps, row_labels, title, settings, metric_count):
    plt = pyplot()
    for start in range(0, len(maps), 6):
        page = maps[start:start + 6]
        fig, axes = plt.subplots(len(page), metric_count, squeeze=False,
                                 figsize=(3 * metric_count, 2.7 * len(page)), layout="constrained")
        for row, values in enumerate(page):
            for col in range(metric_count):
                ax = axes[row, col]
                cmap, norm = settings[col]
                ax.imshow(values[col], cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
                height, width = values[col].shape
                ax.set_xticks(np.unique(np.linspace(0, width - 1, 5, dtype=int)))
                ax.set_yticks(np.unique(np.linspace(0, height - 1, 5, dtype=int)))
                ax.set(xlabel="Token column w", ylabel="Token row h",
                       title=f"{row_labels[start + row]}\n{LABELS[col]}")
        for col in range(metric_count):
            fig.colorbar(axes[0, col].images[0], ax=list(axes[:, col]), shrink=.6, pad=.01)
        fig.suptitle(title + "\nSquared distances; margin = d_anchor - d_target; mask frequencies [0, 1]")
        fig.savefig(folder / f"{prefix}_{start // 6 + 1:02d}.png")
        plt.close(fig)


def scalar_plots(folder, data, cohort, group, title, metric_count):
    plt = pyplot()
    layers, steps, times = data["layers"], data["steps"], data["timesteps"]
    cells = data[f"{cohort}__layer_step"]
    fig, axes = plt.subplots(1, metric_count, figsize=(4 * metric_count, 5), layout="constrained")
    for col, ax in enumerate(np.atleast_1d(axes)):
        values = cells[:, :, col]
        kwargs = {"vmin": 0, "vmax": 1} if col >= 3 else {}
        if col == 2:
            limit = max(float(np.nanmax(np.abs(values))), 1e-9)
            kwargs = {"vmin": -limit, "vmax": limit}
        image = ax.imshow(values, aspect="auto", interpolation="nearest",
                          cmap="RdBu_r" if col == 2 else "magma" if col >= 3 else "viridis", **kwargs)
        ticks = np.unique(np.linspace(0, len(steps) - 1, min(6, len(steps)), dtype=int))
        ax.set_xticks(ticks, [f"{steps[i]}\nt={times[i]:.1f}" for i in ticks])
        tick_layers = np.unique(np.linspace(0, len(layers) - 1, min(12, len(layers)), dtype=int))
        ax.set_yticks(tick_layers, layers[tick_layers])
        ax.set(xlabel="Sampling step index / native timestep t", ylabel="Transformer layer index",
               title=LABELS[col] + " (spatial mean)")
        fig.colorbar(image, ax=ax, shrink=.7)
    fig.suptitle(title)
    fig.savefig(folder / "layer_timestep_summary.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, (axis_values, dim, xlabel) in enumerate(((layers, "layer_maps", "Transformer layer index"),
                                                      (steps, "step_maps", "Sampling step index"))):
        means = data[f"{cohort}__{dim}"].mean(axis=(-2, -1))
        for metric, label in ((0, "d_target"), (1, "d_anchor")):
            axes[0, column].plot(axis_values, means[:, metric], marker=".", label=label)
        axes[0, column].set(xlabel=xlabel, ylabel="Mean squared distance", title="Distance comparison")
        axes[0, column].legend()
        axes[1, column].plot(axis_values, means[:, 3], marker=".", label="Saved mask frequency")
        if metric_count == 5:
            axes[1, column].plot(axis_values, means[:, 4], marker=".", label="Selected by winner")
        axes[1, column].set(xlabel=xlabel, ylabel="Fraction of spatial positions", ylim=(-.02, 1.02),
                            title=f"Saved threshold tau = {group['tau']:g}")
        axes[1, column].legend()
    fig.suptitle(title)
    fig.savefig(folder / "distance_and_mask_trends.png")
    plt.close(fig)


def threshold_plots(folder, data, group, title):
    plt = pyplot()
    distributions = []
    for label in ("target", "other"):
        values, pixels = data[f"distribution_{label}"], int(data[f"pixels_{label}"])
        if len(values):
            distributions.append((label, values, pixels))
    if not distributions:
        return
    pooled = np.concatenate([values for _, values, _ in distributions])
    weights = np.concatenate([np.full(len(values), pixels / len(values)) for _, values, pixels in distributions])
    margin = pooled[:, 2]
    grid = np.unique(np.r_[np.quantile(margin, np.linspace(0, 1, 301)), 0., group["tau"]])
    total_pixels = sum(pixels for _, _, pixels in distributions)
    curves = {}
    for label, values, pixels in distributions:
        ordered = np.sort(values[:, 2])
        # Match the saved FP32 comparison semantics, including strict equality at tau.
        curves[label] = 1 - np.searchsorted(ordered, grid.astype(ordered.dtype), side="right") / len(ordered)
    all_curve = sum(curves[label] * pixels for label, _, pixels in distributions) / total_pixels
    curves["all"] = all_curve
    rows = [{"tau": float(tau), "all_coverage": float(all_curve[i]),
             "target_prompt_coverage": float(curves["target"][i]) if "target" in curves else "",
             "other_prompt_coverage": float(curves["other"][i]) if "target" in curves and "other" in curves else "",
             "estimate": "reservoir estimate on saved trajectory"} for i, tau in enumerate(grid)]
    write_csv(folder / "threshold_sweep.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(grid, all_curve, color="black", label="All prompts")
    if "target" in curves:
        ax.plot(grid, curves["target"], label="Target-concept prompts")
        if "other" in curves:
            ax.plot(grid, curves["other"], label="Other-concept prompts")
    linthresh = max(float(np.quantile(np.abs(margin), .25)), 1e-6)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.axvline(group["tau"], color="red", linestyle="--", label=f"Saved tau={group['tau']:g}")
    exact_mask = float(data["all__global_maps"][3].mean())
    ax.scatter([group["tau"]], [exact_mask], color="red", zorder=4, label="Exact saved mask coverage")
    ax.set(xlabel="Threshold tau (symlog scale)", ylabel="Fraction with d_anchor - d_target > tau",
           ylim=(-.02, 1.02), title=title + "\nEstimated threshold sweep on the saved trajectory")
    ax.legend(fontsize=8)
    fig.savefig(folder / "threshold_sweep.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for index, label in ((0, "d_target"), (1, "d_anchor")):
        axes[0].hist(np.log10(1 + pooled[:, index]), bins=80, weights=weights / weights.sum(),
                     histtype="step", label=label)
    axes[0].set(xlabel="log10(1 + squared distance)", ylabel="Probability per bin",
                title="Distance distributions (sampled pixels)")
    axes[0].legend()
    lo, hi = np.quantile(margin, [.005, .995])
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    bins = np.linspace(lo, hi, 101)
    for label, values, _ in distributions:
        legend = ("All prompts" if len(distributions) == 1 else
                  "Target-concept prompts" if label == "target" else "Other-concept prompts")
        axes[1].hist(values[:, 2], bins=bins, weights=np.full(len(values), 1 / len(values)),
                     histtype="step", label=legend)
    axes[1].axvline(group["tau"], color="red", linestyle="--", label=f"Saved tau={group['tau']:g}")
    axes[1].set(xlabel="Margin = d_anchor - d_target", ylabel="Probability per bin",
                title="Margin distributions (central 99% display range)")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.savefig(folder / "distance_distributions.png")
    plt.close(fig)
    quantiles = []
    for label, values, pixels in distributions:
        for q in (.01, .05, .25, .5, .75, .95, .99):
            quantiles.append({"prompt_group": label, "quantile": q,
                              "margin": float(np.quantile(values[:, 2], q)),
                              "sampled_pixels": len(values), "total_pixels": pixels})
    write_csv(folder / "margin_quantiles.csv", quantiles)


def render_group(output, manifest, group):
    folder = output / group["directory"]
    title = f"{group['target']} / {group['anchor']} | {group['branch']}"
    data = np.load(folder / "statistics.npz")
    metric_count = 5 if len(manifest["pairs"]) > 1 else 4
    rows = []
    # Share distance/margin scales between cohorts, layers and time averages.
    settings = color_settings(
        np.concatenate([data[f"{cohort}__layer_maps"] for cohort in group["cohorts"]]),
        np.concatenate([data[f"{cohort}__step_maps"] for cohort in group["cohorts"]] +
                       [data[f"{cohort}__global_maps"][None] for cohort in group["cohorts"]]),
        data["all__global_maps"])
    for cohort in group["cohorts"]:
        directory = folder / cohort
        directory.mkdir(exist_ok=True)
        label = {"all": "All prompts", "target_prompt": "Target-concept prompts",
                 "other_prompts": "Other-concept prompts"}[cohort]
        caption = title + " | " + label
        layer_maps, step_maps = data[f"{cohort}__layer_maps"], data[f"{cohort}__step_maps"]
        global_maps = data[f"{cohort}__global_maps"]
        spatial_pages(directory, "by_layer", layer_maps, [f"Layer {i}" for i in data["layers"]],
                      caption + " | Mean over samples and recorded timesteps", settings, metric_count)
        spatial_pages(directory, "by_timestep", step_maps,
                      [f"Step {step}, t={t:.2f}" for step, t in zip(data["steps"], data["timesteps"])],
                      caption + " | Mean over samples and layers", settings, metric_count)
        spatial_pages(directory, "global", global_maps[None], ["All layers / timesteps"],
                      caption + " | Mean over all observed samples, layers and timesteps", settings, metric_count)
        scalar_plots(directory, data, cohort, group, caption, metric_count)
        for dimension, maps, indices in (("layer", layer_maps, data["layers"]),
                                         ("step", step_maps, data["steps"]),
                                         ("global", global_maps[None], [-1])):
            for index, values in zip(indices, maps):
                row = {"cohort": cohort, "dimension": dimension, "index": int(index)}
                row.update({name: float(np.nanmean(value)) for name, value in zip(METRICS, values)})
                rows.append(row)
    write_csv(folder / "means.csv", rows)
    threshold_plots(folder, data, group, title)
    data.close()


def sample_figures(run, output, limit=8):
    """A few unaveraged maps for checking individual images, rather than only position frequencies."""
    if limit <= 0:
        return
    plt = pyplot()
    from PIL import Image
    samples = sorted(p for p in run.iterdir() if (p / "metadata.json").is_file())
    folder = output / "sample_examples"
    folder.mkdir(exist_ok=True)
    for i, index in enumerate(np.unique(np.linspace(0, len(samples) - 1, min(limit, len(samples)), dtype=int))):
        sample = samples[index]
        meta = read_json(sample / "metadata.json")
        steps = list(range(meta["settings"]["step_start"], meta["settings"]["step_end"]))
        step = steps[(i * max(1, len(steps) // max(limit, 1))) % len(steps)]
        sites = meta["sites"]
        site = sites[(i * max(1, len(sites) // max(limit, 1))) % len(sites)]
        path = sample / f"step_{step:04d}" / (site["id"] + ".safetensors")
        if not path.exists():
            continue
        tensors = load_tensors(path.read_bytes())
        c = i % len(meta["settings"]["pairs"])
        pair = meta["settings"]["pairs"][c]
        fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained")
        image_path = Path(meta["record"]["image"])
        if image_path.exists():
            with Image.open(image_path) as image:
                axes[0].imshow(image.convert("RGB"))
            axes[0].set(xlabel="Image x (pixels)", ylabel="Image y (pixels)", title="Generated image")
        else:
            axes[0].text(.5, .5, "Image not found", ha="center")
        distance_min = min(float(tensors[f"d_tar_{c}"].min()), float(tensors[f"d_anc_{c}"].min()))
        distance_max = max(float(tensors[f"d_tar_{c}"].max()), float(tensors[f"d_anc_{c}"].max()))
        raw_labels = ("d_target", "d_anchor", "d_anchor - d_target", "Binary mask M")
        for ax, key, name in zip(axes[1:], (f"d_tar_{c}", f"d_anc_{c}", f"margin_{c}", f"mask_{c}"), raw_labels):
            values = tensors[key]
            options = {"vmin": 0, "vmax": 1} if key.startswith("mask") else {}
            if key.startswith("margin"):
                extent = max(float(np.abs(values).max()), 1e-9)
                options = {"vmin": -extent, "vmax": extent}
            elif key.startswith("d_"):
                options = {"vmin": distance_min, "vmax": max(distance_max, distance_min + 1e-9)}
            image = ax.imshow(values, interpolation="nearest", cmap="RdBu_r" if key.startswith("margin") else
                              "magma" if key.startswith("mask") else "viridis", **options)
            ax.set(xlabel="Token column w", ylabel="Token row h", title=name.replace("Mean ", ""))
            fig.colorbar(image, ax=ax, shrink=.7)
        fig.suptitle(f"{sample.name} | {pair['target']} | Step {step}, {site['id']}\n"
                     + meta["record"]["prompt"][:160])
        fig.savefig(folder / f"example_{i:02d}.png")
        plt.close(fig)


def write_run_index(output, manifest):
    body = [f"<h1>{html.escape(Path(manifest['source_run']).name)}</h1>",
            f"<p>{manifest['samples']} samples; {manifest['validation']['files_read']:,} trace files. "
            "Spatial axes: token column w / token row h. Mask averages are trigger frequencies.</p>",
            "<p>Threshold curves re-threshold saved margins on an already edited trajectory; "
            "they do not predict image quality after regenerating with another threshold.</p>",
            '<p><a href="manifest.json">Source and validation</a> | <a href="samples.csv">Sample inventory</a></p>']
    for group in manifest["groups"]:
        directory = group["directory"]
        body.append(f"<h2>{html.escape(group['target'])} / {html.escape(group['anchor'])} — {group['branch']}</h2>")
        for name in ("threshold_sweep.png", "distance_distributions.png"):
            url = f"{directory}/{name}"
            body.append(f'<a href="{url}"><img src="{url}" style="max-width:650px;width:48%"></a>')
        for cohort in group["cohorts"]:
            body.append(f"<h3>{cohort}</h3><p>")
            for path in sorted((output / directory / cohort).glob("*.png")):
                relative = path.relative_to(output).as_posix()
                body.append(f'<a href="{relative}">{path.name}</a> &nbsp; ')
            body.append("</p>")
            url = f"{directory}/{cohort}/global_01.png"
            body.append(f'<a href="{url}"><img src="{url}" style="max-width:1200px;width:100%"></a>')
    examples = sorted((output / "sample_examples").glob("*.png"))
    if examples:
        body.append("<h2>Individual images and unaveraged maps</h2>")
        for path in examples:
            url = path.relative_to(output).as_posix()
            body.append(f'<a href="{url}"><img src="{url}" style="max-width:1200px;width:100%"></a>')
    (output / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Trace analysis</title>"
        "<style>body{font-family:sans-serif;margin:2em;max-width:1500px}a{color:#1766aa}</style>"
        + "\n".join(body), encoding="utf-8")


def analyze_run(arguments):
    run, output, heatmaps, plots_only = arguments
    run, output = Path(run), Path(output)
    print(f"[experiment] {run} -> {output}", flush=True)
    manifest = read_json(output / "manifest.json") if plots_only else collect_statistics(run, output)
    for group in manifest["groups"]:
        print(f"[plots] {group['target']} / {group['branch']}: {output}", flush=True)
        render_group(output, manifest, group)
    sample_figures(run, output, heatmaps)
    write_run_index(output, manifest)
    print(f"[complete] {output / 'index.html'}", flush=True)
    return {"source": str(run), "output": str(output), "samples": manifest["samples"],
            "files": manifest["validation"]["files_read"], "validation": manifest["validation"]}


def write_overview(output, reports):
    """Refresh cross-experiment tables alongside every batch, including plots-only runs."""
    rows = []
    for report in reports:
        folder = Path(report["output"])
        meta = read_json(folder / "manifest.json")
        for group in meta["groups"]:
            with np.load(folder / group["directory"] / "statistics.npz") as data:
                for cohort in group["cohorts"]:
                    row = {"experiment": str(folder.relative_to(output)), "target": group["target"],
                           "anchor": group["anchor"], "branch": group["branch"], "prompt_group": cohort,
                           "tau": group["tau"], "experiment_samples": meta["samples"],
                           "observed_maps": int(data[f"{cohort}__counts"].sum()),
                           "first_step": meta["steps"][0], "last_step": meta["steps"][-1]}
                    row.update({name: float(np.nanmean(value)) for name, value in
                                zip(METRICS, data[f"{cohort}__global_maps"])})
                    rows.append(row)
    write_csv(output / "summary.csv", rows)
    totals = {key: sum(report["validation"][key] for report in reports) for key in (
        "files_read", "missing_files", "mask_vs_margin_mismatches", "winner_outside_mask",
        "union_vs_winner_mismatches")}
    totals.update(experiments=len(reports), sample_records=sum(report["samples"] for report in reports))
    write_json(output / "validation_summary.json", totals)


def analyze_results(directory, output=None, workers=3, heatmaps=8, plots_only=False):
    directory = Path(directory).resolve()
    output = Path(output or DIRECTORY / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    runs = discover_runs(directory)
    if not runs:
        raise ValueError(f"No saved attention traces found in {directory}")
    jobs = []
    for run in runs:
        # Preserve the model/task/case/method identity when a results tree is supplied.
        try:
            relative = run.relative_to(DIRECTORY / "results")
        except ValueError:
            relative = run.relative_to(directory) if run != directory else Path(run.name)
        relative = Path(*(part for part in relative.parts if part != "traces"))
        jobs.append((str(run), str(output / relative), heatmaps, plots_only))
    started = time.monotonic()
    print(f"[analysis] {len(jobs)} experiments; output={output}; workers={workers}", flush=True)
    if workers == 1:
        reports = [analyze_run(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(analyze_run, jobs))
    write_json(output / "analysis_runs.json", {"elapsed_seconds": time.monotonic() - started,
                                               "experiments": reports})
    write_overview(output, reports)
    links = []
    for report in reports:
        relative = (Path(report["output"]) / "index.html").relative_to(output).as_posix()
        links.append(f'<li><a href="{relative}">{html.escape(relative)}</a> — '
                     f'{report["samples"]} samples, {report["files"]:,} files</li>')
    (output / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Attention trace analyses</title>"
        "<style>body{font-family:sans-serif;margin:2em}li{margin-bottom:1em}</style>"
        "<h1>Attention Replacement: trace analyses</h1>"
        '<p><a href="summary.csv">Cross-experiment summary CSV</a> | '
        '<a href="validation_summary.json">Consistency checks</a></p>'
        "<p>Each experiment contains spatial means by layer, by sampling time, and globally; "
        "distance distributions, threshold sweeps, mask checks and individual examples.</p><ul>"
        + "\n".join(links) + "</ul>", encoding="utf-8")
    print(f"[analysis finished] {time.monotonic() - started:.1f}s; {output / 'index.html'}", flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DIRECTORY / "results")
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY / "analysis")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--heatmaps", type=int, default=8, help="Unaveraged sample examples per experiment")
    parser.add_argument("--plots-only", action="store_true", help="Replot previously saved statistics.npz")
    args = parser.parse_args()
    if args.workers < 1 or args.heatmaps < 0:
        parser.error("workers must be positive and heatmaps must be nonnegative")
    analyze_results(args.results_dir, args.output_dir, args.workers, args.heatmaps, args.plots_only)


if __name__ == "__main__":
    main()
