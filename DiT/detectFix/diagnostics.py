"""Per-sample masks, latent previews and run statistics for detectFix."""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from benchmark.common import read_json, write_json


def save_mask_plot(path, maps, labels, title):
    """Use a fixed [0, 1] scale for binary masks and detection-frequency maps."""
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    columns = min(3, len(maps))
    rows = math.ceil(len(maps) / columns)
    figure = Figure(figsize=(4.4 * columns, 4 * rows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = list(figure.subplots(rows, columns, squeeze=False).flat)
    figure.suptitle(title)
    for axis, values, label in zip(axes, maps, labels):
        plot = axis.imshow(values.float().numpy(), vmin=0, vmax=1, cmap="Blues",
                           interpolation="nearest", origin="upper")
        axis.set(title=label, xticks=[], yticks=[])
        figure.colorbar(plot, ax=axis, shrink=0.8)
    for axis in axes[len(maps):]:
        axis.set_visible(False)
    figure.savefig(path, dpi=130)
    figure.clear()


def decode_latents(adapter, latents, sampling):
    """Decode the actual noisy latent, using the pipeline's native VAE conversion."""
    pipe, vae = adapter.pipe, adapter.pipe.vae
    # Model CPU offload normally visits the VAE once, after denoising. Diagnostic
    # previews alternate between Transformer and VAE, so release each explicitly.
    offload_hooks = getattr(pipe, "_all_hooks", []) if sampling.get("cpu_offload") else []
    for hook in offload_hooks:
        if hook.model is pipe.transformer:
            hook.offload()
    if adapter.flux:
        latents = pipe._unpack_latents(latents, sampling["height"], sampling["width"], pipe.vae_scale_factor)
    hook = getattr(vae, "_hf_hook", None)
    device = getattr(hook, "execution_device", vae.device)
    latents = latents.to(device=device, dtype=vae.dtype)
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    with torch.inference_mode():
        images = vae.decode(latents, return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, output_type="pil")
    for hook in offload_hooks:
        if hook.model is vae:
            hook.offload()
    return images


class DetectFixDiagnostics:
    """Keep only small CPU masks and counters while saving previews as they arrive."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.summaries = {}

    def begin(self, records, sampling_config):
        self.sampling = sampling_config
        self.samples = []
        adapter, pairs = self.adapter, self.adapter.settings["pairs"]
        patch = 2 if adapter.flux else adapter.pipe.transformer.config.patch_size
        self.grid = (sampling_config["height"] // adapter.pipe.vae_scale_factor // patch,
                     sampling_config["width"] // adapter.pipe.vae_scale_factor // patch)
        for record in records:
            image = Path(record["image"])
            benchmark_image = image.parent.name != "images"
            setting = image.parent.name if benchmark_image else adapter.config["method"]
            if adapter.settings["trace_root"]:
                root = (Path(adapter.settings["trace_root"]) / adapter.config["model_name"] /
                        adapter.config["task"] / adapter.config["case_id"] / adapter.config["method"])
                if benchmark_image:
                    root /= setting
            else:
                root = ((image.parent.parent.parent if benchmark_image else image.parent.parent) /
                        "traces")
            directory = root / setting / image.stem if not adapter.settings["trace_root"] else root / image.stem
            directory.mkdir(parents=True, exist_ok=True)
            if directory.parent not in self.summaries:
                self.summaries[directory.parent] = {path.parent.name: read_json(path)
                    for path in directory.parent.glob("*/metadata.json")}
            # A missing image can be regenerated into an existing run directory.
            # Remove only this module's old plots, not unrelated sample files.
            for pattern in ("first_detection_step_*.png", "mask/step_*.png", "mask/step_*.safetensors",
                            "mid/step_*.png", "mask_mean.png", "mask_mean.safetensors"):
                for path in directory.glob(pattern):
                    path.unlink()
            self.samples.append({"directory": directory, "record": record, "detection_steps": [],
                "detections": [], "checked_steps": [],
                "mask_area_sum_by_pair": torch.zeros(len(pairs), dtype=torch.float64),
                "mask_count_by_pair": torch.zeros(len(pairs), dtype=torch.long),
                "detected_mask_sum": torch.zeros(len(pairs), *self.grid),
                "target_detection_counts": torch.zeros(len(pairs), dtype=torch.long),
                "union_mask_sum": torch.zeros(self.grid)})

    def record_detection(self, *, step, timestep, indices, masks, ratios, triggered, choices, latents):
        masks = masks.detach().reshape(len(indices), -1, *self.grid).cpu()
        ratios, triggered = ratios.cpu(), triggered.cpu()
        pairs = self.adapter.settings["pairs"]
        first_hits = []
        for local, index in enumerate(indices.tolist()):
            sample = self.samples[index]
            sample["checked_steps"].append(step)
            sample["mask_area_sum_by_pair"] += ratios[local]
            sample["mask_count_by_pair"] += 1
            hits = triggered[local]
            if not hits.any():
                continue
            if not sample["detection_steps"]:
                first_hits.append(index)
            sample["detection_steps"].append(step)
            sample["detections"].append({"step_index": step, "timestep": timestep,
                "triggered_pairs": hits.nonzero().flatten().tolist(), "target_area": ratios[local].tolist(),
                "selected_pair": int(choices[index])})
            detected_masks = masks[local] & hits[:, None, None]
            sample["detected_mask_sum"] += detected_masks.float()
            sample["target_detection_counts"] += hits.long()
            sample["union_mask_sum"] += detected_masks.any(0).float()
            directory = sample["directory"] / "mask"
            directory.mkdir(exist_ok=True)
            save_file({"mask": masks[local].contiguous(), "triggered": hits.contiguous(),
                       "target_area": ratios[local].contiguous(), "selected_pair": choices[index].reshape(1)},
                      str(directory / f"step_{step:04d}.safetensors"))
            labels = [f"{pair['target']} -> {pair['anchor']}\narea={float(ratios[local, k]):.1%}"
                      for k, pair in enumerate(pairs) if hits[k]]
            save_mask_plot(directory / f"step_{step:04d}.png", masks[local][hits], labels,
                           f"Detected mask | step {step} (denoising step {step + 1})")
        if first_hits:
            selected = torch.tensor(first_hits, device=latents.device)
            images = decode_latents(self.adapter, latents.index_select(0, selected), self.sampling)
            for index, image in zip(first_hits, images):
                image.save(self.samples[index]["directory"] / f"first_detection_step_{step:04d}.png")

    def step_end(self, pipeline, step, timestep, callback_kwargs):
        """Save updated latents from the first detection onward, even on later misses."""
        indices = [index for index, sample in enumerate(self.samples) if sample["detection_steps"]]
        if indices:
            latents = callback_kwargs["latents"]
            selected = torch.tensor(indices, device=latents.device)
            images = decode_latents(self.adapter, latents.index_select(0, selected), self.sampling)
            for index, image in zip(indices, images):
                directory = self.samples[index]["directory"] / "mid"
                directory.mkdir(exist_ok=True)
                image.save(directory / f"step_{step:04d}.png")
        return callback_kwargs

    def finish(self, decisions, forward_calls):
        pairs, roots = self.adapter.settings["pairs"], set()
        for sample, steps in zip(self.samples, decisions):
            directory, detected = sample["directory"], sample["detection_steps"]
            counts, sums = sample["mask_count_by_pair"], sample["mask_area_sum_by_pair"]
            metadata = {"record": sample["record"], "settings": self.adapter.settings,
                "sampling": self.sampling, "localization_sites": self.adapter.sites,
                "detection_steps": detected, "detection_step_numbers": [step + 1 for step in detected],
                "first_detection_step": detected[0] if detected else None,
                "first_detection_step_number": detected[0] + 1 if detected else None,
                "checked_steps": sample["checked_steps"], "detections": sample["detections"],
                "mask_count": int(counts.sum()), "mask_area_sum": float(sums.sum()),
                "mask_area_mean": float(sums.sum() / counts.sum()) if counts.sum() else None,
                "mask_count_by_pair": counts.tolist(), "mask_area_sum_by_pair": sums.tolist(),
                "steps": steps, "batch_forward_calls": forward_calls,
                "latent_preview": {"first_detection": "before_scheduler_update",
                                   "mid": "after_scheduler_update", "clean_prediction": False}}
            write_json(directory / "metadata.json", metadata)
            self.summaries[directory.parent][directory.name] = metadata
            roots.add(directory.parent)
            if self.adapter.settings["detection_mode"] == "every_step" and detected:
                target_counts = sample["target_detection_counts"]
                mean = sample["detected_mask_sum"] / target_counts.clamp_min(1)[:, None, None]
                union = sample["union_mask_sum"] / len(detected)
                save_file({"target_mask_mean": mean.contiguous(), "union_mask_mean": union.contiguous(),
                           "target_detection_counts": target_counts.contiguous()},
                          str(directory / "mask_mean.safetensors"))
                hits = target_counts > 0
                labels = [f"Union of detected targets\nmean over {len(detected)} detected steps"]
                labels += [f"{pair['target']}\nmean over {int(target_counts[k])} detected steps"
                           for k, pair in enumerate(pairs) if hits[k]]
                save_mask_plot(directory / "mask_mean.png", torch.cat((union[None], mean[hits])),
                               labels, "Mean detected masks | untriggered steps excluded")
        # Cached sample metadata is loaded once per setting. Replacing an entry
        # preserves aggregate statistics across batches and resumed runs.
        for root in roots:
            metadata = list(self.summaries[root].values())
            first_steps = [item["first_detection_step"] for item in metadata
                           if item["first_detection_step"] is not None]
            mask_count = sum(item["mask_count"] for item in metadata)
            mask_sum = sum(item["mask_area_sum"] for item in metadata)
            summary = {"sample_count": len(metadata), "detected_sample_count": len(first_steps),
                "undetected_sample_count": len(metadata) - len(first_steps),
                "mean_first_detection_step": sum(first_steps) / len(first_steps) if first_steps else None,
                "mean_first_detection_step_number": (sum(first_steps) / len(first_steps) + 1
                                                     if first_steps else None),
                "mask_count": mask_count, "mean_mask_area": mask_sum / mask_count if mask_count else None,
                "mask_area_denominator": "all computed (sample, detection step, target pair) masks",
                "pairs": []}
            for k, pair in enumerate(pairs):
                count = sum(item["mask_count_by_pair"][k] for item in metadata)
                area = sum(item["mask_area_sum_by_pair"][k] for item in metadata)
                summary["pairs"].append({**pair, "mask_count": count,
                                         "mean_mask_area": area / count if count else None})
            write_json(root / "summary.json", summary)
