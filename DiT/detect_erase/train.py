"""Learn CLIP detect conditions and negative CFG prompts with frozen model weights."""
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import argparse
import copy
import csv
import json
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from benchmark.common import configuration, read_json, slug, write_json
from benchmark.dit_cases import definitions, select_cases
from benchmark.model_profiles import DIT_MODELS
from benchmark.sampling_time import cuda_devices, synchronize
from DiT.detect_erase.outputs import prompt_checkpoint, run_directory, run_log, update_run
from DiT.detect_erase.prompts import PromptModel, classification_loss, supervised_contrastive_loss
from DiT.detect_erase.encoding import TextEncoder, cosine_scores

HERE = Path(__file__).resolve().parent


class TrainingTrace:
    """Optional SwanLab curves and local step/epoch timing tables."""

    def __init__(self, args, directory, config):
        import swanlab

        self.directory = directory
        self.run = swanlab.init(project=args.swanlab_project,
                                name=args.swanlab_name or f"train-{config['model_name']}-{directory.name}",
                                mode=args.swanlab_mode, log_dir=str(directory.parent / "swanlog"),
                                config={"model": config["base_model"], "sampling": config["sampling"],
                                        "training": read_json(directory / "training.json")})
        self.loss_file = (directory / "loss.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self.loss_writer = csv.DictWriter(self.loss_file, fieldnames=[
            "target", "stage", "epoch", "update", "global_update", "timestep", "loss", "learning_rate",
            "gradient_norm", "parameter_norm", "batch_size", "accuracy", "positive_cosine", "negative_cosine",
            "ti_loss", "classification_loss", "classification_batch_size"])
        self.loss_writer.writeheader()
        self.epoch_file = (directory / "epochs.csv").open("w", encoding="utf-8", newline="", buffering=1)
        self.epoch_writer = csv.DictWriter(self.epoch_file, fieldnames=[
            "target", "stage", "epoch", "updates", "examples", "mean_loss", "accuracy", "seconds", "partial"])
        self.epoch_writer.writeheader()
        self.devices, self.epochs, self.global_update = [], [], 0
        self.started = time.perf_counter()

    def start_epoch(self, target, stage, epoch):
        synchronize(self.devices)
        self.epoch_started = time.perf_counter()
        self.current = dict(target=target, stage=stage, epoch=epoch, updates=0, examples=0)
        self.loss_sum, self.correct_sum, self.has_accuracy = 0., 0., False

    def log_step(self, update, loss, optimizer, parameters, batch_size, timestep=None, metrics=None):
        self.global_update += 1
        gradient_norm = torch.stack([p.grad.detach().float().norm().square() for p in parameters]).sum().sqrt().item()
        parameter_norm = torch.stack([p.detach().float().norm().square() for p in parameters]).sum().sqrt().item()
        row = {key: self.current[key] for key in ("target", "stage", "epoch")}
        row.update(update=update, global_update=self.global_update, timestep=timestep, loss=loss.item(),
                   learning_rate=optimizer.param_groups[0]["lr"], gradient_norm=gradient_norm,
                   parameter_norm=parameter_norm, batch_size=batch_size, **(metrics or {}))
        self.loss_writer.writerow(row)
        self.current["updates"] += 1
        self.current["examples"] += batch_size
        self.loss_sum += row["loss"] * batch_size
        if metrics and "accuracy" in metrics:
            self.correct_sum += metrics["accuracy"] * batch_size
            self.has_accuracy = True
        prefix = f"train/{slug(row['target'])}/{row['stage']}"
        self.run.log({f"{prefix}/{key}": value for key, value in row.items()
                      if isinstance(value, (int, float))}, step=self.global_update)

    def end_epoch(self, expected_examples):
        synchronize(self.devices)
        row = {**self.current, "mean_loss": self.loss_sum / self.current["examples"],
               "accuracy": self.correct_sum / self.current["examples"] if self.has_accuracy else None,
               "seconds": time.perf_counter() - self.epoch_started,
               "partial": self.current["examples"] < expected_examples}
        self.epochs.append(row)
        self.epoch_writer.writerow(row)
        prefix = f"epoch/{slug(row['target'])}/{row['stage']}"
        self.run.log({f"{prefix}/{key}": value for key, value in row.items()
                      if isinstance(value, (int, float))}, step=self.global_update)
        print(f"[{row['target']}:{row['stage']}] epoch={row['epoch']} time={row['seconds']:.2f}s "
              f"mean_loss={row['mean_loss']:.6f} partial={row['partial']}", flush=True)

    def finish(self):
        synchronize(self.devices)
        total = time.perf_counter() - self.started
        epochs = sum(row["seconds"] for row in self.epochs)
        summary = dict(total_seconds=total, epoch_seconds=epochs, outside_epochs_seconds=total - epochs,
                       updates=self.global_update, epochs=self.epochs)
        write_json(self.directory / "training_summary.json", summary)
        self.run.log({"time/total_seconds": total, "time/epoch_seconds": epochs,
                      "time/outside_epochs_seconds": total - epochs}, step=self.global_update + 1)
        self.loss_file.close()
        self.epoch_file.close()
        self.run.finish()
        print(f"Training time: {total:.2f}s (epochs: {epochs:.2f}s)", flush=True)


def encode_dataset(model, texts, batch_size, *, clip_only=False):
    """Cache frozen query features; the learned token always keeps its encoder graph."""
    batches = [model.encode(texts[start:start + batch_size], clip_only=clip_only)
               for start in range(0, len(texts), batch_size)]
    return tuple(torch.cat([batch[i].detach().cpu() for batch in batches]) for i in range(len(batches[0])))


def train_clip(text, entry, rows, parameters, args, tracker, target):
    features = encode_dataset(text, [row["prompt"] for row in rows], args.batch_size, clip_only=True)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    for epoch in range(1, args.epochs + 1):
        if tracker:
            tracker.start_epoch(target, "detect", epoch)
        for indices in torch.randperm(len(rows), generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = tuple(value[indices].to(text.device) for value in features)
            truth = labels[indices].to(text.device)
            prototype = text.condition(entry, "detect", features_only=True)
            scores = cosine_scores(batch, prototype).squeeze(1)
            loss = classification_loss(scores, truth, args.tau, args.temperature)
            metrics = None
            if tracker:
                with torch.no_grad():
                    metrics = {"accuracy": ((scores >= args.tau) == truth.bool()).float().mean().item()}
                    for label, key in ((1, "positive_cosine"), (0, "negative_cosine")):
                        if (truth == label).any():
                            metrics[key] = scores[truth == label].mean().item()
            loss.backward()
            optimizer.step()
            step += 1
            if tracker:
                tracker.log_step(step, loss, optimizer, parameters, len(indices), metrics=metrics)
            if step % 50 == 0:
                print(f"[{target}:detect] epoch={epoch} update={step} loss={loss.item():.6f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                if tracker:
                    tracker.end_epoch(len(rows))
                return step
        if tracker:
            tracker.end_epoch(len(rows))
    return step


@torch.no_grad()
def encode_images(model, records, batch_size):
    """Encode fixed images with the native VAE shift/scaling, cached on CPU."""
    import numpy as np
    from PIL import Image

    latents = []
    vae = model.pipe.vae
    for start in range(0, len(records), batch_size):
        pixels = []
        for record in records[start:start + batch_size]:
            with Image.open(record["image"]) as image:
                pixels.append(torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1)
        z = vae.encode(torch.stack(pixels).to(device=vae.device, dtype=vae.dtype)).latent_dist.mode()
        z = (z - (vae.config.shift_factor or 0)) * vae.config.scaling_factor
        latents.append(z.float().cpu())
    return torch.cat(latents)


def ti_images(model, rows, args, stage_dir, target):
    """Generate one fixed base-model image per positive prompt, and reuse the cache."""
    from benchmark.adapters import atomic_image
    from benchmark.data import valid_image
    from benchmark.dit_adapter import DiTAdapter
    from benchmark.sampling_profile import canonical_protocol

    directory = stage_dir / "ti_images" / slug(target).lower().replace("-", "_")
    directory.mkdir(parents=True, exist_ok=True)
    sampling = {**model.config["sampling"], "negative_prompt": ""}
    sampling["steps"] = args.ti_generation_steps or sampling["steps"]
    seed = args.seed if args.ti_seed is None else args.ti_seed
    records = [{"prompt": row["prompt"], "seed": seed + index,
                "image": str(directory / f"positive_{index:03d}.png")} for index, row in enumerate(rows)]
    identity = dict(base_model=model.config["base_model"], sampling=canonical_protocol(sampling), records=records)
    path = directory / "images.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError(f"TI image configuration changed; use a new run directory: {directory}")
    write_json(path, identity)
    missing = [record for record in records if not valid_image(record["image"])]
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        images = DiTAdapter.generate(model, batch, sampling)
        for image, record in zip(images, batch):
            atomic_image(image, record["image"])
    print(f"[{target}:ti-flow] images={len(records)}, generated={len(missing)}, cache={directory}", flush=True)
    return encode_images(model, records, args.batch_size)


def erase_images(model, entry, args, target):
    """Reuse a compatible concept-image batch, or generate a new random-seed batch."""
    from benchmark.adapters import atomic_image
    from benchmark.data import valid_image
    from benchmark.dit_adapter import DiTAdapter
    from benchmark.layout import timestamp, unique_directory
    from benchmark.sampling_profile import canonical_protocol

    root = args.erase_image_dir / model.config["model_name"] / slug(target).lower().replace("-", "_")
    sampling = {**model.config["sampling"], "negative_prompt": ""}
    sampling["steps"] = args.erase_image_generation_steps or sampling["steps"]
    protocol = dict(base_model=model.config["base_model"], prompt=target, sampling=canonical_protocol(sampling))
    path, records = None, []
    if args.erase_reuse_images:
        candidates = sorted(root.glob("*/images.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        previous = entry.get("erase_image_manifest")
        if previous and Path(previous).is_file():
            candidates.insert(0, Path(previous))
        for candidate in candidates:
            saved = read_json(candidate)
            if saved["protocol"] == protocol:
                path, records = candidate, saved["records"]
                break
    if path is None:
        directory = unique_directory(root, timestamp())
        path = directory / "images.json"
    for index in range(len(records), args.erase_image_count):
        records.append({"prompt": target, "seed": secrets.randbits(63),
                        "image": str(path.parent / f"concept_{index:03d}.png")})
    write_json(path, {"schema": "detect-erase-images-v1", "protocol": protocol, "records": records})
    selected = records[:args.erase_image_count]
    missing = [record for record in selected if not valid_image(record["image"])]
    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        images = DiTAdapter.generate(model, batch, sampling)
        for image, record in zip(images, batch):
            atomic_image(image, record["image"])
    print(f"[{target}:erase-images] images={len(selected)}, generated={len(missing)}, cache={path.parent}", flush=True)
    return encode_images(model, selected, args.batch_size), str(path)


def train_ti_flow(model, entry, rows, parameters, args, tracker, target, stage_dir):
    joint = args.detect_loss in {"ti-flow-bce", "ti-flow-supcon"}
    positive_rows = [row for row in rows if row["label"] == 1][:args.ti_image_count]
    clean = ti_images(model, positive_rows, args, stage_dir, target)
    entry["detect_ti_images"] = len(clean)
    if joint:
        text_rows = list({(row["prompt"], row["label"]): row for row in rows}.values())
        features = encode_dataset(model.text, [row["prompt"] for row in text_rows],
                                  args.classification_batch_size, clip_only=True)
        labels = torch.tensor([row["label"] for row in text_rows], dtype=torch.float32)
        entry["detect_classification_samples"] = len(text_rows)
        positive_indices = torch.where(labels == 1)[0]
        negative_indices = torch.where(labels == 0)[0]
        text_generator = torch.Generator().manual_seed(args.seed)
        per_class = args.classification_batch_size // 2
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    for epoch in range(1, args.epochs + 1):
        if tracker:
            tracker.start_epoch(target, "detect", epoch)
        for indices in torch.randperm(len(clean), generator=generator).split(args.batch_size):
            z0 = clean[indices].to(model.device)
            noise = torch.randn(z0.shape, generator=generator).to(model.device)
            sigma = torch.rand((len(indices),), generator=generator).to(model.device)
            amount = sigma[:, None, None, None]
            noisy = (1 - amount) * z0 + amount * noise
            velocity_target = noise - z0
            if model.flux:
                noisy = model.pipe._pack_latents(noisy, *noisy.shape)
                velocity_target = model.pipe._pack_latents(velocity_target, *velocity_target.shape)
            optimizer.zero_grad(set_to_none=True)
            condition = model.text.condition(entry, "detect")
            prediction = model.velocity(noisy.to(model.dtype), sigma * 1000, condition, model.config["sampling"])
            ti_loss = torch.nn.functional.mse_loss(prediction.float(), velocity_target.float())
            loss = ti_loss
            metrics = {"ti_loss": ti_loss.item()} if tracker else None
            if joint:
                # No repeated text within a batch; later batches sample independently.
                text_indices = torch.cat((
                    positive_indices[torch.randperm(len(positive_indices), generator=text_generator)[:per_class]],
                    negative_indices[torch.randperm(len(negative_indices), generator=text_generator)[:per_class]]))
                batch = tuple(value[text_indices].to(model.text.device) for value in features)
                truth = labels[text_indices].to(model.text.device)
                prototype = model.text.split_features(condition[1])
                scores = cosine_scores(batch, prototype).squeeze(1)
                if args.detect_loss == "ti-flow-bce":
                    cls_loss = classification_loss(scores, truth, args.tau, args.temperature)
                else:
                    cls_loss = supervised_contrastive_loss(scores, truth, args.contrastive_temperature)
                loss = ti_loss + args.classification_weight * cls_loss
                if tracker:
                    metrics.update(classification_loss=cls_loss.item(), classification_batch_size=len(text_indices),
                                   accuracy=((scores >= args.tau) == truth.bool()).float().mean().item(),
                                   positive_cosine=scores[truth == 1].mean().item(),
                                   negative_cosine=scores[truth == 0].mean().item())
            loss.backward()
            optimizer.step()
            step += 1
            if tracker:
                tracker.log_step(step, loss, optimizer, parameters, len(indices),
                                 timestep=float(sigma.mean() * 1000), metrics=metrics)
            if step % 50 == 0:
                print(f"[{target}:ti-flow] epoch={epoch} update={step} loss={loss.item():.6f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                if tracker:
                    tracker.end_epoch(len(clean))
                return step
        if tracker:
            tracker.end_epoch(len(clean))
    return step


def train_erase(model, entry, rows, parameters, args, tracker, target):
    """Fit safe velocities with all branches evaluated at one shared latent."""
    sampling = model.config["sampling"]
    scale = sampling["true_cfg_scale"] if model.flux else sampling["guidance_scale"]
    positive = encode_dataset(model, [row["target_prompt"] for row in rows], args.batch_size)
    anchors = encode_dataset(model, [row["anchor_prompt"] for row in rows], args.batch_size)
    null = model.encode([""])
    image_noise = args.erase_state == "image-noise"
    if image_noise:
        clean, manifest = erase_images(model, entry, args, target)
        entry.update(erase_image_manifest=manifest, erase_image_count=len(clean))
    updates_per_pair = 1 if image_noise else sampling["steps"]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    for epoch in range(1, args.epochs + 1):
        if tracker:
            tracker.start_epoch(target, "erase", epoch)
        for indices in torch.randperm(len(rows), generator=generator).split(args.batch_size):
            positive_batch = tuple(value[indices] for value in positive)
            anchor_batch = tuple(value[indices] for value in anchors)
            if image_noise:
                image_indices = torch.randint(len(clean), (len(indices),), generator=generator)
                z0 = clean[image_indices].to(model.device)
                noise = torch.randn(z0.shape, generator=generator).to(model.device)
                sigma = torch.rand((len(indices),), generator=generator).to(model.device)
                amount = sigma[:, None, None, None]
                latents = (1 - amount) * z0 + amount * noise
                if model.flux:
                    latents = model.pipe._pack_latents(latents, *latents.shape)
                latents = latents.to(model.dtype)
                timesteps = (sigma * 1000,)
            else:
                seeds = torch.randint(0, 2 ** 31, (len(indices),), generator=generator).tolist()
                latents = model.latents(seeds, sampling)
                timesteps = model.timesteps(latents, sampling)
            for timestep in timesteps:
                with torch.no_grad():
                    v_positive = model.velocity(latents, timestep, positive_batch, sampling).float()
                    v_null = model.velocity(latents, timestep, null, sampling).float()
                    v_anchor = model.velocity(latents, timestep, anchor_batch, sampling).float()
                    unsafe_cfg = v_null + scale * (v_positive - v_null)
                    safe_cfg = v_null + scale * (v_anchor - v_null)
                    teacher = safe_cfg if args.erase_target == "cfg" else v_anchor
                optimizer.zero_grad(set_to_none=True)
                condition = model.text.condition(entry, "erase")
                v_erase = model.velocity(latents, timestep, condition, sampling).float()
                guided = v_erase + scale * (v_positive - v_erase)
                loss = torch.nn.functional.mse_loss(guided, teacher)
                if not image_noise:
                    # Save the current velocity before updating the shared condition.
                    rollout = {"unsafe": unsafe_cfg, "safe": safe_cfg,
                               "student": guided}[args.erase_trajectory].detach()
                loss.backward()
                optimizer.step()
                step += 1
                if tracker:
                    tracker.log_step(step, loss, optimizer, parameters, len(indices), timestep=timestep.float().mean().item())
                if step % 50 == 0:
                    print(f"[{target}:erase] epoch={epoch} update={step} loss={loss.item():.6f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    if tracker:
                        tracker.end_epoch(len(rows) * updates_per_pair)
                    return step
                if not image_noise:
                    with torch.no_grad():
                        latents = model.pipe.scheduler.step(rollout, timestep, latents, return_dict=False)[0]
                        latents = latents.detach().to(model.dtype)
        if tracker:
            tracker.end_epoch(len(rows) * updates_per_pair)
    return step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DIT_MODELS)
    parser.add_argument("--base-model", help="Local Diffusers directory or Hub ID")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--targets", nargs="+", help='Optional subset, e.g. --targets nudity "Van Gogh"')
    parser.add_argument("--stage", choices=("all", "detect", "erase"), default="all")
    parser.add_argument("--resume", action="store_true", help="Continue selected concept files in --run-dir with a fresh optimizer")
    parser.add_argument("--detect-loss", choices=("clip", "ti-flow", "ti-flow-bce", "ti-flow-supcon"),
                        help="Required for detect training; no default")
    parser.add_argument("--prompt-space", choices=("condition", "soft", "hidden"),
                        help="Default: condition for detect, hidden for erase; resume uses saved spaces")
    parser.add_argument("--erase-target", choices=("conditional", "cfg"),
                        help="Safe teacher velocity: raw conditional or CFG; default cfg, resume uses saved setting")
    parser.add_argument("--erase-state", choices=("trajectory", "image-noise"),
                        help="Training states: sampling trajectory or randomly noised concept images; default trajectory")
    parser.add_argument("--erase-trajectory", choices=("unsafe", "safe", "student"),
                        help="For --erase-state trajectory: velocity advancing the shared latent; default unsafe")
    parser.add_argument("--erase-image-count", type=int, help="Images per concept for image-noise training; default 10")
    parser.add_argument("--erase-image-dir", type=Path, help="Concept-image cache; default dataset/erase/concept_images")
    parser.add_argument("--erase-image-generation-steps", type=int, help="Concept-image generation steps; default sampling steps")
    parser.add_argument("--erase-reuse-images", action=argparse.BooleanOptionalAction, default=None,
                        help="Reuse compatible concept images; --no-erase-reuse-images generates a new random-seed batch")
    parser.add_argument("--soft-template", help="One {} placeholder; default: {}")
    parser.add_argument("--clip-projection-model", help="FLUX contrastive CLIP head; default: openai/clip-vit-large-patch14")
    parser.add_argument("--ti-image-count", type=int, help="Positive images per concept; default: all positive rows")
    parser.add_argument("--ti-generation-steps", type=int, help="Base-model steps for cached TI images; default: sampling steps")
    parser.add_argument("--ti-seed", type=int, help="TI image seeds start here; default: seed")
    # training hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", "--steps", type=int, default=2000, help="Optimizer updates per target/stage; 0 disables the cap")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--classification-batch-size", type=int, help="Joint loss: equal positive/negative counts; default 16")
    parser.add_argument("--classification-weight", type=float, help="Joint classification loss weight; default 1")
    parser.add_argument("--contrastive-temperature", type=float, help="SupCon temperature; default 0.1")
    parser.add_argument("--tau", type=float, help="Default: 0.2, or the saved training setting")
    parser.add_argument("--temperature", type=float, help="Default: 0.1, or the saved training setting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-prompts", type=int, help="Use first N rows of each training stage for a quick run")
    parser.add_argument("--dataset-dir", type=Path, default=HERE / "dataset/detect", help="Detect JSON directory")
    parser.add_argument("--erase-dir", type=Path, default=HERE / "dataset/erase", help="Erase JSON directory")
    parser.add_argument("--run-dir", type=Path, help="Exact experiment directory shared by both training stages and sampling")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results", help="Parent for a new model-time experiment directory")
    parser.add_argument("--sampling-steps", type=int, help="Denoising steps in each teacher trajectory")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--true-cfg-scale", type=float)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--device-map", choices=("balanced", "auto"))
    parser.add_argument("--max-memory", type=json.loads)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--track", action="store_true", help="Record SwanLab curves, loss.csv and epoch/total timings")
    parser.add_argument("--swanlab-mode", choices=("local", "offline", "online"), default="local")
    parser.add_argument("--swanlab-project", default="detect-erase")
    parser.add_argument("--swanlab-name")
    args = parser.parse_args(argv)
    requested_trajectory = args.erase_trajectory

    stages = ("detect", "erase") if args.stage == "all" else (args.stage,)
    if "detect" in stages and args.detect_loss is None:
        parser.error("Detect training requires an explicit --detect-loss; choose ti-flow-bce or ti-flow-supcon for joint training.")
    if args.track and args.swanlab_mode == "local":
        from importlib.util import find_spec
        if find_spec("swanboard") is None:
            parser.error("Local SwanLab requires: pip install 'swanlab[dashboard]'; or use --swanlab-mode offline.")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir.")

    directory = args.run_dir.expanduser().resolve() if args.run_dir else None
    info = read_json(directory / "run.json") if directory and (directory / "run.json").exists() else {}
    saved_detect = directory / "train_detect/training.json" if directory else None
    defaults = read_json(saved_detect) if saved_detect and saved_detect.exists() else {}
    stage_defaults = {}
    for stage in stages:
        path = directory / f"train_{stage}/training.json" if directory else None
        stage_defaults[stage] = read_json(path) if path and path.exists() else {}
    spaces = {stage: args.prompt_space or stage_defaults[stage].get(
        "prompt_space", "condition" if stage == "detect" else "hidden") for stage in stages}
    templates = {stage: ("a photo of {}" if spaces[stage] == "condition" else
                         args.soft_template or stage_defaults[stage].get("soft_template", "{}")) for stage in stages}
    args.tau = defaults.get("tau", 0.2) if args.tau is None else args.tau
    args.temperature = defaults.get("temperature", 0.1) if args.temperature is None else args.temperature
    for key, default in (("classification_batch_size", 16), ("classification_weight", 1.0),
                         ("contrastive_temperature", 0.1)):
        if getattr(args, key) is None:
            setattr(args, key, defaults.get(key, default))
    erase_defaults = stage_defaults.get("erase", {})
    for key, default in (("erase_target", "cfg"), ("erase_state", "trajectory"),
                         ("erase_trajectory", "unsafe"), ("erase_image_count", 10), ("erase_reuse_images", True)):
        if getattr(args, key) is None:
            # Older erase runs fitted the raw conditional safe velocity.
            if key == "erase_target" and args.resume and erase_defaults:
                default = "conditional"
            saved = erase_defaults.get(key)
            setattr(args, key, default if saved is None else saved)
    args.erase_image_dir = Path(args.erase_image_dir or erase_defaults.get("erase_image_dir")
                                or HERE / "dataset/erase/concept_images").expanduser().resolve()
    if args.erase_image_generation_steps is None:
        args.erase_image_generation_steps = erase_defaults.get("erase_image_generation_steps")
    if "erase" in stages and args.erase_state == "image-noise":
        if requested_trajectory is not None:
            parser.error("--erase-trajectory only applies to --erase-state trajectory; omit it for image-noise training.")
        args.erase_trajectory = None
    if args.erase_image_count <= 0 or (args.erase_image_generation_steps is not None
                                       and args.erase_image_generation_steps <= 0):
        parser.error("Erase image count and generation steps must be positive.")
    if (args.learning_rate <= 0 or args.epochs <= 0 or args.max_steps < 0 or args.batch_size <= 0
            or args.temperature <= 0 or not -1 <= args.tau <= 1
            or (args.limit_prompts is not None and args.limit_prompts <= 0)):
        parser.error("Use positive learning rate/epochs/batch size/temperature, max-steps >= 0 and tau in [-1, 1].")

    if any(template.count("{}") != 1 for template in templates.values()):
        parser.error("--soft-template must contain exactly one {} placeholder.")
    if spaces.get("erase") == "condition":
        parser.error("The condition space is for detect only; erase uses soft or hidden.")
    joint = "detect" in stages and args.detect_loss in {"ti-flow-bce", "ti-flow-supcon"}
    if joint and spaces["detect"] != "condition":
        parser.error("Joint detect training requires --prompt-space condition (the detect default).")
    if joint and (args.classification_batch_size < 2 or args.classification_batch_size % 2
                  or args.classification_weight <= 0 or args.contrastive_temperature <= 0):
        parser.error("Use an even classification-batch-size >= 2, positive classification weight and contrastive temperature.")
    if any(value is not None and value <= 0 for value in (args.ti_image_count, args.ti_generation_steps)):
        parser.error("TI image count and generation steps must be positive.")
    if args.ti_seed is not None and not 0 <= args.ti_seed < 2 ** 63 - 100000:
        parser.error("--ti-seed must be nonnegative and leave room for per-image seeds.")

    cases = definitions()["cases"]
    selected = select_cases(args.cases)
    wanted = list(dict.fromkeys((cases[case]["group"], target) for case in selected for target in cases[case]["targets"]))
    if args.targets:
        if set(args.targets) - {target for _, target in wanted}:
            parser.error("--targets must belong to the selected benchmark cases.")
        wanted = [(group, target) for group, target in wanted if target in args.targets]
    existing = {(stage, target): prompt_checkpoint(directory, stage, target)
                for stage in stages for _, target in wanted
                if directory and prompt_checkpoint(directory, stage, target).exists()}
    if existing and not args.resume:
        parser.error("Selected concept files already exist; start a new run, or use --resume to continue them.")
    resumed = {key: torch.load(path, map_location="cpu", weights_only=True) for key, path in existing.items()}
    if any(entry[f"{stage}_space"] != spaces[stage] or entry[f"{stage}_template"] != templates[stage]
           for (stage, target), entry in resumed.items()):
        parser.error("Resume requires the saved prompt space and soft template.")

    sampling = dict(device=args.device, batch_size=args.batch_size, local_files_only=args.offline,
                    device_map=args.device_map, max_memory=args.max_memory)
    for field in ("height", "width", "guidance_scale", "true_cfg_scale", "max_sequence_length", "dtype"):
        if getattr(args, field) is not None:
            sampling[field] = getattr(args, field)
    if args.sampling_steps is not None:
        sampling["steps"] = args.sampling_steps
    configs = info.get("config", {})
    previous = configs.get("erase" if args.stage == "all" else args.stage) or next(iter(configs.values()), {})
    model_name = args.model or info.get("model_name", "sd35_medium")
    if model_name.startswith("flux1_"):
        sampling.setdefault("true_cfg_scale", previous.get("sampling", {}).get("true_cfg_scale", 2.0))
    base = copy.deepcopy(previous)
    base.update(model_name=model_name, adapter={"kind": "original"})
    if args.base_model:
        base["base_model"] = args.base_model
    if previous and (model_name != previous["model_name"]
                     or args.base_model and args.base_model != previous["base_model"]
                     or any(getattr(args, key) is not None and getattr(args, key) != previous["sampling"][key]
                            for key in ("max_sequence_length", "dtype"))):
        parser.error("Concept files in one run require the same model, text length and dtype.")
    base["sampling"] = {**base.get("sampling", {}), **sampling}
    if args.clip_projection_model:
        base["text"] = {"clip_projection_model": args.clip_projection_model}
    config = configuration(task="i2p", base=base)
    scale = config["sampling"]["true_cfg_scale"] if model_name.startswith("flux1_") else config["sampling"]["guidance_scale"]
    if "erase" in stages and scale <= 1:
        parser.error("CFG prompt fitting requires guidance-scale > 1 (FLUX: true-cfg-scale > 1).")
    detect_data, erase_data = {}, {}
    for group in dict.fromkeys(group for group, _ in wanted):
        if "detect" in stages:
            detect_data.update({data["target"]: data for data in read_json(args.dataset_dir / f"{group}.json")["datasets"]})
        if "erase" in stages:
            erase_data.update({data["target"]: data for data in read_json(args.erase_dir / f"{group}.json")["datasets"]})
    detect_rows = {}
    if "detect" in stages:
        for _, target in wanted:
            rows = detect_data[target]["samples"]
            if args.detect_loss == "ti-flow":
                rows = [row for row in rows if row["label"] == 1][:args.ti_image_count]
            rows = rows[:args.limit_prompts]
            if not rows:
                parser.error(f"No detect training samples for {target}.")
            if joint and any(len({row["prompt"] for row in rows if row["label"] == label}) < args.classification_batch_size // 2
                             for label in (0, 1)):
                parser.error(f"{target} needs at least {args.classification_batch_size // 2} texts per class "
                             "for sampling without replacement within a batch.")
            detect_rows[target] = rows

    directory = run_directory(model_name, directory, args.output_dir)
    config.update(output_dir=str(directory), result_layout="flat")
    for stage in stages:
        stage_space, stage_template = spaces[stage], templates[stage]
        stage_dir = directory / f"train_{stage}"
        stage_dir.mkdir(exist_ok=True)
        write_json(stage_dir / "training.json", {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "stage": stage, "run_dir": str(directory),
            "prompt_space": stage_space, "soft_template": stage_template})
        with run_log(directory, f"train_{stage}"):
            print(f"Run: {directory}\nStage: {stage}", flush=True)
            tracker = TrainingTrace(args, stage_dir, config) if args.track else None
            diffusion_needed = stage == "erase" or args.detect_loss in {"ti-flow", "ti-flow-bce", "ti-flow-supcon"}
            model = PromptModel(config) if diffusion_needed else TextEncoder(config)
            text = model.text if diffusion_needed else model
            for tokenizer in text.tokenizers:
                fixed_length = sum(len(tokenizer.encode(part, add_special_tokens=False))
                                   for part in stage_template.split("{}"))
                if stage_space == "soft" and fixed_length + 3 > tokenizer.model_max_length:
                    parser.error("--soft-template must leave room for BOS, one soft token and EOS.")
            if tracker:
                tracker.devices = cuda_devices(model, config["sampling"])
            if args.gradient_checkpointing:
                for encoder in text.encoders:
                    encoder.gradient_checkpointing_enable()
                if diffusion_needed:
                    model.pipe.transformer.enable_gradient_checkpointing()
            update_run(directory, model_name, config={stage: model.config})
            for group, target in wanted:
                checkpoint = prompt_checkpoint(directory, stage, target)
                if (stage, target) in existing:
                    entry = resumed[stage, target]
                else:
                    entry = {"schema": "detect-erase-concept-v3" if stage_space == "condition" else "detect-erase-concept-v2",
                             "target": target, "stage": stage, "group": group}
                    text.initialize(entry, stage, stage_space, stage_template)
                parameters = text.parameters(entry, stage)
                if stage == "detect":
                    rows = detect_rows[target]
                    if args.detect_loss == "clip":
                        updates = train_clip(text, entry, rows, parameters, args, tracker, target)
                    else:
                        updates = train_ti_flow(model, entry, rows, parameters, args, tracker, target, stage_dir)
                else:
                    rows = erase_data[target]["pairs"][:args.limit_prompts]
                    updates = train_erase(model, entry, rows, parameters, args, tracker, target)
                for key, value in list(entry.items()):
                    if isinstance(value, torch.Tensor):
                        entry[key] = value.detach().float().cpu()
                    elif key.endswith(("_soft_tokens", "_clip_conditions")):
                        entry[key] = [token.detach().float().cpu() for token in value]
                entry[f"{stage}_updates"] = updates
                entry[f"{stage}_samples"] = len(rows)
                if stage == "detect":
                    entry.update(detect_loss=args.detect_loss, tau=args.tau, temperature=args.temperature,
                                 classification_weight=args.classification_weight,
                                 classification_batch_size=args.classification_batch_size,
                                 contrastive_temperature=args.contrastive_temperature)
                else:
                    entry.update(cfg_scale=scale, erase_target=args.erase_target,
                                 erase_state=args.erase_state, erase_trajectory=args.erase_trajectory)
                torch.save(entry, checkpoint)
                update_run(directory, model_name, training={stage: {target: str(checkpoint)}})
                print(f"[{target}:{stage}] completed {updates} updates; checkpoint: {checkpoint}", flush=True)
            model.close()
            if tracker:
                tracker.finish()
    print(f"Run: {directory}", flush=True)


if __name__ == "__main__":
    main()
