"""Learn CLIP soft tokens (or hidden conditions) for detection and negative CFG."""
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import argparse
import copy
import csv
import json
from pathlib import Path
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
from DiT.detect_erase.prompts import PromptModel, cfg_loss, classification_loss
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
            "gradient_norm", "parameter_norm", "batch_size", "accuracy", "positive_cosine", "negative_cosine"])
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


def ti_images(model, rows, args, stage_dir, target):
    """Generate one fixed base-model image per positive prompt, and reuse the cache."""
    import numpy as np
    from PIL import Image
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
    latents = []
    vae = model.pipe.vae
    for start in range(0, len(records), args.batch_size):
        pixels = []
        for record in records[start:start + args.batch_size]:
            with Image.open(record["image"]) as image:
                pixels.append(torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1)
        with torch.no_grad():
            z = vae.encode(torch.stack(pixels).to(device=vae.device, dtype=vae.dtype)).latent_dist.mode()
            z = (z - (vae.config.shift_factor or 0)) * vae.config.scaling_factor
        latents.append(z.float().cpu())
    return torch.cat(latents)


def train_ti_flow(model, entry, rows, parameters, args, tracker, target, stage_dir):
    clean = ti_images(model, rows, args, stage_dir, target)
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
            loss = torch.nn.functional.mse_loss(prediction.float(), velocity_target.float())
            loss.backward()
            optimizer.step()
            step += 1
            if tracker:
                tracker.log_step(step, loss, optimizer, parameters, len(indices), timestep=float(sigma.mean() * 1000))
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
    """Fit CFG at each state of a detached base-model trajectory."""
    sampling = model.config["sampling"]
    scale = sampling["true_cfg_scale"] if model.flux else sampling["guidance_scale"]
    positive = encode_dataset(model, [row["target_prompt"] for row in rows], args.batch_size)
    anchors = encode_dataset(model, [row["anchor_prompt"] for row in rows], args.batch_size)
    null = model.encode([""])
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    for epoch in range(1, args.epochs + 1):
        if tracker:
            tracker.start_epoch(target, "erase", epoch)
        for indices in torch.randperm(len(rows), generator=generator).split(args.batch_size):
            positive_batch = tuple(value[indices] for value in positive)
            anchor_batch = tuple(value[indices] for value in anchors)
            seeds = torch.randint(0, 2 ** 31, (len(indices),), generator=generator).tolist()
            latents = model.latents(seeds, sampling)
            for timestep in model.timesteps(latents, sampling):
                with torch.no_grad():
                    v_positive = model.velocity(latents, timestep, positive_batch, sampling)
                    v_null = model.velocity(latents, timestep, null, sampling)
                    teacher_step = v_null + scale * (v_positive - v_null)
                    v_anchor = model.velocity(latents, timestep, anchor_batch, sampling)
                optimizer.zero_grad(set_to_none=True)
                condition = model.text.condition(entry, "erase")
                prediction = model.velocity(latents.detach(), timestep, condition, sampling)
                loss = cfg_loss(v_positive, v_anchor, prediction, scale)
                loss.backward()
                optimizer.step()
                step += 1
                if tracker:
                    tracker.log_step(step, loss, optimizer, parameters, len(indices), timestep=float(timestep))
                if step % 50 == 0:
                    print(f"[{target}:erase] epoch={epoch} update={step} loss={loss.item():.6f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    if tracker:
                        tracker.end_epoch(len(rows) * sampling["steps"])
                    return step
                with torch.no_grad():
                    latents = model.pipe.scheduler.step(teacher_step, timestep, latents, return_dict=False)[0]
                    latents = latents.detach().to(model.dtype)
        if tracker:
            tracker.end_epoch(len(rows) * sampling["steps"])
    return step


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=DIT_MODELS)
    parser.add_argument("--base-model", help="Local Diffusers directory or Hub ID")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--targets", nargs="+", help='Optional subset, e.g. --targets nudity "Van Gogh"')
    parser.add_argument("--stage", choices=("all", "detect", "erase"), default="all")
    parser.add_argument("--resume", action="store_true", help="Continue selected concept files in --run-dir with a fresh optimizer")
    parser.add_argument("--detect-loss", choices=("clip", "ti-flow"), help="Default: clip, or the saved training setting")
    parser.add_argument("--prompt-space", choices=("soft", "hidden"), help="Default: soft; hidden learns encoder outputs")
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
    args.detect_loss = args.detect_loss or defaults.get("detect_loss", "clip")
    stage_settings = directory / f"train_{args.stage}/training.json" if directory else None
    stage_defaults = read_json(stage_settings) if stage_settings and stage_settings.exists() else {}
    args.prompt_space = args.prompt_space or stage_defaults.get("prompt_space", "soft")
    args.soft_template = args.soft_template or stage_defaults.get("soft_template", "{}")
    args.tau = defaults.get("tau", 0.2) if args.tau is None else args.tau
    args.temperature = defaults.get("temperature", 0.1) if args.temperature is None else args.temperature
    if (args.learning_rate <= 0 or args.epochs <= 0 or args.max_steps < 0 or args.batch_size <= 0
            or args.temperature <= 0 or not -1 <= args.tau <= 1
            or (args.limit_prompts is not None and args.limit_prompts <= 0)):
        parser.error("Use positive learning rate/epochs/batch size/temperature, max-steps >= 0 and tau in [-1, 1].")

    if args.soft_template.count("{}") != 1:
        parser.error("--soft-template must contain exactly one {} placeholder.")
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
    stages = ("detect", "erase") if args.stage == "all" else (args.stage,)
    existing = {(stage, target): prompt_checkpoint(directory, stage, target)
                for stage in stages for _, target in wanted
                if directory and prompt_checkpoint(directory, stage, target).exists()}
    if existing and not args.resume:
        parser.error("Selected concept files already exist; start a new run, or use --resume to continue them.")
    resumed = {key: torch.load(path, map_location="cpu", weights_only=True) for key, path in existing.items()}
    if any(entry[f"{stage}_space"] != args.prompt_space or entry[f"{stage}_template"] != args.soft_template
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

    directory = run_directory(model_name, directory, args.output_dir)
    config.update(output_dir=str(directory), result_layout="flat")
    for stage in stages:
        stage_dir = directory / f"train_{stage}"
        stage_dir.mkdir(exist_ok=True)
        write_json(stage_dir / "training.json", {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "stage": stage, "run_dir": str(directory)})
        with run_log(directory, f"train_{stage}"):
            print(f"Run: {directory}\nStage: {stage}", flush=True)
            tracker = TrainingTrace(args, stage_dir, config) if args.track else None
            diffusion_needed = stage == "erase" or args.detect_loss == "ti-flow"
            model = PromptModel(config) if diffusion_needed else TextEncoder(config)
            text = model.text if diffusion_needed else model
            for tokenizer in text.tokenizers:
                fixed_length = sum(len(tokenizer.encode(part, add_special_tokens=False))
                                   for part in args.soft_template.split("{}"))
                if args.prompt_space == "soft" and fixed_length + 3 > tokenizer.model_max_length:
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
                    entry = {"schema": "detect-erase-concept-v2", "target": target, "stage": stage, "group": group}
                    text.initialize(entry, stage, args.prompt_space, args.soft_template)
                parameters = text.parameters(entry, stage)
                if stage == "detect":
                    rows = detect_data[target]["samples"]
                    if args.detect_loss == "ti-flow":
                        rows = [row for row in rows if row["label"] == 1][:args.ti_image_count]
                    rows = rows[:args.limit_prompts]
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
                    elif key.endswith("_soft_tokens"):
                        entry[key] = [token.detach().float().cpu() for token in value]
                entry[f"{stage}_updates"] = updates
                entry[f"{stage}_samples"] = len(rows)
                if stage == "detect":
                    entry.update(detect_loss=args.detect_loss, tau=args.tau, temperature=args.temperature)
                else:
                    entry["cfg_scale"] = scale
                torch.save(entry, checkpoint)
                update_run(directory, model_name, training={stage: {target: str(checkpoint)}})
                print(f"[{target}:{stage}] completed {updates} updates; checkpoint: {checkpoint}", flush=True)
            model.close()
            if tracker:
                tracker.finish()
    print(f"Run: {directory}", flush=True)


if __name__ == "__main__":
    main()
