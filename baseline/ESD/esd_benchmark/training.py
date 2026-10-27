"""Joint target contexts around the native SD adapter and optimizer loop."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from benchmark.common import file_digest, locked, read_json, write_json
from benchmark.layout import timestamp, unique_directory
from benchmark.sampling_profile import initial_noise, make_scheduler
from utils.esd_trainer import ESDConfig, StableDiffusionESDAdapter, run_esd_training


class TrainingPipeline(StableDiffusionPipeline):
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype,
                        device, generator, latents=None):
        shape = (num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if latents is None:
            generators = generator if isinstance(generator, list) else [generator]
            if len(generators) != batch_size or any(g is None for g in generators):
                raise ValueError("ESD rollout needs one explicit generator per record.")
            latents = torch.stack([initial_noise(shape, g.initial_seed(), dtype=dtype, device=device)
                                   for g in generators])
        else:
            if tuple(latents.shape) != (batch_size, *shape):
                raise ValueError("Invalid ESD rollout latent shape.")
            latents = latents.to(device=device, dtype=dtype)
        return latents * self.scheduler.init_noise_sigma


class JointSDAdapter(StableDiffusionESDAdapter):
    def __init__(self, settings, *, offline=False):
        self.settings, self.offline = settings, offline
        self.targets = list(settings["targets"])
        self.counts = Counter({t: 0 for t in self.targets})
        self.order = []
        self.selection_rng = random.Random(settings["training_seed"])
        self.parameter_names = []

    def load_pipeline(self, config):
        pipe = TrainingPipeline.from_pretrained(
            config.base_model_id, torch_dtype=config.torch_dtype, use_safetensors=True,
            revision=self.settings["model_revision"], local_files_only=self.offline,
            safety_checker=None, requires_safety_checker=False)
        pipe.scheduler = make_scheduler(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        pipe.to(config.device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        return pipe

    def create_prepared_component(self, pipe, train_method, config):
        prepared = super().create_prepared_component(pipe, train_method, config)
        self.parameter_names = list(prepared.student_params)
        return prepared

    def prepare_context(self, pipe, config):
        contexts = []
        for target in self.targets:
            # The native method offloads this encoder after preparing a context.
            pipe.text_encoder.to(config.device)
            contexts.append(super().prepare_context(pipe, replace(config, erase_concept=target)))
        return {"target_contexts": contexts}

    def training_step(self, pipe, prepared, context, config):
        if not self.order:
            self.order = list(range(len(self.targets)))
            self.selection_rng.shuffle(self.order)
        index = self.order.pop()
        result = super().training_step(pipe, prepared, context["target_contexts"][index], config)
        self.counts[self.targets[index]] += 1
        result.metrics["concept"] = self.targets[index]
        return result

    def build_checkpoint_path(self, config):
        return str(Path(config.save_path) / "esd.safetensors")

    def build_metadata(self, config):
        metadata = super().build_metadata(config)
        metadata.update(erase_concept=json.dumps(self.targets),
                        training_protocol=json.dumps(self.settings["training_protocol"], sort_keys=True),
                        training_exception=self.settings["training_exception"],
                        target_counts=json.dumps(dict(self.counts), sort_keys=True))
        return metadata


def train(checkpoint, settings, *, offline=False, cpu_threads=4):
    from safetensors.torch import load_file
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(settings["training_seed"])
    np.random.seed(settings["training_seed"])
    random.seed(settings["training_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with locked(checkpoint.parent / ".training.lock"):
        if checkpoint.exists():
            info = read_json(checkpoint / "metadata.json")
            if (info["settings"] != settings
                    or file_digest(checkpoint / "esd.safetensors") != info["weights_sha256"]):
                raise ValueError("ESD settings/source/weights changed; choose another --run-root.")
            print(f"[resume] {checkpoint}", flush=True)
            return checkpoint
        pending = unique_directory(checkpoint.parent, f".checkpoint-pending-{timestamp()}")
        try:
            rollout = settings["training_protocol"]
            config = ESDConfig(family="sd", base_model_id=settings["base_model"],
                erase_concept=settings["targets"][0], erase_from=None,
                train_method=settings["train_method"], iterations=settings["iterations"],
                lr=settings["lr"], negative_guidance=settings["negative_guidance"],
                num_inference_steps=rollout["steps"], guidance_scale=rollout["guidance_scale"],
                batch_size=1, resolution=rollout["width"], save_path=str(pending),
                device=settings["device"], torch_dtype=getattr(torch, rollout["dtype"]))
            adapter = JointSDAdapter(settings, offline=offline)
            path = run_esd_training(config, adapter=adapter)
            weights = load_file(path, device="cpu")
            if (not adapter.parameter_names or set(weights) != set(adapter.parameter_names)
                    or not all(torch.isfinite(w).all() for w in weights.values())
                    or min(adapter.counts.values()) < 1 or sum(adapter.counts.values()) != config.iterations):
                raise ValueError("Incomplete/non-finite ESD checkpoint or uncovered targets.")
            write_json(pending / "metadata.json", {
                "format": "esd-sd14-benchmark-v1", "base_model": settings["base_model"],
                "task": settings["task"], "targets": settings["targets"], "settings": settings,
                "trainable_parameters": adapter.parameter_names, "target_counts": dict(adapter.counts),
                "weights_sha256": file_digest(path), "created_at": timestamp()})
            pending.rename(checkpoint)
            print(f"[checkpoint] {checkpoint}", flush=True)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
    return checkpoint
