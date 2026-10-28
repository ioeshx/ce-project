"""Joint ESD training: upstream FLUX step and an SD3 flow-matching adapter."""
from collections import Counter
import copy
import json
from pathlib import Path
import random

import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps

from benchmark.adapters import release_memory
from benchmark.execution import runtime_device
from benchmark.common import file_digest, read_json, write_json
from utils.esd_trainer import ESDConfig, FluxESDAdapter, StepResult, make_sampling_generator, run_esd_training

FORMAT = "esd-dit-benchmark-v1"


def sd3_prediction(transformer, latents, timestep, embeddings, pooled):
    return transformer(hidden_states=latents, timestep=timestep.expand(latents.shape[0]),
                       encoder_hidden_states=embeddings, pooled_projections=pooled, return_dict=False)[0]


class JointDiTAdapter(FluxESDAdapter):
    """Reuse the native parameter selector, teacher swaps, Adam and MSE loop."""

    def __init__(self, settings, offline=False):
        self.settings, self.offline = settings, offline
        self.text_device = runtime_device(settings["text_device"])
        self.flux = settings["model_name"] == "flux1_dev"
        self.family = "flux" if self.flux else "sd3"
        self.counts = Counter({target: 0 for target in settings["targets"]})
        self.order = []
        self.rng = random.Random(settings["training_seed"])

    def load_pipeline(self, config):
        cls = FluxPipeline if self.flux else StableDiffusion3Pipeline
        pipe = cls.from_pretrained(config.base_model_id, vae=None, torch_dtype=config.torch_dtype,
                                   revision=self.settings["model_revision"], local_files_only=self.offline)
        # Keep encoders on the text device; training only places the denoiser on
        # the training GPU. No VAE or second full teacher model is allocated.
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(pipe, name, None)
            if encoder is not None:
                encoder.requires_grad_(False).to(self.text_device)
        pipe.transformer.to(config.device)
        return pipe

    def prepare_context(self, pipe, config):
        prompts = self.settings["targets"] + [""]
        kwargs = dict(prompt=prompts, prompt_2=None, device=self.text_device,
                      num_images_per_prompt=1, max_sequence_length=config.max_sequence_length)
        if not self.flux:
            kwargs.update(prompt_3=None, do_classifier_free_guidance=False)
        with torch.no_grad():
            encoded = pipe.encode_prompt(**kwargs)
        embeddings = encoded[0].to(config.device)
        pooled = encoded[1 if self.flux else 2].to(config.device)
        contexts = []
        for index in range(len(self.settings["targets"])):
            erase, erase_pooled = embeddings[index:index + 1], pooled[index:index + 1]
            context = {"resolution": config.resolution,
                       "erase_prompt_embeds": erase, "erase_pooled_prompt_embeds": erase_pooled,
                       "null_prompt_embeds": embeddings[-1:], "null_pooled_prompt_embeds": pooled[-1:],
                       "erase_from_prompt_embeds": None, "erase_from_pooled_prompt_embeds": None,
                       "sample_prompt_embeds": erase, "sample_pooled_prompt_embeds": erase_pooled,
                       "student_prompt_embeds": erase, "student_pooled_prompt_embeds": erase_pooled}
            if self.flux:
                context["text_ids"] = encoded[2].to(config.device)
            contexts.append(context)
        for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            if getattr(pipe, name, None) is not None:
                pipe.register_modules(**{name: None})
        release_memory()
        return {"target_contexts": contexts}

    def training_step(self, pipe, prepared, context, config):
        if not self.order:
            self.order = list(range(len(self.settings["targets"])))
            self.rng.shuffle(self.order)
        index = self.order.pop()
        selected = context["target_contexts"][index]
        result = (super().training_step(pipe, prepared, selected, config) if self.flux
                  else self.sd3_step(pipe, prepared, selected, config))
        target = self.settings["targets"][index]
        self.counts[target] += 1
        result.metrics["concept"] = target
        return result

    def sd3_step(self, pipe, prepared, context, config):
        """Original ESD target on SD3's native velocity and timestep convention."""
        index = random.randint(0, config.num_inference_steps - 1)
        seed = random.randint(0, 2**15)
        model = prepared.component
        prepared.use_base()
        model.eval()
        with torch.no_grad():
            side = config.resolution // pipe.vae_scale_factor
            xt = torch.randn((1, model.config.in_channels, side, side), dtype=config.torch_dtype,
                             device=config.device, generator=make_sampling_generator(config.device, seed))
            scheduler_kwargs = {}
            if pipe.scheduler.config.get("use_dynamic_shifting", False):
                image_seq_len = (side // model.config.patch_size) ** 2
                scheduler_kwargs["mu"] = calculate_shift(
                    image_seq_len, pipe.scheduler.config.get("base_image_seq_len", 256),
                    pipe.scheduler.config.get("max_image_seq_len", 4096),
                    pipe.scheduler.config.get("base_shift", 0.5), pipe.scheduler.config.get("max_shift", 1.16))
            timesteps, _ = retrieve_timesteps(pipe.scheduler, config.num_inference_steps,
                                              config.device, **scheduler_kwargs)
            pipe.scheduler.set_begin_index(0)
            embeddings = torch.cat([context["null_prompt_embeds"], context["erase_prompt_embeds"]])
            pooled = torch.cat([context["null_pooled_prompt_embeds"], context["erase_pooled_prompt_embeds"]])
            # Stop BEFORE the selected timestep, just like upstream FLUX ESD.
            # This is the native SD3 CFG Euler rollout; SD3 does not divide t by 1000.
            for timestep in timesteps[:index]:
                null, erase = sd3_prediction(model, torch.cat([xt, xt]), timestep, embeddings, pooled).chunk(2)
                guided = (null + config.inference_guidance_scale * (erase - null)
                          if config.inference_guidance_scale > 1 else erase)
                xt = pipe.scheduler.step(guided, timestep, xt, return_dict=False)[0]
            timestep = timesteps[index]
            null, erase = sd3_prediction(model, torch.cat([xt, xt]), timestep, embeddings, pooled).chunk(2)
            target = erase - config.negative_guidance * (erase - null)

        prepared.use_student()
        model.train()
        prediction = sd3_prediction(model, xt, timestep, context["student_prompt_embeds"],
                                    context["student_pooled_prompt_embeds"])
        return StepResult(model_pred=prediction, target=target, timestep_index=index)

    def build_checkpoint_path(self, config):
        return str(Path(config.save_path) / "esd.safetensors")

    def build_metadata(self, config):
        metadata = super().build_metadata(config)
        metadata.update(erase_concept=json.dumps(self.settings["targets"]),
                        target_counts=json.dumps(dict(self.counts)),
                        target_selection="shuffled cycles; one target per update")
        return metadata


def train(checkpoint, settings, *, offline=False, cpu_threads=4):
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    weights_path = checkpoint / "esd.safetensors"
    if checkpoint.exists():
        info = read_json(checkpoint / "metadata.json")
        if info["settings"] != settings or info["weights_sha256"] != file_digest(weights_path):
            raise ValueError(f"Checkpoint differs from this training request; choose another --run-root: {checkpoint}")
        print(f"[reuse training] {checkpoint}", flush=True)
        return
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(settings["training_seed"])
    random.seed(settings["training_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    rollout = settings["training_protocol"]
    config = ESDConfig(
        family="flux" if settings["model_name"] == "flux1_dev" else "sd3",
        base_model_id=settings["base_model"], erase_concept=settings["targets"][0], erase_from=None,
        train_method=settings["train_method"], iterations=settings["iterations"], lr=settings["lr"],
        negative_guidance=settings["negative_guidance"], num_inference_steps=rollout["steps"],
        guidance_scale=settings["training_guidance"], inference_guidance_scale=rollout["guidance_scale"],
        batch_size=1, resolution=rollout["width"], save_path=str(checkpoint), device=runtime_device(settings["device"]),
        torch_dtype=getattr(torch, rollout["dtype"]), max_sequence_length=rollout["max_sequence_length"],
        gradient_checkpointing=settings["gradient_checkpointing"])
    adapter = JointDiTAdapter(settings, offline=offline)
    run_esd_training(config, adapter=adapter)
    weights = load_file(str(weights_path))
    if not all(torch.isfinite(value).all() for value in weights.values()):
        raise RuntimeError("ESD produced non-finite weights; this run cannot be evaluated.")
    write_json(checkpoint / "metadata.json", {"format": FORMAT, "settings": settings,
               "target_counts": dict(adapter.counts), "trainable_parameters": list(weights),
               "weights_sha256": file_digest(weights_path)})
    release_memory()


def create(config):
    """Apply the native partial transformer checkpoint to the shared sampler."""
    from safetensors.torch import load_file
    from benchmark.dit_adapter import DiTAdapter

    checkpoint = Path(config["adapter"]["checkpoint"])
    info = read_json(checkpoint / "metadata.json")
    settings = info["settings"]
    if (info["format"] != FORMAT or settings != config["erasure_parameters"]
            or settings["base_model"] != config["base_model"] or settings["targets"] != config["targets"]
            or settings["model_revision"] != config["sampling"].get("model_revision")
            or info["weights_sha256"] != file_digest(checkpoint / "esd.safetensors")):
        raise ValueError("ESD checkpoint does not match this DiT experiment.")
    native_config = copy.deepcopy(config)
    native_config["adapter"] = {"kind": "original"}
    adapter = DiTAdapter(native_config)
    adapter.pipe.transformer.load_state_dict(load_file(str(checkpoint / "esd.safetensors")), strict=False)
    return adapter
