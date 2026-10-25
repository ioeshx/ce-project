"""Native FLUX.1 and SD3 generation behind the shared image-record interface."""
from __future__ import annotations

from .adapters import load_state, release_memory
from .model_profiles import resolve_scheduler


class DiTAdapter:
    def __init__(self, config):
        import copy
        import torch
        from diffusers import FluxPipeline, StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

        self.config = copy.deepcopy(config)
        resolve_scheduler(self.config)
        opts, spec = self.config["sampling"], self.config["adapter"]
        if spec["kind"] not in {"original", "diffusers", "transformer", "lora"}:
            raise ValueError("DiT supports original, diffusers, transformer, lora, or a custom plugin.")
        self.flux = self.config["model_name"].startswith("flux1_")
        self.device = torch.device(opts["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; set sampling.device=cpu and use a suitable dtype.")
        self.dtype = getattr(torch, opts["dtype"] if opts["dtype"] != "auto" else
                             ("bfloat16" if self.device.type == "cuda" else "float32"))
        cls = FluxPipeline if self.flux else StableDiffusion3Pipeline
        source = spec.get("model_path") if spec["kind"] == "diffusers" else config["base_model"]
        self.pipe = cls.from_pretrained(source, torch_dtype=self.dtype,
                                       local_files_only=opts["local_files_only"], revision=opts.get("model_revision"))
        # An edited Diffusers directory must use the SAME original-model schedule.
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(opts["scheduler_config"])
        if spec["kind"] == "transformer":
            self.pipe.transformer.load_state_dict(load_state(spec["checkpoint"]), strict=True)
        elif spec["kind"] == "lora":
            self.pipe.load_lora_weights(spec["checkpoint"], local_files_only=opts["local_files_only"])
            self.pipe.fuse_lora(lora_scale=float(spec.get("scale", 1.0)))
        if opts.get("cpu_offload", False):
            if self.device.type != "cuda":
                raise ValueError("Model CPU offload needs a CUDA execution device.")
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)
        self.pipe.vae.enable_slicing()
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, records, sampling_config):
        import torch
        from .sampling_profile import initial_noise

        opts, pipe = sampling_config, self.pipe
        channels = pipe.transformer.config.in_channels // 4 if self.flux else pipe.transformer.config.in_channels
        shape = (channels, opts["height"] // pipe.vae_scale_factor, opts["width"] // pipe.vae_scale_factor)
        # Draw float32 on CPU for each record, then cast. Batch boundaries and
        # missing-image resume do not alter a record's initial latent.
        latents = torch.stack([initial_noise(shape, r["seed"], dtype=self.dtype, device="cpu") for r in records])
        if self.flux:
            latents = pipe._pack_latents(latents, len(records), *shape)
        kwargs = dict(prompt=[r["prompt"] for r in records], latents=latents.to(self.device),
                      generator=[torch.Generator("cpu").manual_seed(r["seed"]) for r in records],
                      height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                      guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"],
                      num_images_per_prompt=1)
        if self.flux:
            kwargs["true_cfg_scale"] = opts.get("true_cfg_scale", 1.0)
            if opts.get("negative_prompt") is not None:
                kwargs["negative_prompt"] = [opts["negative_prompt"]] * len(records)
        else:
            kwargs["negative_prompt"] = [opts.get("negative_prompt") or ""] * len(records)
        with torch.inference_mode():
            return pipe(**kwargs).images

    def close(self):
        if self.pipe is not None:
            self.pipe.maybe_free_model_hooks()
        self.pipe = None
        release_memory()
