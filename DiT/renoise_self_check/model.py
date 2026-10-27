"""SD3.5 / FLUX model calls; generation settings come from benchmark."""
from __future__ import annotations

import numpy as np
import torch

from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise


class FlowModel(DiTAdapter):
    """Reuse benchmark loading, placement and cleanup; expose velocity queries."""

    def __init__(self, config):
        super().__init__({**config, "adapter": {"kind": "original"}})
        self.options = self.config["sampling"]
        self.pipe.transformer.eval()
        self.conditions = {}
        self.forward_calls = self.sample_evaluations = 0
        self.user = self.negative = None
        self.image_ids = None
        self.cfg_scale = self.options.get("true_cfg_scale", 1) if self.flux else self.options["guidance_scale"]
        self.use_cfg = (self.cfg_scale > 1 and
                        (not self.flux or self.options.get("negative_prompt") is not None))

    def encode(self, texts):
        kwargs = dict(prompt=texts, prompt_2=None, device=self.device, num_images_per_prompt=1,
                      max_sequence_length=self.options["max_sequence_length"])
        if self.flux:
            tokens, pooled, ids = self.pipe.encode_prompt(**kwargs)
        else:
            tokens, _, pooled, _ = self.pipe.encode_prompt(
                **kwargs, prompt_3=None, do_classifier_free_guidance=False)
            ids = None
        return {"tokens": tokens, "pooled": pooled, "ids": ids}

    def condition(self, text, size):
        """Cache each reference prompt once, then expand it to the current batch."""
        if text not in self.conditions:
            self.conditions[text] = self.encode([text])
        c = self.conditions[text]
        return {"tokens": c["tokens"].expand(size, -1, -1),
                "pooled": c["pooled"].expand(size, -1), "ids": c["ids"]}

    def prepare(self, records):
        opts, pipe = self.options, self.pipe
        pipe._guidance_scale = opts["guidance_scale"]
        pipe._joint_attention_kwargs = {}
        pipe._interrupt = False
        if not self.flux:
            pipe._clip_skip = None
        self.user = self.encode([r["prompt"] for r in records])
        self.negative = self.condition(opts.get("negative_prompt") or "", len(records)) if self.use_cfg else None

        channels = pipe.transformer.config.in_channels // 4 if self.flux else pipe.transformer.config.in_channels
        shape = (channels, opts["height"] // pipe.vae_scale_factor, opts["width"] // pipe.vae_scale_factor)
        latents = torch.stack([initial_noise(shape, r["seed"], dtype=self.dtype, device="cpu") for r in records])
        if self.flux:
            latents = pipe._pack_latents(latents, len(records), *shape)
        args = (len(records), channels, opts["height"], opts["width"],
                self.user["tokens"].dtype, self.device, None, latents)

        # Use the native resolution-dependent time schedule of each pipeline.
        schedule = pipe.scheduler.config
        kwargs = {}
        if self.flux:
            from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
            latents, self.image_ids = pipe.prepare_latents(*args)
            kwargs["mu"] = calculate_shift(
                latents.shape[1], schedule.get("base_image_seq_len", 256),
                schedule.get("max_image_seq_len", 4096), schedule.get("base_shift", .5),
                schedule.get("max_shift", 1.15))
            kwargs["sigmas"] = (None if schedule.get("use_flow_sigmas") else
                                 np.linspace(1, 1 / opts["steps"], opts["steps"]))
        else:
            from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps
            latents = pipe.prepare_latents(*args)
            if schedule.get("use_dynamic_shifting"):
                patch = pipe.transformer.config.patch_size
                length = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
                kwargs["mu"] = calculate_shift(
                    length, schedule.get("base_image_seq_len", 256),
                    schedule.get("max_image_seq_len", 4096), schedule.get("base_shift", .5),
                    schedule.get("max_shift", 1.16))
        retrieve_timesteps(pipe.scheduler, opts["steps"], self.device, **kwargs)
        pipe.scheduler.set_begin_index(0)
        return latents

    def raw_velocity(self, latents, time, condition):
        size = latents.shape[0]
        time = torch.as_tensor(time, device=latents.device).expand(size)
        kwargs = dict(hidden_states=latents, timestep=time, encoder_hidden_states=condition["tokens"],
                      pooled_projections=condition["pooled"], joint_attention_kwargs={}, return_dict=False)
        if self.flux:
            kwargs.update(
                timestep=time.to(latents.dtype) / 1000, txt_ids=condition["ids"], img_ids=self.image_ids,
                guidance=(torch.full((size,), self.options["guidance_scale"], device=latents.device,
                                     dtype=torch.float32)
                          if self.pipe.transformer.config.guidance_embeds else None))
        self.forward_calls += 1
        self.sample_evaluations += size
        return self.pipe.transformer(**kwargs)[0]

    def velocity(self, latents, time, condition, uncond=None):
        if self.use_cfg and uncond is None and not self.flux:
            # Preserve SD3's native [negative, positive] CFG batch and rounding.
            combined = {"tokens": torch.cat((self.negative["tokens"], condition["tokens"])),
                        "pooled": torch.cat((self.negative["pooled"], condition["pooled"]))}
            uncond, positive = self.raw_velocity(torch.cat((latents, latents)), time, combined).chunk(2)
            return uncond + self.cfg_scale * (positive - uncond)
        positive = self.raw_velocity(latents, time, condition)
        if not self.use_cfg:
            return positive
        if uncond is None:
            uncond = self.raw_velocity(latents, time, self.negative)
        return uncond + self.cfg_scale * (positive - uncond)

    def pair(self, latents, time, forget_text, anchor_text):
        """Both conditions use benchmark guidance and share the unconditional call."""
        uncond = self.raw_velocity(latents, time, self.negative) if self.use_cfg else None
        forget = self.velocity(latents, time, self.condition(forget_text, len(latents)), uncond)
        anchor = self.velocity(latents, time, self.condition(anchor_text, len(latents)), uncond)
        return forget, anchor

    def decode(self, latents):
        pipe, opts = self.pipe, self.options
        if self.flux:
            latents = pipe._unpack_latents(latents, opts["height"], opts["width"], pipe.vae_scale_factor)
        latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        if self.flux:
            from diffusers.utils.torch_utils import get_module_execution_device
            latents = latents.to(get_module_execution_device(pipe.vae))
        images = pipe.image_processor.postprocess(pipe.vae.decode(latents, return_dict=False)[0], output_type="pil")
        pipe.maybe_free_model_hooks()
        return images

    def close(self):
        self.conditions.clear()
        self.user = self.negative = None
        super().close()
