"""Text conditions, prompt objectives and the benchmark sampling adapter."""
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise
from DiT.detect_erase.encoding import TextEncoder


def classification_loss(scores, labels, tau, temperature):
    return F.binary_cross_entropy_with_logits((scores - tau) / temperature, labels.float())


def supervised_contrastive_loss(scores, labels, temperature):
    """Use the learned detect prototype as anchor and all positive texts as positives."""
    logits = scores.float() / temperature
    return torch.logsumexp(logits, dim=0) - logits[labels.bool()].mean()


def load_concept_prompts(checkpoints):
    """Combine only the requested per-concept files in memory."""
    prompts = {}
    for item in checkpoints:
        entry = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
        prompts.setdefault(item["target"], {}).update(entry)
    return prompts


class PromptModel(DiTAdapter):
    """Share native model loading between prompt training and benchmark sampling."""

    def __init__(self, config):
        original = copy.deepcopy(config)
        original["adapter"] = {"kind": "original"}
        super().__init__(original)
        self.options = config["adapter"]
        self.config["adapter"] = copy.deepcopy(self.options)
        self.pipe.transformer.eval().requires_grad_(False)
        for component in self.pipe.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval().requires_grad_(False)
        self.text = TextEncoder(self.config, self.pipe)
        self.prompts = load_concept_prompts(self.options.get("checkpoints", []))
        if self.prompts:
            detectors = {item["target"]: self.prompts[item["target"]]
                         for item in self.options["checkpoints"] if item["stage"] == "detect"}
            self.text.prepare_detector(detectors, self.options)
        self.erase_conditions = {}

    def encode(self, texts, *, clip_only=False):
        return self.text.features(texts) if clip_only else self.text.encode(texts)

    def close(self):
        self.erase_conditions.clear()
        self.text.close()
        super().close()

    def latents(self, seeds, sampling):
        channels = self.pipe.transformer.config.in_channels // 4 if self.flux else self.pipe.transformer.config.in_channels
        shape = (channels, sampling["height"] // self.pipe.vae_scale_factor,
                 sampling["width"] // self.pipe.vae_scale_factor)
        noise = torch.stack([initial_noise(shape, seed, dtype=self.dtype, device="cpu") for seed in seeds])
        if self.flux:
            noise = self.pipe._pack_latents(noise, len(seeds), *shape)
        return noise.to(self.device)

    def timesteps(self, latents, sampling):
        """Match each pipeline's native flow schedule, including resolution shifts."""
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        scheduler = self.pipe.scheduler
        kwargs = {}
        if self.flux:
            import numpy as np
            length = latents.shape[1]
            if not scheduler.config.get("use_flow_sigmas", False):
                kwargs["sigmas"] = np.linspace(1.0, 1 / sampling["steps"], sampling["steps"])
            kwargs["mu"] = calculate_shift(length, scheduler.config.get("base_image_seq_len", 256),
                                           scheduler.config.get("max_image_seq_len", 4096),
                                           scheduler.config.get("base_shift", 0.5),
                                           scheduler.config.get("max_shift", 1.15))
        elif scheduler.config.get("use_dynamic_shifting", False):
            patch = self.pipe.transformer.config.patch_size
            length = (latents.shape[-2] // patch) * (latents.shape[-1] // patch)
            kwargs["mu"] = calculate_shift(length, scheduler.config.get("base_image_seq_len", 256),
                                           scheduler.config.get("max_image_seq_len", 4096),
                                           scheduler.config.get("base_shift", 0.5),
                                           scheduler.config.get("max_shift", 1.16))
        scheduler.set_timesteps(sampling["steps"], device=self.device, **kwargs)
        scheduler.set_begin_index(0)
        return scheduler.timesteps

    def velocity(self, latents, timestep, condition, sampling):
        """Keep autograd through the transformer to the learned text conditions."""
        sequence, pooled = condition
        batch = latents.shape[0]
        sequence = sequence.to(device=self.device, dtype=self.dtype).expand(batch, -1, -1)
        pooled = pooled.to(device=self.device, dtype=self.dtype).expand(batch, -1)
        kwargs = dict(hidden_states=latents, encoder_hidden_states=sequence,
                      pooled_projections=pooled, timestep=timestep.expand(batch), return_dict=False)
        if self.flux:
            height = sampling["height"] // self.pipe.vae_scale_factor // 2
            width = sampling["width"] // self.pipe.vae_scale_factor // 2
            kwargs.update(
                timestep=timestep.expand(batch).to(self.dtype) / 1000,
                txt_ids=torch.zeros(sequence.shape[1], 3, device=self.device, dtype=self.dtype),
                img_ids=self.pipe._prepare_latent_image_ids(batch, height, width, self.device, self.dtype),
                guidance=(torch.full((batch,), sampling["guidance_scale"], device=self.device, dtype=torch.float32)
                          if self.pipe.transformer.config.guidance_embeds else None),
            )
        return self.pipe.transformer(**kwargs)[0]

    def negatives(self, texts, condition):
        """Detect once, then average the selected native erase conditions."""
        decisions = self.text.detect(texts, condition)
        targets = self.config["targets"]
        device = condition[1].device
        for target in targets:
            if target not in self.erase_conditions:
                self.erase_conditions[target] = self.text.condition(self.prompts[target], "erase")
        weights = torch.tensor([[target in decision["selected"] for target in targets] for decision in decisions],
                               device=device, dtype=torch.float32)
        weights /= weights.sum(1, keepdim=True).clamp_min(1)
        negative_sequence = torch.cat([self.erase_conditions[target][0] for target in targets]).float()
        negative_pooled = torch.cat([self.erase_conditions[target][1] for target in targets]).float()
        combined_sequence = torch.einsum("bc,csd->bsd", weights, negative_sequence)
        if not self.flux:
            clip_length = self.text.tokenizers[0].model_max_length
            clip_width = sum(encoder.config.hidden_size for encoder in self.text.encoders)
            # Match normal SD3 prompt padding at the inference input.
            combined_sequence[:, :clip_length, clip_width:] = 0
        combined_pooled = weights @ negative_pooled
        null_sequence, null_pooled = self.encode([""])
        active = weights.any(dim=1)
        result = (torch.where(active[:, None, None], combined_sequence, null_sequence.float()).to(self.dtype),
                  torch.where(active[:, None], combined_pooled, null_pooled.float()).to(self.dtype))
        return result, decisions

    @torch.inference_mode()
    def generate(self, records, sampling_config):
        condition = self.encode([record["prompt"] for record in records])
        negative, decisions = self.negatives([record["prompt"] for record in records], condition)
        with Path(self.options["gate_log"]).open("a", encoding="utf-8") as output:
            for record, decision in zip(records, decisions):
                output.write(json.dumps({"sample_id": record["sample_id"], "prompt": record["prompt"],
                                         "concept": record.get("concept", ""),
                                         "seed": record["seed"], **decision}, ensure_ascii=False) + "\n")
        opts = sampling_config
        kwargs = dict(prompt_embeds=condition[0], pooled_prompt_embeds=condition[1],
                      negative_prompt_embeds=negative[0], negative_pooled_prompt_embeds=negative[1],
                      latents=self.latents([record["seed"] for record in records], opts),
                      generator=[torch.Generator("cpu").manual_seed(record["seed"]) for record in records],
                      height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                      guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"])
        if self.flux:
            kwargs["true_cfg_scale"] = opts["true_cfg_scale"]
        return self.pipe(**kwargs).images
