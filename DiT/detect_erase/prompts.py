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


def classification_loss(prompt, pooled, labels, tau, temperature):
    scores = F.cosine_similarity(pooled.float(), prompt.float(), dim=-1)
    return F.binary_cross_entropy_with_logits((scores - tau) / temperature, labels.float())


def cfg_loss(positive, anchor, negative, scale):
    guided = negative.float() + scale * (positive.float() - negative.float())
    return F.mse_loss(guided, anchor.float())


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
        self.track = config.get("tracking", {}).get("enabled", False)
        self.config["adapter"] = copy.deepcopy(self.options)
        self.pipe.transformer.eval().requires_grad_(False)
        for component in self.pipe.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval().requires_grad_(False)
        self.prompts = load_concept_prompts(self.options.get("checkpoints", []))

    @torch.no_grad()
    def encode(self, texts, *, clip_only=False):
        """Use the same pooled representation for detector training and inference."""
        if clip_only:
            if self.flux:
                return self.pipe._get_clip_prompt_embeds(texts, device=self.device)
            pooled = [self.pipe._get_clip_prompt_embeds(texts, device=self.device, clip_model_index=i)[1]
                      for i in (0, 1)]
            return torch.cat(pooled, dim=-1)
        kwargs = dict(prompt=texts, device=self.device,
                      max_sequence_length=self.config["sampling"]["max_sequence_length"])
        if self.flux:
            sequence, pooled, _ = self.pipe.encode_prompt(**kwargs)
        else:
            sequence, _, pooled, _ = self.pipe.encode_prompt(
                prompt_2=None, prompt_3=None, do_classifier_free_guidance=False, **kwargs)
        return sequence.detach(), pooled.detach()

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

    def negatives(self, condition):
        """Select learned negative embeddings once per user prompt."""
        sequence, pooled = condition
        targets = self.config["targets"]
        entries = [self.prompts[target] for target in targets]
        device = pooled.device
        if self.options["similarity"] == "cosine":
            prototypes = torch.cat([entry["detect_pooled"] for entry in entries]).to(device)
            scores = F.normalize(pooled.float(), dim=-1) @ F.normalize(prototypes.float(), dim=-1).T
            thresholds = torch.tensor([entry["tau"] for entry in entries], device=device)
            hits = scores >= (self.options["tau"] if self.options["tau"] is not None else thresholds)
            best = scores.argmax(dim=1)
        else:
            distances = []
            for entry in entries:
                square = (sequence.float() - entry["detect_sequence"].to(device)).square().flatten(1).sum(1)
                square += (pooled.float() - entry["detect_pooled"].to(device)).square().sum(1)
                if self.options["l2_reduction"] == "rms":
                    square /= sequence[0].numel() + pooled[0].numel()
                distances.append(square.sqrt())
            scores = torch.stack(distances, dim=1)
            hits = scores <= self.options["l2_threshold"]
            best = scores.argmin(dim=1)

        selected = hits.clone()
        if self.options["multi_concept"] == "top1":
            selected.zero_()
            selected.scatter_(1, best[:, None], hits.any(dim=1, keepdim=True))
        weights = selected.float() / selected.sum(dim=1, keepdim=True).clamp_min(1)
        negative_sequence = torch.cat([entry["erase_sequence"] for entry in entries]).to(device)
        negative_pooled = torch.cat([entry["erase_pooled"] for entry in entries]).to(device)
        combined_sequence = torch.einsum("bc,csd->bsd", weights, negative_sequence)
        combined_pooled = weights @ negative_pooled
        null_sequence, null_pooled = self.encode([""])
        active = selected.any(dim=1)
        result = (torch.where(active[:, None, None], combined_sequence, null_sequence.float()).to(self.dtype),
                  torch.where(active[:, None], combined_pooled, null_pooled.float()).to(self.dtype))
        decisions = [{"scores": dict(zip(targets, row.tolist())),
                      "matched": [target for target, flag in zip(targets, matched.tolist()) if flag],
                      "selected": [target for target, flag in zip(targets, chosen.tolist()) if flag]}
                     for row, matched, chosen in zip(scores, hits, selected)]
        return result, decisions

    @torch.inference_mode()
    def generate(self, records, sampling_config):
        condition = self.encode([record["prompt"] for record in records])
        negative, decisions = self.negatives(condition)
        opts = sampling_config
        kwargs = dict(prompt_embeds=condition[0], pooled_prompt_embeds=condition[1],
                      negative_prompt_embeds=negative[0], negative_pooled_prompt_embeds=negative[1],
                      latents=self.latents([record["seed"] for record in records], opts),
                      generator=[torch.Generator("cpu").manual_seed(record["seed"]) for record in records],
                      height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                      guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"])
        if self.flux:
            kwargs["true_cfg_scale"] = opts["true_cfg_scale"]
        images = self.pipe(**kwargs).images
        if self.track:
            with Path(self.options["gate_log"]).open("a", encoding="utf-8") as output:
                for record, decision in zip(records, decisions):
                    output.write(json.dumps({"sample_id": record["sample_id"], "prompt": record["prompt"],
                                             "concept": record.get("concept", ""),
                                             "seed": record["seed"], **decision}, ensure_ascii=False) + "\n")
        return images
