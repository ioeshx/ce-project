"""Detect target concepts, then denoise with the anchor's full text condition."""
from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path

import torch

from DiT.queryGraft.method import ConceptModulation, QueryAttention
from DiT.detectFix.diagnostics import DetectFixDiagnostics
from benchmark.common import ROOT, read_json, write_json
from benchmark.dit_adapter import DiTAdapter


SUPPORTED_MODELS = ("sd35_medium", "flux1_dev")
DETECTION_MODES = ("once", "every_step")
DEFAULTS = dict(background="background", localization_layers="all", concept_include_prompt=False,
                rho=0.0, detection_start_step=0, detection_mode="once", trace=False, trace_root=None)


def resolve_settings(config, overrides=None):
    """Resolve method options once at the run entry, before loading the model."""
    settings = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(config["adapter"].get("detect_fix", {})),
                **(overrides or {})}
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    anchors.update(settings.pop("anchor_overrides", {}))
    settings["pairs"] = settings.get("pairs") or [
        {"target": target, "anchor": config.get("erasure_anchor") or anchors[target]}
        for target in config["targets"]]
    layers = settings["localization_layers"]
    if isinstance(layers, str) and layers != "all":
        layers = [int(index) for index in layers.split(",")]
    settings["localization_layers"] = layers
    problems = []
    if config["model_name"] not in SUPPORTED_MODELS:
        problems.append(f"supported models: {SUPPORTED_MODELS}")
    if layers != "all" and (not isinstance(layers, list) or not layers or
                            any(type(i) is not int or i < 0 for i in layers) or
                            len(set(layers)) != len(layers)):
        problems.append("localization_layers must be all or a nonempty list of unique 0-based indices")
    if not isinstance(settings["background"], str) or not settings["background"].strip():
        problems.append("background must be a nonempty concept phrase")
    if not settings["pairs"] or any(not pair["target"].strip() or not pair["anchor"].strip()
                                    for pair in settings["pairs"]):
        problems.append("provide nonempty target/anchor pairs")
    if settings["detection_mode"] not in DETECTION_MODES:
        problems.append(f"detection_mode must be one of {DETECTION_MODES}")
    if type(settings["detection_start_step"]) is not int or settings["detection_start_step"] < 0:
        problems.append("detection_start_step must be a nonnegative integer")
    if not math.isfinite(settings["rho"]) or not 0 <= settings["rho"] <= 1:
        problems.append("rho must lie in [0,1]")
    if not isinstance(settings["concept_include_prompt"], bool):
        problems.append("concept_include_prompt must be a boolean")
    if not isinstance(settings["trace"], bool):
        problems.append("trace must be a boolean")
    if config["sampling"].get("true_cfg_scale", 1) != 1:
        problems.append("FLUX uses embedded guidance; set true_cfg_scale=1")
    if config["sampling"].get("skip_guidance_layers"):
        problems.append("SD3 skip-layer guidance is outside the native CFG path used here")
    if problems:
        raise ValueError("; ".join(problems))
    if settings["trace_root"]:
        settings["trace_root"] = str(Path(settings["trace_root"]).expanduser().resolve())
    return settings


class DetectFix(DiTAdapter):
    """Read-only concept detection plus per-image, global text-condition switching."""

    def __init__(self, config):
        self.settings = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(config["adapter"]["detect_fix"])}
        native = copy.deepcopy(config)
        model_path = config["adapter"].get("model_path")
        native["adapter"] = {"kind": "diffusers", "model_path": model_path} if model_path else {"kind": "original"}
        super().__init__(native)
        # `active` enables only the concept readout; anchor passes use native attention.
        self.sampling_active, self.active, self.mode = False, False, "idle"
        self.handles, self.processors, self.block_forwards = [], [], []
        transformer = self.pipe.transformer
        blocks = [(f"transformer_blocks.{i}", block, False)
                  for i, block in enumerate(transformer.transformer_blocks)]
        if self.flux:
            blocks += [(f"single_transformer_blocks.{i}", block, True)
                       for i, block in enumerate(transformer.single_transformer_blocks)]
        layers = self.settings["localization_layers"]
        self.localizers = list(range(len(blocks))) if layers == "all" else layers
        self.sites = [{"layer": i, "path": blocks[i][0]} for i in self.localizers]
        self.concept_count = 3 * len(self.settings["pairs"])
        self.handles.append(transformer.time_text_embed.register_forward_hook(self.concept_condition))
        for layer, (_, block, single) in enumerate(blocks):
            original = block.attn.processor
            self.processors.append((block.attn, original))
            block.attn.set_processor(QueryAttention(self, layer, single, original))
            modulation = ConceptModulation(self, block, single)
            self.block_forwards.append((block, block.forward))
            block.forward = modulation
            self.handles.append((block.norm if single else block.norm1_context)
                                .register_forward_hook(modulation.normalize))
            self.handles.append(block.attn.register_forward_hook(modulation.attention))
        self.native_forward = transformer.forward
        self.forward_signature = inspect.signature(self.native_forward)
        transformer.forward = self.forward

        # Encode each phrase independently; localization uses its first T5 token.
        phrases = [text for pair in self.settings["pairs"]
                   for text in (pair["target"], pair["anchor"], self.settings["background"])]
        anchors = list(dict.fromkeys(pair["anchor"] for pair in self.settings["pairs"]))
        opts = config["sampling"]
        with torch.inference_mode():
            self.concepts = self.pipe._get_t5_prompt_embeds(
                prompt=phrases, max_sequence_length=opts["max_sequence_length"],
                num_images_per_prompt=1, device=self.device, dtype=self.dtype)[:, 0].unsqueeze(0).cpu()
            encoding = dict(prompt=anchors, device=self.device, max_sequence_length=opts["max_sequence_length"])
            if self.flux:
                sequence, pooled, _ = self.pipe.encode_prompt(**encoding)
            else:
                sequence, _, pooled, _ = self.pipe.encode_prompt(
                    **encoding, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False)
            self.anchor_conditions = {anchor: (sequence[i:i+1].cpu(), pooled[i:i+1].cpu())
                                      for i, anchor in enumerate(anchors)}
        self.diagnostics = DetectFixDiagnostics(self) if self.settings["trace"] else None

    def concept_condition(self, module, args, output):
        if self.active:
            # Retain native time/guidance modulation with zero pooled CLIP input.
            self.concept_temb = module.forward(*args[:-1], torch.zeros_like(args[-1]))

    def attention_output(self, layer, image_av):
        # QueryAttention's image path remains unchanged; only concepts are read.
        return image_av

    def record_concepts(self, layer, image_av, concept_av):
        if layer not in self.localizers:
            return
        # Native SD3 CFG rows are [negative, positive]; only positive rows detect.
        image_av, concept_av = image_av[-self.batch:], concept_av[-self.batch:]
        concepts = concept_av.reshape(self.batch, -1, 3, concept_av.shape[-1]).float()
        scores = torch.einsum("bpd,bkcd->bkpc", image_av.float(), concepts).softmax(dim=-1)
        if self.score_sum is None:
            self.score_sum = scores
        else:
            self.score_sum.add_(scores.to(self.score_sum.device))

    def select_inputs(self, call, indices):
        """Select whole images, preserving SD3's negative/positive CFG ordering."""
        rows = torch.cat((indices, indices + self.sample_count)) if self.cfg else indices
        inputs = dict(call)
        for key in ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep", "guidance"):
            if inputs.get(key) is not None:
                inputs[key] = inputs[key].index_select(0, rows.to(inputs[key].device))
        # FLUX txt_ids/img_ids are shared position IDs, without a batch dimension.
        return inputs, rows

    def forward(self, *args, **kwargs):
        if not self.sampling_active:
            return self.native_forward(*args, **kwargs)
        call = dict(self.forward_signature.bind(*args, **kwargs).arguments)
        self.cfg = not self.flux and call["hidden_states"].shape[0] == 2 * self.sample_count
        choices = (self.selected_pairs.clone() if self.settings["detection_mode"] == "once"
                   else torch.full_like(self.selected_pairs, -1))
        eligible = choices < 0
        if self.step < self.settings["detection_start_step"]:
            eligible.fill_(False)
        indices = eligible.nonzero().flatten()
        areas = [None] * self.sample_count
        prediction = None

        # 1. Detect from the current latent with the original user text condition.
        if indices.numel():
            probe, rows = self.select_inputs(call, indices)
            self.batch, self.text_count = len(indices), probe["encoder_hidden_states"].shape[1]
            text = probe["encoder_hidden_states"]
            concepts = self.concepts.to(text).expand(text.shape[0], -1, -1)
            probe["encoder_hidden_states"] = torch.cat((probe["encoder_hidden_states"], concepts), dim=1)
            if self.flux:
                ids = probe["txt_ids"]
                probe["txt_ids"] = torch.cat((ids, ids.new_zeros(self.concept_count, 3)))
            self.score_sum = None
            self.active, self.mode = True, "probe"
            self.forward_calls += 1
            output = self.native_forward(**probe)
            self.active, self.mode = False, "idle"
            scores = self.score_sum / len(self.localizers)
            target = scores[..., 0]
            mask = ((target > target.mean(dim=-1, keepdim=True)) &
                    (target > scores[..., 1]) & (target > scores[..., 2]))
            ratios = mask.float().mean(-1).cpu()
            triggered = ratios > self.settings["rho"]
            # A whole image has one text condition: largest qualifying area wins.
            best = ratios.masked_fill(~triggered, -1).argmax(dim=-1)
            choices[indices] = torch.where(triggered.any(-1), best, -1)
            for index, values in zip(indices.tolist(), ratios.tolist()):
                areas[index] = values
            if self.diagnostics is not None:
                self.diagnostics.record_detection(
                    step=self.step, timestep=float(call["timestep"][0].item()), indices=indices,
                    masks=mask, ratios=ratios, triggered=triggered, choices=choices,
                    latents=call["hidden_states"][-self.sample_count:])
            prediction = output[0].new_empty((call["hidden_states"].shape[0], *output[0].shape[1:]))
            prediction.index_copy_(0, rows.to(prediction.device), output[0])
            self.score_sum = self.concept_temb = None

        # 2. Replace the entire positive text condition, then predict from the same x_t.
        # The pipeline performs CFG and advances its scheduler exactly once afterward.
        indices = (choices >= 0).nonzero().flatten()
        if indices.numel():
            fixed, rows = self.select_inputs(call, indices)
            fixed["encoder_hidden_states"] = fixed["encoder_hidden_states"].clone()
            fixed["pooled_projections"] = fixed["pooled_projections"].clone()
            positive_start = len(indices) if self.cfg else 0
            for pair_index, pair in enumerate(self.settings["pairs"]):
                positions = (choices[indices] == pair_index).nonzero().flatten() + positive_start
                if not positions.numel():
                    continue
                sequence, pooled = self.anchor_conditions[pair["anchor"]]
                for key, condition in (("encoder_hidden_states", sequence), ("pooled_projections", pooled)):
                    fixed[key][positions.to(fixed[key].device)] = condition.to(fixed[key])
            self.forward_calls += 1
            output = self.native_forward(**fixed)
            if prediction is None:
                prediction = output[0].new_empty((call["hidden_states"].shape[0], *output[0].shape[1:]))
            prediction.index_copy_(0, rows.to(prediction.device), output[0])
        elif prediction is None:  # Initial denoising steps before detection starts.
            self.forward_calls += 1
            output = self.native_forward(**call)
            prediction = output[0]

        self.selected_pairs = choices
        for index, pair_index in enumerate(choices.tolist()):
            condition = (self.settings["pairs"][pair_index]["anchor"] if pair_index >= 0
                         else self.records[index]["prompt"])
            self.decisions[index].append({"step_index": self.step, "timestep": float(call["timestep"][0].item()),
                "checked": bool(eligible[index]), "target_area": areas[index],
                "selected_pair": pair_index, "condition": condition})
        self.step += 1
        if isinstance(output, tuple):
            return (prediction, *output[1:])
        output.sample = prediction
        return output

    def generate(self, records, sampling_config):
        self.records, self.sample_count = records, len(records)
        self.step, self.forward_calls = 0, 0
        self.selected_pairs = torch.full((len(records),), -1, dtype=torch.long)
        self.decisions = [[] for _ in records]
        if self.diagnostics is not None:
            self.diagnostics.begin(records, sampling_config)
        self.sampling_active = True
        try:
            images = super().generate(records, sampling_config,
                callback_on_step_end=self.diagnostics.step_end if self.diagnostics is not None else None)
        finally:
            self.sampling_active, self.active, self.mode = False, False, "idle"
        for record, decisions in zip(records, self.decisions):
            write_json(Path(record["image"]).with_suffix(".detectFix.json"), {
                "record": record, "settings": self.settings, "sampling": sampling_config,
                "localization_sites": self.sites, "steps": decisions, "batch_forward_calls": self.forward_calls})
        if self.diagnostics is not None:
            self.diagnostics.finish(self.decisions, self.forward_calls)
        return images

    def close(self):
        self.sampling_active, self.active = False, False
        self.pipe.transformer.forward = self.native_forward
        for module, processor in self.processors:
            module.set_processor(processor)
        for block, forward in self.block_forwards:
            block.forward = forward
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.processors.clear()
        self.block_forwards.clear()
        self.anchor_conditions.clear()
        self.native_forward = None
        super().close()
