"""Read-only concept queries localize targets for source updates or explicit zeroing."""
from __future__ import annotations

import copy
from functools import partial
import inspect
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from DiT.queryGraft.diagnostics import save_sample_diagnostics, save_step_diagnostics
from DiT.queryGraft.source import (
    SOURCE_MODES, controlled_attention, joint_prompts, text_message_attention, token_groups,
)
from benchmark.common import ROOT, read_json, write_json
from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise


DIRECTORY = Path(__file__).resolve().parent
SUPPORTED_MODELS = ("sd35_medium", "flux1_dev")
MASK_RULES = ("target_max", "anchor_min", "mean_target_max")
UPDATE_RULES = ("replace", "blend", "add", "zero")
REPRESENTATIONS = ("text_output", "av", "residual", "block_output", "velocity")
DEFAULTS = dict(background="background", mask_rule="target_max", concept_include_prompt=False,
                my_layer_aggre=False, my_layer_policy=False,
                localization_layers="all", edit_layers="all",
                representation="text_output", eta=0.0, rho=0.0, update_rule="replace", strength=1.0,
                source_mode="anchor", source_user_weight=1.0, source_anchor_weight=1.0, source_scale=1.0,
                trace=False, trace_root=None)


def resolve_settings(config, overrides=None):
    """Resolve and validate method configuration at the runner entry."""
    settings = {**copy.deepcopy(DEFAULTS),
                **config["adapter"].get("query_graft", {}), **(overrides or {})}
    if (settings["representation"] == "velocity"
            and "edit_layers" not in config["adapter"].get("query_graft", {})
            and "edit_layers" not in (overrides or {})):
        settings["edit_layers"] = "last"
    anchors = read_json(ROOT / "configs/dit_target_anchor.json")
    anchors.update(settings.pop("anchor_overrides", {}))
    settings["pairs"] = settings.get("pairs") or [
        {"target": target, "anchor": config.get("erasure_anchor") or anchors[target]}
        for target in config["targets"]]
    problems = []
    for key, names in (("localization_layers", {"all"}), ("edit_layers", {"all", "last"})):
        layers = settings[key]
        if isinstance(layers, str) and layers not in names:
            layers = [int(value) for value in layers.split(",")]
        settings[key] = layers
        if not (isinstance(layers, str) and layers in names):
            if not isinstance(layers, list) or not layers or any(type(i) is not int or i < 0 for i in layers):
                problems.append(f"{key}: use {sorted(names)} or a nonempty list of 0-based layers")
            elif len(set(layers)) != len(layers):
                problems.append(f"{key}: layer indices must be unique")
    if config["model_name"] not in SUPPORTED_MODELS:
        problems.append(f"supported models: {SUPPORTED_MODELS}")
    if not isinstance(settings["background"], str) or not settings["background"].strip():
        problems.append("background must be a nonempty concept phrase")
    if not settings["pairs"] or any(not p["target"].strip() or not p["anchor"].strip()
                                    for p in settings["pairs"]):
        problems.append("provide nonempty target/anchor pairs")
    if settings["mask_rule"] not in MASK_RULES:
        problems.append(f"mask_rule must be one of {MASK_RULES}")
    for name in ("concept_include_prompt", "my_layer_aggre", "my_layer_policy"):
        if not isinstance(settings[name], bool):
            problems.append(f"{name} must be a boolean")
    for name in ("eta", "rho"):
        if not math.isfinite(settings[name]) or not 0 <= settings[name] <= 1:
            problems.append(f"{name} must lie in [0,1]")
    if settings["representation"] not in REPRESENTATIONS:
        problems.append(f"representation must be one of {REPRESENTATIONS}")
    if settings["representation"] == "velocity" and settings["edit_layers"] != "last":
        problems.append("velocity has one final Transformer output site; use edit_layers=last")
    if settings["source_mode"] not in SOURCE_MODES:
        problems.append(f"source_mode must be one of {SOURCE_MODES}")
    for name in ("source_user_weight", "source_anchor_weight", "source_scale"):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            problems.append(f"{name} must be finite and nonnegative")
    if settings["source_mode"] == "logits" and (
            settings["source_user_weight"] <= 0 or settings["source_anchor_weight"] <= 0):
        problems.append("logits source weights must be positive")
    if settings["update_rule"] not in UPDATE_RULES:
        problems.append(f"update_rule must be one of {UPDATE_RULES}")
    if not math.isfinite(settings["strength"]) or settings["strength"] < 0:
        problems.append("strength must be finite and nonnegative")
    if settings["update_rule"] == "blend" and settings["strength"] > 1:
        problems.append("blend requires strength in [0,1]")
    if config["sampling"].get("true_cfg_scale", 1) != 1:
        problems.append("FLUX uses embedded guidance; set true_cfg_scale=1")
    if config["sampling"].get("skip_guidance_layers"):
        problems.append("SD3 skip-layer guidance is outside the native CFG path used here")
    if problems:
        raise ValueError("; ".join(problems))
    if settings["trace_root"]:
        settings["trace_root"] = str(Path(settings["trace_root"]).expanduser().resolve())
    return settings


class QueryAttention:
    """Native Q/K/V projections with separate base and read-only concept attention."""

    def __init__(self, owner, layer, single, original):
        self.owner, self.layer, self.single, self.original = owner, layer, single, original

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        owner = self.owner
        if not owner.active:
            native_kwargs = dict(encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask, **kwargs)
            if owner.flux:
                native_kwargs["image_rotary_emb"] = image_rotary_emb
            return self.original(attn, hidden_states, **native_kwargs)

        # [batch,tokens,heads*dim] -> [batch,heads,tokens,dim].
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        if attn.norm_q is not None:
            query, key = attn.norm_q(query), attn.norm_k(key)

        if not self.single:
            cq = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            ck = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            cv = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            if attn.norm_added_q is not None:
                cq, ck = attn.norm_added_q(cq), attn.norm_added_k(ck)
            # SD3: [image,text,concept]; FLUX: [text,concept,image].
            if owner.flux:
                query, key, value = (torch.cat(parts, dim=2)
                                     for parts in ((cq, query), (ck, key), (cv, value)))
            else:
                query, key, value = (torch.cat(parts, dim=2)
                                     for parts in ((query, cq), (key, ck), (value, cv)))

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=2)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=2)

        reading = owner.mode in {"user", "probe"}
        text_count = owner.text_count
        if reading:
            nc = owner.concept_count
            start = text_count if owner.flux else query.shape[2] - nc
            base_indices = torch.cat((torch.arange(start, device=query.device),
                                      torch.arange(start + nc, query.shape[2], device=query.device)))
            qb, kb, vb = (tensor.index_select(2, base_indices) for tensor in (query, key, value))
            base_mask = None if attention_mask is None else attention_mask[..., base_indices, :][..., base_indices]
            ob = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=base_mask)
            # ConceptAttention Eq. (7)-(9): concepts read image + concept K/V.
            # Including prompt K/V is an explicit opt-in to the previous behavior.
            if owner.settings.get("concept_include_prompt", False):
                read_indices, kr, vr = base_indices, kb, vb
            else:
                image_slice = slice(text_count, None) if owner.flux else slice(0, hidden_states.shape[1])
                read_indices = base_indices[image_slice]
                kr, vr = kb[:, :, image_slice], vb[:, :, image_slice]
            concept_outputs = []
            # Each pair reads the selected tokens plus its own target/anchor/background.
            for pair in range(len(owner.settings["pairs"])):
                cs = slice(start + 3 * pair, start + 3 * pair + 3)
                kc, vc = torch.cat((kr, key[:, :, cs]), 2), torch.cat((vr, value[:, :, cs]), 2)
                concept_mask = None
                if attention_mask is not None:
                    columns = torch.cat((read_indices, torch.arange(cs.start, cs.stop, device=query.device)))
                    concept_mask = attention_mask[..., cs, :].index_select(-1, columns)
                concept_outputs.append(F.scaled_dot_product_attention(
                    query[:, :, cs], kc, vc, attn_mask=concept_mask))
            oc = torch.cat(concept_outputs, dim=2)
            if owner.flux:
                image_av = ob[:, :, text_count:]
                output = torch.cat((ob[:, :, :text_count], oc, image_av), dim=2)
            else:
                image_av = ob[:, :, :hidden_states.shape[1]]
                output = torch.cat((ob, oc), dim=2)
            owner.record_concepts(self.layer,
                                  image_av.transpose(1, 2).flatten(2),
                                  oc.transpose(1, 2).flatten(2))
        else:
            if owner.mode == "anchor" and owner.settings["source_mode"] in {"logits", "values"}:
                user_tokens, anchor_tokens = owner.source_groups[owner.current_anchor]
                output = controlled_attention(
                    query, key, value, attention_mask, text_count=text_count, flux=owner.flux,
                    user_tokens=user_tokens, anchor_tokens=anchor_tokens, region=owner.source_region,
                    mode=owner.settings["source_mode"], user_weight=owner.settings["source_user_weight"],
                    anchor_weight=owner.settings["source_anchor_weight"])
            else:
                output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        message_av = None
        if (owner.settings.get("representation") == "text_output" and self.layer in owner.editors
                and (owner.mode != "probe" or owner.settings["source_mode"] == "extrapolate")):
            text_controls = {}
            if owner.mode == "anchor" and owner.settings["source_mode"] in {"logits", "values"}:
                user_tokens, anchor_tokens = owner.source_groups[owner.current_anchor]
                text_controls = dict(
                    user_tokens=user_tokens, anchor_tokens=anchor_tokens, region=owner.source_region,
                    mode=owner.settings["source_mode"], user_weight=owner.settings["source_user_weight"],
                    anchor_weight=owner.settings["source_anchor_weight"])
            message_av = text_message_attention(
                qb if reading else query, kb if reading else key, vb if reading else value,
                base_mask if reading else attention_mask,
                text_count=text_count, flux=owner.flux, **text_controls)
            message_av = message_av.transpose(1, 2).flatten(2).to(query.dtype)

        output = output.transpose(1, 2).flatten(2).to(query.dtype)
        if self.single:
            image_start = text_count + (owner.concept_count if reading else 0)
            if message_av is not None:
                # The native single-stream block projects AV and parallel MLP together.
                owner.single_text_messages[self.layer] = message_av
            image_av = owner.attention_output(self.layer, output[:, image_start:])
            return torch.cat((output[:, :image_start], image_av), dim=1)

        if owner.flux:
            ntext = encoder_hidden_states.shape[1]
            text_av, image_av = output[:, :ntext], output[:, ntext:]
        else:
            nimage = hidden_states.shape[1]
            image_av, text_av = output[:, :nimage], output[:, nimage:]
        image_av = owner.attention_output(self.layer, image_av)
        image_output = attn.to_out[0](image_av.contiguous())
        if message_av is not None:
            message = F.linear(message_av, attn.to_out[0].weight, bias=None)
            image_output = owner.projected_text_output(self.layer, image_output, message)
        image_output = attn.to_out[1](image_output)
        if not attn.context_pre_only:
            text_av = attn.to_add_out(text_av.contiguous())
        return image_output, text_av


class ConceptModulation:
    """Reuse native text modules with a separate, zero-pooled concept condition."""

    def __init__(self, owner, block, single):
        self.owner, self.block, self.single = owner, block, single
        self.original = block.forward

    def reading(self):
        return self.owner.active and self.owner.mode in {"user", "probe"}

    def normalize(self, module, args, output):
        if not self.reading():
            return
        start = self.owner.text_count
        stop = start + self.owner.concept_count
        self.input = args[0][:, start:stop]
        if not self.single and self.block.attn.context_pre_only:
            normalized = module.forward(self.input, self.owner.concept_temb)
            return torch.cat((output[:, :start], normalized, output[:, stop:]), dim=1)
        self.normalized, *self.gates = module.forward(self.input, emb=self.owner.concept_temb)
        normalized = torch.cat((output[0][:, :start], self.normalized, output[0][:, stop:]), dim=1)
        # Native gates still apply to image/prompt rows. Concept residuals are
        # recomputed below using the gates from the separate concept condition.
        return (normalized, *output[1:])

    def attention(self, module, args, output):
        if self.reading():
            start = self.owner.text_count
            states = output if self.single else output[1]
            self.av = states[:, start:start + self.owner.concept_count]

    def __call__(self, *args, **kwargs):
        if not self.reading():
            return self.original(*args, **kwargs)
        try:
            context, image = self.original(*args, **kwargs)
            if context is None:  # SD3's final context_pre_only block has no text update.
                return context, image
            if self.single:
                mlp = self.block.act_mlp(self.block.proj_mlp(self.normalized))
                concepts = self.input + self.gates[0][:, None] * self.block.proj_out(
                    torch.cat((self.av, mlp), dim=-1))
            else:
                gate_msa, shift_mlp, scale_mlp, gate_mlp = self.gates
                concepts = self.input + gate_msa[:, None] * self.av
                normalized = self.block.norm2_context(concepts)
                normalized = normalized * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                concepts = concepts + gate_mlp[:, None] * self.block.ff_context(normalized)
            if self.owner.flux and concepts.dtype == torch.float16:
                concepts = concepts.clip(-65504, 65504)
            return torch.cat((context[:, :self.owner.text_count], concepts), dim=1), image
        finally:
            # These slices can otherwise retain full image/text activation buffers.
            self.input = self.normalized = self.gates = self.av = None


class QueryGraft(DiTAdapter):
    """Own native sampling, concept readouts and live source activations in one adapter."""

    def __init__(self, config):
        self.settings = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(config["adapter"]["query_graft"])}
        native = copy.deepcopy(config)
        model_path = config["adapter"].get("model_path")
        native["adapter"] = {"kind": "diffusers", "model_path": model_path} if model_path else {"kind": "original"}
        super().__init__(native)
        self.active, self.mode = False, "idle"
        self.handles, self.processors, self.block_forwards = [], [], []
        self.velocity = self.settings["representation"] == "velocity"
        self.text_message = self.settings["representation"] == "text_output"
        self.zero_update = self.settings["update_rule"] == "zero"
        transformer = self.pipe.transformer
        blocks = [(f"transformer_blocks.{i}", block, False)
                  for i, block in enumerate(transformer.transformer_blocks)]
        if self.flux:
            blocks += [(f"single_transformer_blocks.{i}", block, True)
                       for i, block in enumerate(transformer.single_transformer_blocks)]
        count = len(blocks)
        self.localizers = list(range(count)) if self.settings["localization_layers"] == "all" else self.settings["localization_layers"]
        selection = self.settings["edit_layers"]
        self.editors = ([-1] if self.velocity else [count - 1] if selection == "last" else list(range(count))
                        if selection == "all" else selection)
        # Resolving actual module indices also exposes an out-of-range selection at setup.
        self.sites = [{"layer": i, "path": blocks[i][0], "single": blocks[i][2]}
                      for i in sorted(set(self.localizers) | (set() if self.velocity else set(self.editors)))]
        if self.velocity:
            self.sites.append({"layer": -1, "path": "output.sample", "single": False})
        self.inline = self.velocity or (not self.text_message and self.editors == [count - 1])
        self.last_layer = count - 1
        self.concept_count = 3 * len(self.settings["pairs"])
        self.dual_state = {}
        self.single_text_messages = {}
        self.sample_trace_sums = {}
        self.handles.append(transformer.time_text_embed.register_forward_hook(self.concept_condition))
        for layer, (_, block, single) in enumerate(blocks):
            original = block.attn.processor
            self.processors.append((block.attn, original))
            block.attn.set_processor(QueryAttention(self, layer, single, original))
            modulation = ConceptModulation(self, block, single)
            self.block_forwards.append((block, block.forward))
            block.forward = modulation
            normalizer = block.norm if single else block.norm1_context
            self.handles.append(normalizer.register_forward_hook(modulation.normalize))
            self.handles.append(block.attn.register_forward_hook(modulation.attention))
            if self.text_message and single and layer in self.editors:
                self.handles.append(block.proj_out.register_forward_hook(partial(self.single_text_output, layer)))
            elif self.settings["representation"] == "block_output" and layer in self.editors:
                self.handles.append(block.register_forward_hook(partial(self.block_output, layer)))
            elif self.settings["representation"] == "residual" and layer in self.editors:
                if single:
                    self.handles.append(block.register_forward_hook(partial(self.block_output, layer)))
                else:
                    self.handles.append(block.norm2.register_forward_pre_hook(partial(self.residual, layer)))
                    if not self.flux and block.use_dual_attention:
                        self.handles.append(block.norm1.register_forward_hook(partial(self.dual_attention, layer, "gate")))
                        self.handles.append(block.attn.register_forward_hook(partial(self.dual_attention, layer, "main")))
                        self.handles.append(block.attn2.register_forward_hook(partial(self.dual_attention, layer, "output")))
        self.native_forward = transformer.forward
        self.forward_signature = inspect.signature(self.native_forward)
        transformer.forward = self.forward
        patch = 2 if self.flux else transformer.config.patch_size
        opts = config["sampling"]
        self.grid = (opts["height"] // self.pipe.vae_scale_factor // patch,
                     opts["width"] // self.pipe.vae_scale_factor // patch)

        # Full phrases are independent batch items. Only localization uses token zero.
        phrases = [text for pair in self.settings["pairs"]
                   for text in (pair["target"], pair["anchor"], self.settings["background"])]
        self.anchor_conditions, self.source_groups = {}, {}
        with torch.inference_mode():
            self.concepts = self.pipe._get_t5_prompt_embeds(
                prompt=phrases, max_sequence_length=opts["max_sequence_length"],
                num_images_per_prompt=1, device=self.device, dtype=self.dtype)[:, 0].unsqueeze(0).cpu()
            if not self.zero_update and self.settings["source_mode"] == "anchor":
                anchors = list(dict.fromkeys(pair["anchor"] for pair in self.settings["pairs"]))
                sequence, pooled, text_ids = self.encode_source(anchors)
                self.anchor_conditions = {anchor: (sequence[i:i+1], pooled[i:i+1], text_ids)
                                          for i, anchor in enumerate(anchors)}

    def concept_condition(self, module, args, output):
        if self.active and self.mode in {"user", "probe"}:
            # Use the native timestep/guidance scaling and text embedding weights.
            # Only the pooled CLIP input is zeroed; projection biases are retained.
            self.concept_temb = module.forward(*args[:-1], torch.zeros_like(args[-1]))

    def encode_source(self, prompts):
        kwargs = dict(prompt=prompts, device=self.device,
                      max_sequence_length=self.config["sampling"]["max_sequence_length"])
        if self.flux:
            sequence, pooled, text_ids = self.pipe.encode_prompt(**kwargs)
        else:
            sequence, _, pooled, _ = self.pipe.encode_prompt(
                **kwargs, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False)
            text_ids = None
        return sequence.cpu(), pooled.cpu(), text_ids.cpu() if text_ids is not None else None

    def prepare_sources(self, records):
        """Encode native joint prompts per sample; localization concepts stay fixed."""
        if self.zero_update or self.settings["source_mode"] == "anchor":
            return
        self.anchor_conditions.clear()
        self.source_groups.clear()
        prompts = [record["prompt"] for record in records]
        length = self.config["sampling"]["max_sequence_length"]
        for anchor in dict.fromkeys(pair["anchor"] for pair in self.settings["pairs"]):
            joint = joint_prompts(prompts, anchor)
            condition = self.encode_source(joint)
            self.anchor_conditions[anchor] = condition
            if self.settings["source_mode"] in {"logits", "values"}:
                tokenizer = self.pipe.tokenizer_2 if self.flux else self.pipe.tokenizer_3
                user, added = token_groups(tokenizer, prompts, joint, length)
                if not added.any(-1).all():
                    raise ValueError("The appended source anchor was truncated; increase sampling.max_sequence_length "
                                     "in the config or shorten the prompt/anchor.")
                if not self.flux:
                    clip_user, clip_added = token_groups(
                        self.pipe.tokenizer, prompts, joint, self.pipe.tokenizer_max_length)
                    second = token_groups(self.pipe.tokenizer_2, prompts, joint, self.pipe.tokenizer_max_length)
                    # A native SD3 row contains both CLIP encoders' channels.
                    # Mixed/unaligned rows cannot be assigned to either group.
                    clip_user = clip_user & second[0]
                    clip_added = clip_added & second[1]
                    user, added = torch.cat((clip_user, user), -1), torch.cat((clip_added, added), -1)
                if user.shape[1] != condition[0].shape[1]:
                    raise ValueError("Source token groups do not match the native encoded sequence length.")
                self.source_groups[anchor] = user, added

    def with_concepts(self, call):
        inputs = dict(call)
        text = call["encoder_hidden_states"]
        concepts = self.concepts.to(text).expand(text.shape[0], -1, -1)
        inputs["encoder_hidden_states"] = torch.cat((text, concepts), dim=1)
        if self.flux:
            ids = call["txt_ids"]
            inputs["txt_ids"] = torch.cat((ids, ids.new_zeros(self.concept_count, 3)), dim=0)
        return inputs

    def region_mask(self, scores):
        """Apply the selected policy to one layer or the averaged soft maps."""
        if self.settings["my_layer_policy"]:
            if self.settings["mask_rule"] == "anchor_min":
                return (scores[..., 1] < scores[..., 0]) & (scores[..., 1] < scores[..., 2])
            target = scores[..., 0]
            target_max = (target > scores[..., 1]) & (target > scores[..., 2])
            if self.settings["mask_rule"] == "mean_target_max":
                return target_max & (target > target.mean(dim=-1, keepdim=True))
            return target_max
        if scores.shape[1] == 1:
            target = scores[..., 0]
            return target > target.mean(dim=-1, keepdim=True)
        # All configured target/anchor/background channels compete globally.
        labels = scores.transpose(1, 2).flatten(2).argmax(dim=-1)
        targets = 3 * torch.arange(scores.shape[1], device=scores.device)
        return labels[:, None] == targets[None, :, None]

    def record_concepts(self, layer, image_av, concept_av):
        if layer not in self.localizers and not self.settings["trace"]:
            return
        # CFG batch order is [negative,positive]. Score only positive image rows.
        image_av, concept_av = image_av[-self.batch:], concept_av[-self.batch:]
        concepts = concept_av.reshape(self.batch, -1, 3, concept_av.shape[-1]).float()
        # ConceptAttention scores are raw attention-output dot products in FP32.
        scores = torch.einsum("bpd,bkcd->bkpc", image_av.float(), concepts)
        if self.settings["my_layer_policy"]:
            probabilities = scores.softmax(dim=-1)
            # A spatial mean threshold depends on the score scale. Use soft
            # maps for the intersection, including when voting across layers.
            policy_scores = probabilities if self.settings["mask_rule"] == "mean_target_max" else scores
            mask = self.region_mask(policy_scores)
        else:
            batch, pairs, positions, channels = scores.shape
            probabilities = scores.transpose(1, 2).flatten(2).softmax(dim=-1)
            probabilities = probabilities.reshape(batch, positions, pairs, channels).transpose(1, 2)
            mask = self.region_mask(probabilities)
        if layer in self.localizers:
            if self.votes is None:
                self.votes = torch.zeros_like(mask, dtype=torch.int32)
                self.score_sum = torch.zeros_like(probabilities)
            self.votes.add_(mask.to(self.votes.device))
            self.score_sum.add_(probabilities.to(self.score_sum.device))
        if self.settings["trace"]:
            self.layer_ratios[str(layer)] = mask.float().mean(-1).cpu().tolist()
            if self.diagnostic_mask_sum is None:
                self.diagnostic_mask_sum = mask.float()
                self.diagnostic_score_sum = probabilities.detach().clone()
                self.diagnostic_logit_sum = scores.detach().clone()
            else:
                self.diagnostic_mask_sum.add_(mask.to(self.diagnostic_mask_sum.device))
                self.diagnostic_score_sum.add_(probabilities.to(self.diagnostic_score_sum.device))
                self.diagnostic_logit_sum.add_(scores.to(self.diagnostic_logit_sum.device))
            self.diagnostic_layers.append(layer)

    def finish_mask(self):
        self.vote_fraction = self.votes.float() / len(self.localizers)
        self.localization_scores = self.score_sum / len(self.localizers)
        if self.settings["my_layer_aggre"]:
            self.masks = self.vote_fraction > self.settings["eta"]
            confidence = self.vote_fraction
        else:
            self.masks = self.region_mask(self.localization_scores)
            confidence = self.localization_scores[..., 0]
        self.ratios = self.masks.float().mean(-1)
        self.triggered = self.ratios > self.settings["rho"]
        eligible = self.masks & self.triggered[..., None]
        confidence = confidence.masked_fill(~eligible, -1)
        self.winner = torch.where(eligible.any(1), confidence.argmax(1), -1)

    def capture_sources(self):
        """Reenter the native model from this step's original x_t; never step its scheduler."""
        if self.zero_update:
            return
        previous_mode, previous_text_count = self.mode, self.text_count
        self.mode = "anchor"
        try:
            for pair, definition in enumerate(self.settings["pairs"]):
                anchor = definition["anchor"]
                if not self.triggered[:, pair].any().item() or anchor in self.sources:
                    continue
                sequence, pooled, text_ids = self.anchor_conditions[anchor]
                if self.settings["source_mode"] in {"logits", "values"}:
                    self.source_region = torch.zeros_like(self.winner, dtype=torch.bool)
                    for index, pair_definition in enumerate(self.settings["pairs"]):
                        if pair_definition["anchor"] == anchor:
                            self.source_region |= self.winner == index
                call = dict(self.input_call)
                call["hidden_states"] = call["hidden_states"][-self.batch:]
                call["encoder_hidden_states"] = sequence.to(self.input_call["encoder_hidden_states"]).expand(self.batch, -1, -1)
                call["pooled_projections"] = pooled.to(self.input_call["pooled_projections"]).expand(self.batch, -1)
                call["timestep"] = call["timestep"][-self.batch:]
                if self.flux:
                    call["txt_ids"] = text_ids.to(call["txt_ids"])
                    if call.get("guidance") is not None:
                        call["guidance"] = call["guidance"][-self.batch:]
                self.text_count = sequence.shape[1]
                self.current_anchor = anchor
                self.sources[anchor] = {}
                self.forward_calls += 1
                output = self.native_forward(**call)
                if self.velocity:
                    self.sources[anchor][-1] = self.velocity_tokens(output[0]).detach()
        finally:
            self.mode, self.text_count = previous_mode, previous_text_count

    def velocity_tokens(self, velocity):
        """Group the completed latent-space output by the localization patch grid.

        SD3 has already unpatchified to BCHW. Grouping its values only changes
        their layout: selecting one row replaces every latent value of that
        patch. FLUX's native velocity is already packed in this layout.
        """
        if self.flux:
            return velocity
        batch, channels, height, width = velocity.shape
        rows, columns = self.grid
        return (velocity.reshape(batch, channels, rows, height // rows, columns, width // columns)
                .permute(0, 2, 4, 3, 5, 1).reshape(batch, rows * columns, -1))

    def velocity_output(self, output):
        """Patch the positive velocity after the entire native output head, before CFG."""
        velocity = output[0]
        image = self.velocity_tokens(velocity)
        self.remember_user(-1, image)
        self.finish_mask()
        if not self.triggered.any().item():
            return output
        self.capture_sources()
        changed = image.clone()
        changed[-self.batch:] = self.patch(-1, image[-self.batch:])
        if not self.flux:
            batch, channels, height, width = velocity.shape
            rows, columns = self.grid
            changed = (changed.reshape(batch, rows, columns, height // rows, width // columns, channels)
                       .permute(0, 5, 1, 3, 2, 4).reshape(velocity.shape))
        if isinstance(output, tuple):
            return (changed, *output[1:])
        output.sample = changed
        return output

    def patch(self, layer, image):
        """Edit disjoint winning regions; each update reads the same current user tensor."""
        result = image.clone()
        if self.zero_update:
            result.masked_fill_((self.winner >= 0).to(image.device)[..., None], 0)
        else:
            for pair, definition in enumerate(self.settings["pairs"]):
                anchor = definition["anchor"]
                if anchor not in self.sources:
                    continue
                source = self.sources[anchor][layer].to(image)
                if self.settings["source_mode"] == "extrapolate" and self.settings["source_scale"] != 1:
                    baseline = self.user_sources[layer].to(image).float()
                    source = (baseline + self.settings["source_scale"] * (source.float() - baseline)).to(image.dtype)
                if self.settings["update_rule"] == "replace":
                    edited = source
                elif self.settings["update_rule"] == "blend":
                    edited = (image.float() + self.settings["strength"] * (source.float() - image.float())).to(image.dtype)
                else:
                    edited = (image.float() + self.settings["strength"] * source.float()).to(image.dtype)
                mask = (self.winner == pair).to(image.device)[..., None]
                result = torch.where(mask, edited, result)
        if self.settings["trace"]:
            before, after = image.float(), result.float()
            delta = after - before
            before_energy = before.square().sum(-1)
            after_energy = after.square().sum(-1)
            delta_energy = delta.square().sum(-1)
            delta_max = delta.abs().amax(-1)
            statistics = []
            for pair in range(len(self.settings["pairs"])):
                selected = (self.winner == pair).to(image.device).float()
                tokens = selected.sum(-1)
                statistics.append(torch.stack((
                    (before_energy * selected).sum(-1),
                    (after_energy * selected).sum(-1),
                    (delta_energy * selected).sum(-1),
                    (delta_max * selected).amax(-1),
                    tokens * image.shape[-1], tokens), dim=-1))
            self.activation_sums[layer] = torch.stack(statistics, dim=1).detach().cpu()
        return result

    def remember_user(self, layer, image):
        if (not self.zero_update and self.settings["source_mode"] == "extrapolate" and self.mode in {"user", "probe"}
                and layer in self.editors):
            baseline = image[-self.batch:].detach()
            self.user_sources[layer] = baseline if self.inline else baseline.cpu()

    def attention_output(self, layer, image_av):
        if self.settings["representation"] == "av":
            self.remember_user(layer, image_av)
        if self.mode == "anchor":
            if layer in self.editors and self.settings["representation"] == "av":
                self.sources[self.current_anchor][layer] = (image_av.detach() if self.inline
                                                            else image_av.detach().cpu())
            return image_av
        if self.mode == "user" and layer == self.last_layer and not self.velocity:
            self.finish_mask()
            self.capture_sources()
        if (self.mode in {"user", "edit"} and layer in self.editors
                and self.settings["representation"] == "av"):
            changed = image_av.clone()
            changed[-self.batch:] = self.patch(layer, image_av[-self.batch:])
            return changed
        return image_av

    def projected_text_output(self, layer, output, message):
        """Replace projected text messages, retaining the current user output remainder."""
        self.remember_user(layer, message)
        if self.mode == "anchor":
            self.sources[self.current_anchor][layer] = message.detach().cpu()
        elif self.mode == "edit":
            before = message[-self.batch:]
            after = self.patch(layer, before)
            changed = (output[-self.batch:].float() + (after.float() - before.float())).to(output.dtype)
            result = output.clone()
            mask = (self.winner >= 0).to(output.device)[..., None]
            result[-self.batch:] = torch.where(mask, changed, output[-self.batch:])
            return result
        return output

    def single_text_output(self, layer, module, args, output):
        """Apply the text delta after FLUX's fused projection, before its gate/residual."""
        if not self.active or layer not in self.single_text_messages:
            return
        text_av = self.single_text_messages.pop(layer)
        message = F.linear(text_av, module.weight[:, :text_av.shape[-1]], bias=None)
        image_start = self.text_count + (self.concept_count if self.mode in {"user", "probe"} else 0)
        image = self.projected_text_output(layer, output[:, image_start:], message)
        return torch.cat((output[:, :image_start], image), dim=1)

    def dual_attention(self, layer, kind, module, args, output):
        if self.active and (self.mode != "probe" or self.settings["source_mode"] == "extrapolate"):
            if kind == "gate":
                self.dual_state[self.mode, layer, "input"] = args[0]
                self.dual_state[self.mode, layer, "main_gate"] = output[1]
                self.dual_state[self.mode, layer, kind] = output[-1]
            else:
                self.dual_state[self.mode, layer, kind] = output[0] if kind == "main" else output

    def residual(self, layer, module, args):
        """Patch the main-attention residual, retaining SD3.5's image-only attn2 update."""
        if not self.active or (self.mode == "probe" and self.settings["source_mode"] != "extrapolate"):
            return
        image = args[0]
        extra = 0
        main_residual = image
        if (self.mode, layer, "gate") in self.dual_state:
            gate = self.dual_state.pop((self.mode, layer, "gate"))
            extra = gate[:, None] * self.dual_state.pop((self.mode, layer, "output"))
            original = self.dual_state.pop((self.mode, layer, "input"))
            main_gate = self.dual_state.pop((self.mode, layer, "main_gate"))
            main_output = self.dual_state.pop((self.mode, layer, "main"))
            main_residual = original + main_gate[:, None] * main_output
        self.remember_user(layer, main_residual)
        if self.mode == "probe":
            return
        if self.mode == "anchor":
            self.sources[self.current_anchor][layer] = (main_residual.detach() if self.inline
                                                        else main_residual.detach().cpu())
        elif self.zero_update or self.sources:
            changed = self.patch(layer, main_residual[-self.batch:])
            if isinstance(extra, torch.Tensor):
                changed = changed + extra[-self.batch:]
            result = image.clone()
            mask = (self.winner >= 0).to(image.device)[..., None]
            result[-self.batch:] = torch.where(mask, changed, image[-self.batch:])
            return (result, *args[1:])

    def block_output(self, layer, module, args, output):
        """Patch final image states after the block's MLP and residual updates.

        FLUX single-stream residual editing uses this same native output site.
        """
        if not self.active:
            return
        context, image = output
        self.remember_user(layer, image)
        if self.mode == "probe":
            return
        if self.mode == "anchor":
            self.sources[self.current_anchor][layer] = (image.detach() if self.inline
                                                        else image.detach().cpu())
        else:
            changed = image.clone()
            changed[-self.batch:] = self.patch(layer, image[-self.batch:])
            return context, changed

    def forward(self, *args, **kwargs):
        if not self.active:
            return self.native_forward(*args, **kwargs)
        self.input_call = dict(self.forward_signature.bind(*args, **kwargs).arguments)
        self.text_count = self.input_call["encoder_hidden_states"].shape[1]
        self.votes, self.sources, self.layer_ratios = None, {}, {}
        self.score_sum = None
        self.user_sources = {}
        self.diagnostic_mask_sum = self.diagnostic_score_sum = self.diagnostic_logit_sum = None
        self.diagnostic_layers, self.activation_sums = [], {}
        self.dual_state.clear()
        self.single_text_messages.clear()
        self.forward_calls += 1
        self.mode = "user" if self.inline else "probe"
        output = self.native_forward(**self.with_concepts(self.input_call))
        if self.velocity:
            output = self.velocity_output(output)
        elif not self.inline:
            self.finish_mask()
            if self.triggered.any().item():
                self.capture_sources()
                self.mode = "edit"
                self.forward_calls += 1
                output = self.native_forward(**self.input_call)
        self.mode = "idle"
        if self.settings["trace"]:
            pairs = len(self.settings["pairs"])
            applied = self.winner[:, None] == torch.arange(pairs, device=self.winner.device)[None, :, None]
            layers = len(self.diagnostic_layers)
            payload = {
                "mask": self.masks.reshape(self.batch, pairs, *self.grid).cpu(),
                "vote_fraction": self.vote_fraction.reshape(self.batch, pairs, *self.grid).cpu(),
                "winner": self.winner.reshape(self.batch, *self.grid).cpu(),
                "applied_mask": applied.reshape(self.batch, pairs, *self.grid).cpu(),
                "layer_mask_mean": (self.diagnostic_mask_sum / layers).reshape(self.batch, pairs, *self.grid).cpu(),
                "concept_scores_mean": (self.diagnostic_score_sum / layers).reshape(self.batch, pairs, *self.grid, 3).cpu(),
                "concept_logits_mean": (self.diagnostic_logit_sum / layers).reshape(self.batch, pairs, *self.grid, 3).cpu(),
                "localization_scores_mean": self.localization_scores.reshape(self.batch, pairs, *self.grid, 3).cpu(),
            }
            # Record the actual cutoff or concept winner used before the area gate.
            if self.settings["my_layer_aggre"]:
                payload["decision_threshold"] = self.ratios.new_full(self.ratios.shape, self.settings["eta"]).cpu()
            elif self.settings["my_layer_policy"] and self.settings["mask_rule"] == "mean_target_max":
                payload["decision_threshold"] = self.localization_scores[..., 0].mean(-1).cpu()
            elif not self.settings["my_layer_policy"]:
                if pairs == 1:
                    payload["decision_threshold"] = self.localization_scores[..., 0].mean(-1).cpu()
                else:
                    labels = self.localization_scores.transpose(1, 2).flatten(2).argmax(-1)
                    payload["concept_argmax"] = labels.reshape(self.batch, *self.grid).cpu()
            # Keep only running CPU spatial sums; no per-step activations are retained.
            for name in ("applied_mask", "concept_scores_mean", "localization_scores_mean"):
                values = payload[name].float()
                if name not in self.sample_trace_sums:
                    self.sample_trace_sums[name] = values.clone()
                else:
                    self.sample_trace_sums[name].add_(values)
            activation = torch.stack([self.activation_sums.get(layer, torch.zeros(self.batch, pairs, 6))
                                      for layer in self.editors], dim=2)
            for i, directory in enumerate(self.trace_dirs):
                info = {"step_index": self.step,
                    "timestep": float(self.input_call["timestep"][0].item()),
                    "target_area": self.ratios[i].cpu().tolist(),
                    "triggered": self.triggered[i].cpu().tolist(),
                    "layer_target_area": {key: value[i] for key, value in self.layer_ratios.items()},
                    "pairs": self.settings["pairs"], "eta": self.settings["eta"], "rho": self.settings["rho"],
                    "mask_rule": self.settings["mask_rule"],
                    "my_layer_aggre": self.settings["my_layer_aggre"],
                    "my_layer_policy": self.settings["my_layer_policy"],
                    "aggregation": "layer_vote" if self.settings["my_layer_aggre"] else "soft_mean",
                    "region_policy": (self.settings["mask_rule"] if self.settings["my_layer_policy"] else
                                      "spatial_mean" if pairs == 1 else "concept_argmax"),
                    "softmax_scope": "per_pair" if self.settings["my_layer_policy"] else "all_concepts",
                    "decision_threshold": (payload["decision_threshold"][i].tolist()
                                           if "decision_threshold" in payload else None),
                    "representation": self.settings["representation"],
                    "update_rule": self.settings["update_rule"], "strength": self.settings["strength"],
                    "source_mode": self.settings["source_mode"],
                    "source_user_weight": self.settings["source_user_weight"],
                    "source_anchor_weight": self.settings["source_anchor_weight"],
                    "source_scale": self.settings["source_scale"],
                    "diagnostic_layers": self.diagnostic_layers, "localization_layers": self.localizers}
                tensors = {name: value[i] for name, value in payload.items()}
                tensors["edit_layer_indices"] = torch.tensor(self.editors)
                info["activation_change"] = save_step_diagnostics(
                    directory, self.step, tensors, activation[i], info)
                self.step_traces[i].append(info)
        self.sources.clear()
        self.user_sources.clear()
        self.dual_state.clear()
        self.single_text_messages.clear()
        return output

    def generate(self, records, sampling_config):
        opts, pipe = sampling_config, self.pipe
        self.batch, self.step, self.forward_calls = len(records), 0, 0
        with torch.inference_mode():
            self.prepare_sources(records)
        self.step_traces = [[] for _ in records]
        self.sample_trace_sums.clear()
        self.trace_dirs = []
        for record in records:
            image = Path(record["image"])
            root = image.parent.parent.parent / "traces"
            if self.settings["trace_root"]:
                root = (Path(self.settings["trace_root"]) / self.config["model_name"] /
                        self.config["task"] / self.config["case_id"] / self.config["method"])
            self.trace_dirs.append(root / image.parent.name / image.stem)
        channels = pipe.transformer.config.in_channels // 4 if self.flux else pipe.transformer.config.in_channels
        shape = (channels, opts["height"] // pipe.vae_scale_factor, opts["width"] // pipe.vae_scale_factor)
        latents = torch.stack([initial_noise(shape, record["seed"], dtype=self.dtype, device="cpu")
                               for record in records])
        if self.flux:
            latents = pipe._pack_latents(latents, len(records), *shape)

        def step_end(pipeline, step, timestep, callback_kwargs):
            self.step = step + 1
            return {}

        call = dict(prompt=[r["prompt"] for r in records], latents=latents.to(self.device),
                    generator=[torch.Generator("cpu").manual_seed(r["seed"]) for r in records],
                    height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                    guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"],
                    num_images_per_prompt=1, callback_on_step_end=step_end)
        if self.flux:
            call["true_cfg_scale"] = opts.get("true_cfg_scale", 1.0)
            if opts.get("negative_prompt") is not None:
                call["negative_prompt"] = [opts["negative_prompt"]] * len(records)
        else:
            call["negative_prompt"] = [opts.get("negative_prompt") or ""] * len(records)
        started = time.perf_counter()
        self.active = True
        try:
            with torch.inference_mode():
                images = pipe(**call).images
        finally:
            self.active, self.mode = False, "idle"
        if self.settings["trace"]:
            for i, directory in enumerate(self.trace_dirs):
                step_count = len(self.step_traces[i])
                save_sample_diagnostics(directory, {
                    "applied_mask_mean": self.sample_trace_sums["applied_mask"][i] / step_count,
                    "concept_scores_mean": self.sample_trace_sums["concept_scores_mean"][i] / step_count,
                    "localization_scores_mean": self.sample_trace_sums["localization_scores_mean"][i] / step_count,
                }, self.settings["pairs"], step_count, mask_rule=self.settings["mask_rule"],
                    my_layer_aggre=self.settings["my_layer_aggre"],
                    my_layer_policy=self.settings["my_layer_policy"],
                    update_rule=self.settings["update_rule"],
                    localization_layers=self.localizers, diagnostic_layers=self.diagnostic_layers)
                write_json(directory / "metadata.json", {
                    "record": records[i], "settings": self.settings, "sites": self.sites,
                    "sampling": opts, "steps": self.step_traces[i],
                    "forward_calls": self.forward_calls, "batch_size": self.batch,
                    "batch_wall_seconds": time.perf_counter() - started,
                    "runtime_info": self.runtime_info})
        return images

    def close(self):
        self.active = False
        self.pipe.transformer.forward = self.native_forward
        for module, processor in self.processors:
            module.set_processor(processor)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for block, forward in self.block_forwards:
            block.forward = forward
        self.block_forwards.clear()
        self.anchor_conditions.clear()
        self.source_groups.clear()
        super().close()
