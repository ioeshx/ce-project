"""Frozen text encoders with one trainable CLIP input token per encoder."""
import torch
import torch.nn.functional as F

from benchmark.adapters import release_memory
from benchmark.execution import runtime_device


def cosine_scores(features, prototypes):
    """Give each CLIP the same weight, regardless of its feature dimension."""
    return torch.stack([F.normalize(q.float(), dim=-1) @ F.normalize(p.float(), dim=-1).T
                        for q, p in zip(features, prototypes)]).mean(0)


class TextEncoder:
    """Load CLIP alone for classification; load T5 only for full conditions."""

    def __init__(self, config, pipe=None):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self.config, self.pipe = config, pipe
        self.flux = config["model_name"].startswith("flux1_")
        opts = config["sampling"]
        self.device = torch.device(runtime_device(opts["device"]))
        self.dtype = getattr(torch, opts["dtype"] if opts["dtype"] != "auto" else
                             ("bfloat16" if self.device.type == "cuda" else "float32"))
        self.load_options = dict(local_files_only=opts["local_files_only"], revision=opts.get("model_revision"))
        self.encoders, self.tokenizers, self.projections = [], [], []
        for index in range(1 if self.flux else 2):
            suffix = "" if index == 0 else "_2"
            cls = CLIPTextModel if self.flux else CLIPTextModelWithProjection
            encoder = (getattr(pipe, f"text_encoder{suffix}") if pipe is not None else
                       cls.from_pretrained(config["base_model"], subfolder=f"text_encoder{suffix}",
                                           torch_dtype=self.dtype, **self.load_options).to(self.device))
            tokenizer = (getattr(pipe, f"tokenizer{suffix}") if pipe is not None else
                         CLIPTokenizer.from_pretrained(config["base_model"], subfolder=f"tokenizer{suffix}",
                                                       **self.load_options))
            encoder.eval().requires_grad_(False)
            self.encoders.append(encoder)
            self.tokenizers.append(tokenizer)
            self.projections.append(getattr(encoder, "text_projection", None))
        # FLUX stores the CLIP backbone without its pretrained contrastive head.
        # It is needed for cosine detection, but never changes FLUX conditioning.
        self.projection_source = config.get("text", {}).get("clip_projection_model", "openai/clip-vit-large-patch14")
        self.t5, self.t5_tokenizer, self.fixed_t5 = None, None, {}

    def project(self, index, pooled):
        if self.projections[index] is None:
            from transformers import CLIPTextModelWithProjection
            source = CLIPTextModelWithProjection.from_pretrained(
                self.projection_source, torch_dtype=self.dtype,
                local_files_only=self.load_options["local_files_only"])
            self.projections[index] = source.text_projection.to(self.device).eval().requires_grad_(False)
        projection = self.projections[index]
        return projection(pooled.to(device=projection.weight.device, dtype=projection.weight.dtype))

    def clip(self, texts, tokens=None, template="{}", *, project_features=True):
        sequences, pooled, features = [], [], []
        for index, (encoder, tokenizer) in enumerate(zip(self.encoders, self.tokenizers)):
            device = self.device
            if tokens is None:
                ids = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").input_ids.to(device)
                output = encoder(input_ids=ids, output_hidden_states=True, return_dict=True)
            else:
                prefix, suffix = template.split("{}")
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                placeholder = tokenizer.encode("x", add_special_tokens=False)[0]
                content = [tokenizer.bos_token_id, *prefix_ids, placeholder, *suffix_ids, tokenizer.eos_token_id]
                ids = torch.tensor([content + [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(content))],
                                   device=device)
                position = len(prefix_ids) + 1

                def replace_token(module, inputs, embeddings):
                    result = embeddings.clone()
                    result[:, position] = tokens[index].to(device=embeddings.device, dtype=embeddings.dtype)
                    return result

                handle = encoder.get_input_embeddings().register_forward_hook(replace_token)
                try:
                    output = encoder(input_ids=ids, output_hidden_states=True, return_dict=True)
                finally:
                    handle.remove()
            native_pool = output.pooler_output if self.flux else output.text_embeds
            pooled.append(native_pool)
            sequences.append(output.hidden_states[-2])
            feature = self.project(index, native_pool) if self.flux and project_features else native_pool
            features.append(feature.to(self.device))
        return sequences, pooled, features

    @torch.no_grad()
    def features(self, texts):
        return tuple(self.clip(texts)[2])

    def split_features(self, pooled):
        sizes = [encoder.config.hidden_size if self.flux else encoder.config.projection_dim for encoder in self.encoders]
        parts = pooled.split(sizes, dim=-1)
        return tuple(self.project(i, part) if self.flux else part for i, part in enumerate(parts))

    @torch.no_grad()
    def t5_encode(self, texts):
        if self.t5 is None:
            from transformers import T5EncoderModel, T5TokenizerFast
            suffix = "_2" if self.flux else "_3"
            self.t5 = (getattr(self.pipe, f"text_encoder{suffix}") if self.pipe is not None else
                       T5EncoderModel.from_pretrained(self.config["base_model"], subfolder=f"text_encoder{suffix}",
                                                     torch_dtype=self.dtype, **self.load_options).to(self.device))
            self.t5_tokenizer = (getattr(self.pipe, f"tokenizer{suffix}") if self.pipe is not None else
                                 T5TokenizerFast.from_pretrained(self.config["base_model"],
                                                                subfolder=f"tokenizer{suffix}", **self.load_options))
            self.t5.eval().requires_grad_(False)
        ids = self.t5_tokenizer(texts, padding="max_length", truncation=True,
                               max_length=self.config["sampling"]["max_sequence_length"],
                               return_tensors="pt").input_ids.to(self.device)
        return self.t5(input_ids=ids, return_dict=False)[0].to(self.device)

    def combine(self, sequences, pooled, t5):
        pooled = torch.cat([value.to(self.device) for value in pooled], dim=-1)
        if self.flux:
            return t5, pooled
        sequence = torch.cat([value.to(self.device) for value in sequences], dim=-1)
        sequence = F.pad(sequence, (0, t5.shape[-1] - sequence.shape[-1]))
        return torch.cat((sequence, t5), dim=1), pooled

    @torch.no_grad()
    def encode(self, texts, *, clip_only=False):
        if clip_only:
            return self.features(texts)
        sequence, pooled, _ = self.clip(texts, project_features=False)
        return self.combine(sequence, pooled, self.t5_encode(texts))

    def initialize(self, entry, stage, space, template):
        entry[f"{stage}_space"] = space
        entry[f"{stage}_template"] = template
        fixed = entry["target"] if stage == "detect" else ""
        entry[f"{stage}_fixed_text"] = fixed
        if space == "soft":
            tokens = []
            for encoder, tokenizer in zip(self.encoders, self.tokenizers):
                ids = tokenizer.encode(fixed, add_special_tokens=False) or [tokenizer.eos_token_id]
                tokens.append(encoder.get_input_embeddings().weight[ids].detach().float().mean(0, keepdim=True).cpu())
            entry[f"{stage}_soft_tokens"] = tokens
        elif stage == "detect":
            with torch.no_grad():
                pooled = self.clip([fixed], project_features=False)[1]
                entry["detect_pooled"] = torch.cat([pool.to(self.device) for pool in pooled], dim=-1).float().cpu()
        else:
            sequence, pooled = self.encode([fixed])
            entry.update(erase_sequence=sequence.float().cpu(), erase_pooled=pooled.float().cpu())

    def parameters(self, entry, stage):
        keys = [f"{stage}_soft_tokens"] if entry[f"{stage}_space"] == "soft" else (
            ["detect_pooled"] if stage == "detect" else ["erase_sequence", "erase_pooled"])
        parameters = []
        for key in keys:
            if isinstance(entry[key], list):
                entry[key] = [torch.nn.Parameter(t.to(self.device).float().clone()) for t in entry[key]]
                parameters.extend(entry[key])
            else:
                entry[key] = torch.nn.Parameter(entry[key].to(self.device).float().clone())
                parameters.append(entry[key])
        return parameters

    def condition(self, entry, stage, *, features_only=False):
        fixed = entry[f"{stage}_fixed_text"]
        if entry[f"{stage}_space"] == "soft":
            sequence, pooled, features = self.clip(
                None, tokens=entry[f"{stage}_soft_tokens"], template=entry[f"{stage}_template"],
                project_features=features_only)
            if features_only:
                return tuple(features)
            if fixed not in self.fixed_t5:
                self.fixed_t5[fixed] = self.t5_encode([fixed])
            return self.combine(sequence, pooled, self.fixed_t5[fixed])
        pooled = entry[f"{stage}_pooled"].to(self.device)
        if features_only:
            return self.split_features(pooled)
        if f"{stage}_sequence" not in entry:
            entry[f"{stage}_sequence"] = self.encode([fixed])[0].float().cpu()
        return entry[f"{stage}_sequence"].to(self.device), pooled

    @torch.no_grad()
    def prepare_detector(self, entries, options):
        self.entries, self.options = entries, options
        if options["similarity"] == "cosine":
            features = [self.condition(entry, "detect", features_only=True) for entry in entries.values()]
            self.prototypes = tuple(torch.cat([feature[i] for feature in features]) for i in range(len(self.encoders)))
        else:
            self.prototypes = [self.condition(entry, "detect") for entry in entries.values()]

    @torch.no_grad()
    def detect(self, texts, condition=None):
        targets = list(self.entries)
        global_detection = self.options.get("all_detect_prompts", False)
        if self.options["similarity"] == "cosine":
            scores = cosine_scores(self.features(texts), self.prototypes)
            best = scores.argmax(1)
        else:
            sequence, pooled = self.encode(texts) if condition is None else condition
            count = sequence[0].numel() + pooled[0].numel()
            scores = torch.stack([((sequence.float() - seq.float()).square().flatten(1).sum(1)
                                  + (pooled.float() - pool.float()).square().sum(1)).div(count).sqrt()
                                  for seq, pool in self.prototypes], dim=1)
            best = scores.argmin(1)
        if self.options["similarity"] == "cosine":
            threshold = self.options.get("tau")
            thresholds = [entry["tau"] if threshold is None else threshold for entry in self.entries.values()]
            threshold_tensor = torch.tensor(thresholds, device=scores.device, dtype=scores.dtype)
            hits = scores >= threshold_tensor
            thresholds = threshold_tensor.tolist()
            best = scores.masked_fill(~hits, -torch.inf).argmax(1)
        elif global_detection:
            # Global L2 retains its nearest-prototype decision without a distance cutoff.
            thresholds = [None] * len(targets)
            hits = torch.zeros_like(scores, dtype=torch.bool)
            hits.scatter_(1, best[:, None], True)
        else:
            thresholds = [scores.new_tensor(self.options["l2_threshold"]).item()] * len(targets)
            hits = scores <= thresholds[0]
            best = scores.masked_fill(~hits, torch.inf).argmin(1)
        selected = hits.clone()
        if self.options["multi_concept"] == "top1":
            selected.zero_()
            selected.scatter_(1, best[:, None], hits.any(1, keepdim=True))
        erase_targets = set(self.options.get("erase_targets", targets))
        decisions = []
        for row, matched, chosen in zip(scores, hits, selected):
            predicted = [target for target, flag in zip(targets, chosen.tolist()) if flag]
            decisions.append({"scores": dict(zip(targets, row.tolist())),
                              "thresholds": dict(zip(targets, thresholds)),
                              "threshold_applied": self.options["similarity"] == "cosine" or not global_detection,
                              "matched": [target for target, flag in zip(targets, matched.tolist()) if flag],
                              "predicted": predicted,
                              "selected": [target for target in predicted if target in erase_targets]})
        return decisions

    def close(self):
        self.encoders, self.projections, self.fixed_t5 = [], [], {}
        self.t5, self.pipe = None, None
        release_memory()
