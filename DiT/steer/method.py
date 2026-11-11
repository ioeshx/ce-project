"""Offline and online steering of SD3 token states and global pooled text embeddings."""
from __future__ import annotations

import copy
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.common import concept_key, digest, read_json, slug, write_json
from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise


def construct_vectors(positive_means, negative_means, *, differences=None, construction=None, seed=0):
    """Construct unit directions; differences has shape [pairs, ..., channels].

    PCA uses uncentered differences: centering would remove their shared offset.
    All operations act independently at each layer, step and optional position.
    """
    construction = construction or {"method": "mean", "normalize_differences": False}
    method = construction["method"]
    difference = positive_means - negative_means
    if method == "mean" and not construction["normalize_differences"]:
        return difference.div_(difference.norm(dim=-1, keepdim=True).clamp_min_(1e-8))

    samples = differences.float()
    if construction["normalize_differences"]:
        samples = F.normalize(samples, dim=-1, eps=1e-8)
        difference = samples.mean(dim=0)

    if method == "mean":
        direction = difference
    elif method == "robust":
        # Median of means: the same seeded partition is used at every position.
        order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed))
        groups = torch.tensor_split(order, construction["groups"])
        group_means = torch.stack([samples[index].mean(dim=0) for index in groups])
        direction = group_means.mean(dim=0)
        active = torch.ones(direction.shape[:-1], dtype=torch.bool, device=direction.device)
        for _ in range(construction["max_iter"]):
            distance = (group_means - direction).norm(dim=-1, keepdim=True).clamp_min(1e-8)
            weights = distance.reciprocal()
            updated = (weights * group_means).sum(dim=0) / weights.sum(dim=0)
            converged = ((updated - direction).norm(dim=-1) <=
                         construction["tol"] * direction.norm(dim=-1).clamp_min(1.0))
            direction = torch.where(active.unsqueeze(-1), updated, direction)
            active = active & ~converged
            if not active.any():
                break
    else:
        # [..., pairs, channels] -> channel directions, ordered by singular value.
        _, singular_values, vh = torch.linalg.svd(samples.movedim(0, -2), full_matrices=False)
        if method == "pca":
            direction = vh[..., 0, :]
            sign = torch.where((direction * difference).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
            direction = direction * sign * (singular_values[..., :1] > 1e-8)
        else:  # pca-filter: project the mean onto the leading k channel directions.
            basis = vh[..., :construction["rank"], :]
            coordinates = (basis * difference.unsqueeze(-2)).sum(dim=-1, keepdim=True)
            direction = (coordinates * basis).sum(dim=-2)
    return F.normalize(direction, dim=-1, eps=1e-8)


def construct_from_pairs(positive, negative, directory, pairs_by_concept, construction, seed):
    """Read saved differences in bounded chunks, including per-position tensors."""
    vectors = torch.empty_like(positive)
    channels = positive.shape[-1]
    for concept, entry in enumerate(pairs_by_concept):
        samples = [torch.load(Path(directory) / slug(entry["target"]) / f"pair-{index + 1:04d}.pt",
                              map_location="cpu", weights_only=True, mmap=True).reshape(-1, channels)
                   for index in range(len(entry["pairs"]))]
        output = vectors[concept].reshape(-1, channels)
        positive_flat = positive[concept].reshape(-1, channels)
        negative_flat = negative[concept].reshape(-1, channels)
        # About 32 MB of input differences per chunk, independent of patch count.
        chunk_size = max(1, 8_000_000 // (len(samples) * channels))
        for start in range(0, len(output), chunk_size):
            end = start + chunk_size
            differences = torch.stack([sample[start:end] for sample in samples])
            output[start:end] = construct_vectors(positive_flat[start:end], negative_flat[start:end],
                differences=differences, construction=construction, seed=seed)
        print(f"[construct:{entry['target']}] {construction}", flush=True)
    return vectors


def steer_output(output, vectors, beta=-2.0, clipping=True, *, score_gate=True, scores=None):
    """Apply h <- h + beta * max(dot(h, v), 0) * v, in concept order.

    Negative beta suppresses the positive-minus-negative concept direction.
    Without score gating, add beta * v directly and do not compute dot products.
    Optional scores records the raw dot products, before clipping or this edit.
    """
    current = output
    for index, vector in enumerate(vectors.to(device=output.device, dtype=torch.float32)):
        score = 1.0
        if score_gate:
            score = (current.float() * vector).sum(dim=-1, keepdim=True)
            if scores is not None:
                scores[index].copy_(score.squeeze(-1).detach())
            if clipping:
                score = score.clamp_min(0)
        current = (current.float() + beta * score * vector).to(output.dtype)
    return current


def prompt_pairs(group, target, anchor, count=None, *, mode="casteer", repeats=1, seed=0):
    """Build matched text pairs, using the same context on both sides.

    bare uses literal concept names once. subject adds short wrappers. matched
    keeps an explicit scene or painting subject. An empty anchor removes the
    entire concept suffix; bare then has an empty negative prompt. With multiple
    repeats, each returned pair records its seed for matched positive/negative
    sampling. A single repeat retains the legacy metadata representation.
    """
    if mode == "bare":
        pairs = [{"positive": target, "negative": anchor}]

    elif mode in {"subject", "matched"}:
        templates = read_json(Path(__file__).with_name("prompts") / f"{mode}.json")
        template = templates[group if group in {"i2p", "style"} else "ip_object"]
        suffix = template["suffix"]
        pairs = [{"positive": context + suffix.format(concept=target),
                  "negative": context + suffix.format(concept=anchor) if anchor else context}
                 for context in template["contexts"]]
    elif group == "i2p":
        # Keep the original CASteer templates and ordering for existing caches.
        people = ["a girl", "a boy", "two men", "two women", "two people", "a man", "a woman",
                  "an old man", "an old woman", "boys", "girls", "men", "women", "group of people", "a human"]
        scenes = ["", "gloomy image", "zoomed in", "talking", "on the street", "in a strange pose",
                  "realism", "colorful background", "on a beach", "playing guitar", "enjoying nature",
                  "smiling", "in a futuristic spaceship", "with kittens"]
        stems = [f"{person} {scene}" for person in people for scene in scenes]
        pairs = [{"positive": f"{stem}, {target}", "negative": f"{stem}, {anchor}" if anchor else stem.rstrip()}
                 for stem in stems]
    else:
        # CASteer： {imagenet class} + {anchor/target} style/with {anchor/target}
        classes = Path(__file__).with_name("imagenet_classes.txt").read_text().splitlines()
        suffix = f", {target} style" if group == "style" else f" with {target}"
        negative_suffix = (f", {anchor} style" if group == "style" else f" with {anchor}") if anchor else ""
        pairs = [{"positive": name + suffix, "negative": name + negative_suffix} for name in classes]
        count = count or 50
    pairs = pairs[:count]
    if repeats > 1:
        pairs = [{**pair, "seed": seed + repeat} for pair in pairs for repeat in range(repeats)]
    return pairs


class SD3Steering:
    """Own the hooks and per-step state shared by collection and intervention.

    text_value edits V_text at add_v_proj, before attention reads it.
    text_token and text_token_output edit the full text attention result before
    and after to_add_out, respectively, before text gating and residual addition.
    text_to_image edits only W_out * A_image,text * V_text, retaining the native
    joint softmax denominator, image-value contribution and projection bias.
    The three block sites select image/text/both streams at a residual or MLP
    boundary. Pooled text steering is an independent, supplementary hook.
    """

    def __init__(self, transformer, steps, *, cfg, vectors=None, beta=-2.0, clipping=True,
                 steer_target="text_to_image", average_patches=True, score_gate=True,
                 steer_noise=None, noise_seed=0, target_count=1, trace_score=False,
                 steer_token="image", steer_pool=False, pool_vectors=None):
        self.vectors, self.beta, self.clipping = vectors, beta, clipping
        self.collecting = vectors is None and steer_noise is None
        self.score_gate, self.trace_score = score_gate, trace_score and score_gate
        self.steer_noise, self.noise_seed, self.target_count = steer_noise, noise_seed, target_count
        self.noise_vectors, self.noise_generators, self.score_maps = {}, {}, None
        self.steer_token, self.steer_pool, self.pool_vectors = steer_token, steer_pool, pool_vectors
        self.pool_sum, self.pool_scores = None, None
        self.cfg = cfg
        self.steer_target, self.average_patches = steer_target, average_patches
        self.step, self.samples = -1, 0
        self.handles = [transformer.register_forward_pre_hook(self._next_step)]
        layers = transformer.transformer_blocks
        self.steps, self.layer_count = steps, len(layers)
        self.token_counts = [(0, 0)] * len(layers)
        self.original_forwards = []
        # Position count comes from the actual V_text or image-patch tensor.
        self.sums = None
        if steer_pool:
            self.handles.append(transformer.time_text_embed.text_embedder.register_forward_pre_hook(self._pool_embedding))
        for index, block in enumerate(layers):
            if steer_target in {"attention_residual", "mlp_output", "block_output"}:
                self.original_forwards.append((block, block.forward))
                block.forward = partial(self._block_forward, block, index)
            else:
                self._register_block(block.attn, index)

    def _block_forward(self, block, layer, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs=None):
        """Native SD3 block equations, with both streams available at each steering site."""
        from diffusers.models.attention import _chunked_feed_forward

        kwargs = joint_attention_kwargs or {}
        normalized = block.norm1(hidden_states, emb=temb)
        image_norm, gate_attn, shift, scale, gate_mlp = normalized[:5]
        if block.context_pre_only:
            text_norm = block.norm1_context(encoder_hidden_states, temb)
        else:
            text_norm, text_gate_attn, text_shift, text_scale, text_gate_mlp = block.norm1_context(
                encoder_hidden_states, emb=temb)
        image_attn, text_attn = block.attn(hidden_states=image_norm, encoder_hidden_states=text_norm, **kwargs)
        image = hidden_states + gate_attn[:, None] * image_attn
        if block.use_dual_attention:
            image = image + normalized[6][:, None] * block.attn2(hidden_states=normalized[5], **kwargs)
        # Text and image MLPs are independent; form both residual streams before either MLP.
        text = None if block.context_pre_only else encoder_hidden_states + text_gate_attn[:, None] * text_attn
        if self.steer_target == "attention_residual":
            image, text = self._steer_tokens(image, text, layer)

        image_norm = block.norm2(image) * (1 + scale[:, None]) + shift[:, None]
        image_mlp = (block.ff(image_norm) if block._chunk_size is None else
                     _chunked_feed_forward(block.ff, image_norm, block._chunk_dim, block._chunk_size))
        text_mlp = None
        if text is not None:
            text_norm = block.norm2_context(text) * (1 + text_scale[:, None]) + text_shift[:, None]
            text_mlp = (block.ff_context(text_norm) if block._chunk_size is None else
                        _chunked_feed_forward(block.ff_context, text_norm, block._chunk_dim, block._chunk_size))
        if self.steer_target == "mlp_output":
            image_mlp, text_mlp = self._steer_tokens(image_mlp, text_mlp, layer)
        image = image + gate_mlp[:, None] * image_mlp
        if text is not None:
            text = text + text_gate_mlp[:, None] * text_mlp
        if self.steer_target == "block_output":
            image, text = self._steer_tokens(image, text, layer)
        return text, image

    def _steer_tokens(self, image, text, layer):
        self.token_counts[layer] = (image.shape[1], text.shape[1] if text is not None else 0)
        if self.steer_token == "image":
            return self._collect_or_steer(image, layer), text
        if self.steer_token == "text":
            return image, self._collect_or_steer(text, layer) if text is not None else None
        joined = image if text is None else torch.cat((image, text), dim=1)
        joined = self._collect_or_steer(joined, layer)
        return joined[:, :image.shape[1]], joined[:, image.shape[1]:] if text is not None else None

    def _pool_embedding(self, module, args):
        features = args[0]
        conditional = slice(features.shape[0] // 2, None) if self.cfg else slice(None)
        current = features[conditional, None, :]
        if self.collecting:
            if self.step == 0:
                statistic = current.float().mean(dim=(0, 1)).cpu()
                self.pool_sum = statistic if self.pool_sum is None else self.pool_sum + statistic
            return
        vectors = self._noise_directions(current, site="pool") if self.steer_noise else self.pool_vectors
        scores = None
        if self.trace_score:
            if self.pool_scores is None:
                self.pool_scores = torch.empty(len(vectors), self.steps, 1, len(current), 1,
                                                device=current.device, dtype=torch.float32)
            scores = self.pool_scores[:, self.step, 0]
        updated = features.clone()
        updated[conditional] = steer_output(current, vectors, self.beta, self.clipping,
                                             score_gate=self.score_gate, scores=scores).squeeze(1)
        return (updated, *args[1:])

    def _next_step(self, module, args):
        self.step += 1
        if self.step == 0:
            self.samples += 1

    def _register_block(self, attn, layer):
        if self.steer_target in {"text_token", "text_token_output"}:
            # The final context_pre_only block discards its text output and has
            # no to_add_out. Keep its statistics zero, as for text block sites.
            if attn.context_pre_only:
                return

            def steer_text(features):
                self.token_counts[layer] = (0, features.shape[1])
                return self._collect_or_steer(features, layer)

            if self.steer_target == "text_token":
                def before_projection(module, args):
                    updated = steer_text(args[0])
                    # Preserve the native strided text view on no-op runs.
                    if self.collecting or self.beta == 0:
                        return
                    return (updated, *args[1:])

                self.handles.append(attn.to_add_out.register_forward_pre_hook(before_projection))
            else:
                def after_projection(module, args, output):
                    return steer_text(output)

                self.handles.append(attn.to_add_out.register_forward_hook(after_projection))
            return

        if self.steer_target == "text_value":
            def steer_values(module, args, output):
                return self._collect_or_steer(output, layer)

            self.handles.append(attn.add_v_proj.register_forward_hook(steer_values))
            return

        # One closure per block keeps its cross contribution until to_out runs.
        state = {}

        def extract(module, args, kwargs):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            context = kwargs["encoder_hidden_states"] if "encoder_hidden_states" in kwargs else args[1]
            batch, heads = hidden.shape[0], attn.heads
            head_dim = attn.to_q.out_features // heads
            query = attn.to_q(hidden).view(batch, -1, heads, head_dim).transpose(1, 2)
            image_key = attn.to_k(hidden).view(batch, -1, heads, head_dim).transpose(1, 2)
            text_key = attn.add_k_proj(context).view(batch, -1, heads, head_dim).transpose(1, 2)
            text_value = attn.add_v_proj(context).view(batch, -1, heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                image_key = attn.norm_k(image_key)
            if attn.norm_added_k is not None:
                text_key = attn.norm_added_k(text_key)
            key = torch.cat((image_key, text_key), dim=2)
            value = torch.cat((torch.zeros_like(image_key), text_value), dim=2)
            cross = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            state["cross"] = cross.transpose(1, 2).reshape(batch, -1, heads * head_dim)

        def steer_contribution(module, args, output):
            # Exclude projection bias: it belongs to the full native output.
            cross = F.linear(state.pop("cross"), module.weight, bias=None)
            updated = self._collect_or_steer(cross, layer)
            if self.collecting:
                return output
            return output + (updated.float() - cross.float()).to(output.dtype)

        self.handles.append(attn.register_forward_pre_hook(extract, with_kwargs=True))
        self.handles.append(attn.to_out[0].register_forward_hook(steer_contribution))

    def _collect_or_steer(self, features, layer):
        """features: [batch, positions, channels]; vectors may retain positions."""
        conditional = slice(features.shape[0] // 2, None) if self.cfg else slice(None)
        current = features[conditional]
        if self.collecting:
            statistic = current.float().mean(dim=0)  # Always average samples.
            if self.average_patches:
                statistic = statistic.mean(dim=0)
            if self.sums is None:
                self.sums = torch.zeros(self.steps, self.layer_count, *statistic.shape, dtype=torch.float32)
            if self.average_patches:
                self.sums[self.step, layer] += statistic.cpu()
            else:
                self.sums[self.step, layer, :statistic.shape[0]] += statistic.cpu()
            return features
        vectors = (self._noise_directions(current) if self.steer_noise else
                   self.vectors[:, self.step, layer])
        if not self.steer_noise and not self.average_patches:
            vectors = vectors[:, :current.shape[1]]
        scores = None
        if self.trace_score:
            if self.score_maps is None:
                self.score_maps = torch.zeros(len(vectors), self.steps, self.layer_count,
                                              *current.shape[:2], device=current.device, dtype=torch.float32)
            scores = self.score_maps[:, self.step, layer, :, :current.shape[1]]
        updated = features.clone()
        updated[conditional] = steer_output(current, vectors, self.beta, self.clipping,
                                             score_gate=self.score_gate, scores=scores)
        return updated

    def _noise_directions(self, current, *, site="tokens"):
        """Use unit Gaussian directions and a separate RNG from diffusion noise."""
        positions = 1 if self.average_patches else current.shape[1]
        shape = (self.target_count, positions, current.shape[-1])
        seeds = self.noise_seeds if site == "tokens" else [int(digest([seed, site])[:16], 16) % 2**63 for seed in self.noise_seeds]
        if self.steer_noise == "all":
            for seed in seeds:
                if (site, seed) not in self.noise_vectors:
                    generator = torch.Generator("cpu").manual_seed(seed)
                    self.noise_vectors[site, seed] = F.normalize(torch.randn(shape, generator=generator), dim=-1).to(current.device)
            return torch.stack([self.noise_vectors[site, seed][:, :positions] for seed in seeds], dim=1)
        if site not in self.noise_generators:
            self.noise_generators[site] = [torch.Generator(current.device).manual_seed(seed) for seed in seeds]
        # One RNG per sample keeps batch size and resumed sampling from changing its noise.
        noise = torch.stack([torch.randn(shape, device=current.device, generator=generator)
                             for generator in self.noise_generators[site]], dim=1)
        return F.normalize(noise, dim=-1)

    def reset(self, *, clear_statistics=False, records=(), noise_groups=()):
        """Start another sampling call; optionally clear collected statistics."""
        self.step = -1
        self.score_maps, self.pool_scores, self.noise_generators = None, None, {}
        self.noise_seeds = []
        if self.steer_noise:
            for record, group in zip(records, noise_groups):
                identity = [self.noise_seed, group]
                if self.steer_noise == "every":
                    identity.extend((record["sample_id"], record["seed"]))
                self.noise_seeds.append(int(digest(identity)[:16], 16) % 2**63)
        if clear_statistics:
            self.sums, self.samples = None, 0
            self.pool_sum = None

    def close(self):
        for block, forward in self.original_forwards:
            block.forward = forward
        self.original_forwards.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def fit(checkpoint, settings, config, statistics_checkpoint=None):
    """Collect means and optional pair differences, or reconstruct cached data.

    Native sampling stops at latents: fitting neither decodes images nor uses
    an output classifier. The same noise seed is used on both sides of a pair.
    """
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    construction = settings.get("construction", {"method": "mean", "normalize_differences": False})
    paired = construction["method"] != "mean" or construction["normalize_differences"]
    steer_pool = settings.get("steer_pool", False)
    if statistics_checkpoint is not None:
        # Reconstruct another direction without loading a diffusion model.
        source = Path(statistics_checkpoint)
        state = torch.load(source / "steering.pt", map_location="cpu", weights_only=True, mmap=True)
        metadata = read_json(source / "metadata.json")
        result = {}
        for prefix in (["", "pool_"] if steer_pool else [""]):
            positive, negative = state[prefix + "positive_means"], state[prefix + "negative_means"]
            result[prefix + "vectors"] = (construct_from_pairs(positive, negative, metadata[prefix + "paired_differences"],
                metadata["prompt_pairs"], construction, settings["seed"]) if paired else construct_vectors(positive, negative))
            result.update({prefix + "positive_means": positive, prefix + "negative_means": negative})
        torch.save(result, checkpoint / "steering.pt")
        metadata.update(settings=settings, statistics_checkpoint=str(source))
        write_json(checkpoint / "metadata.json", metadata)
        return

    torch.manual_seed(settings["seed"])
    config = copy.deepcopy(config)
    config["adapter"] = {"kind": "original"}
    sampler = DiTAdapter(config)
    pipe, opts = sampler.pipe, sampler.config["sampling"]
    # Preserve existing token checkpoint keys; supplementary pooled tensors use pool_ keys.
    prefixes = ["", "pool_"] if steer_pool else [""]
    means = {prefix: {"positive": [], "negative": []} for prefix in prefixes}
    pairs_by_concept = []
    shape = (pipe.transformer.config.in_channels, opts["height"] // pipe.vae_scale_factor,
             opts["width"] // pipe.vae_scale_factor)
    try:
        for target, anchor in zip(settings["targets"], settings["anchors"]):
            pairs = prompt_pairs(settings["group"], target, anchor, settings["calibration_prompts"],
                                 mode=settings.get("prompt_mode", "casteer"),
                                 repeats=settings.get("prompt_mode_num", 1), seed=settings["seed"])
            pairs_by_concept.append({"target": target, "anchor": anchor, "pairs": pairs})
            if paired:
                for prefix in prefixes:
                    (checkpoint / (prefix + "paired_differences") / slug(target)).mkdir(parents=True, exist_ok=True)
            for side in ("positive", "negative"):
                hooks = SD3Steering(pipe.transformer, opts["steps"], cfg=opts["guidance_scale"] > 1,
                    steer_target=settings["steer_target"], average_patches=settings["average_patches"],
                    steer_token=settings.get("steer_token", "image"), steer_pool=steer_pool)
                try:
                    total = {}
                    for index, pair in enumerate(pairs):
                        pair_seed = pair.get("seed", settings["seed"])
                        hooks.reset(clear_statistics=paired)
                        latent = initial_noise(shape, pair_seed, dtype=sampler.dtype, device="cpu")
                        pipe(prompt=[pair[side]], negative_prompt=[opts.get("negative_prompt") or ""],
                             latents=latent.unsqueeze(0).to(sampler.device),
                             generator=torch.Generator("cpu").manual_seed(pair_seed),
                             height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                             guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"],
                             output_type="latent")
                        if paired:
                            for prefix in prefixes:
                                statistic = hooks.pool_sum if prefix else hooks.sums
                                total[prefix] = total[prefix] + statistic if prefix in total else statistic.clone()
                                pair_directory = checkpoint / (prefix + "paired_differences") / slug(target)
                                positive_path = pair_directory / f"positive-{index + 1:04d}.pt"
                                if side == "positive":
                                    torch.save(statistic, positive_path)
                                else:
                                    positive_sample = torch.load(positive_path, map_location="cpu", weights_only=True, mmap=True)
                                    torch.save(positive_sample - statistic, pair_directory / f"pair-{index + 1:04d}.pt")
                                    del positive_sample
                                    positive_path.unlink()
                        print(f"[fit:{target}:{side}] {index + 1}/{len(pairs)} seed={pair_seed}", flush=True)
                    for prefix in prefixes:
                        statistic = hooks.pool_sum if prefix else hooks.sums
                        means[prefix][side].append(total[prefix] / len(pairs) if paired else statistic / hooks.samples)
                    token_counts = hooks.token_counts
                finally:
                    hooks.close()
                del hooks, total
        state = {}
        for prefix, values in means.items():
            positive, negative = torch.stack(values["positive"]), torch.stack(values["negative"])
            vectors = (construct_from_pairs(positive, negative, checkpoint / (prefix + "paired_differences"),
                        pairs_by_concept, construction, settings["seed"]) if paired else construct_vectors(positive, negative))
            state.update({prefix + "vectors": vectors, prefix + "positive_means": positive, prefix + "negative_means": negative})
        torch.save(state, checkpoint / "steering.pt")
        write_json(checkpoint / "metadata.json", {
            "settings": settings, "sampling": opts, "prompt_pairs": pairs_by_concept,
            "shape": ("concept, step, layer, channel" if settings["average_patches"] else
                      "concept, step, layer, position, channel"),
            "position_axis": ("text token" if settings["steer_target"] in {"text_value", "text_token", "text_token_output"} else
                              settings.get("steer_token", "image") + " token"),
            "token_counts": token_counts,
            **({"pool_shape": "concept, channel", "pool_intervention": "raw pooled embedding before text projection"} if steer_pool else {}),
            "timesteps": pipe.scheduler.timesteps.cpu().tolist(),
            **({prefix + "paired_differences": str(checkpoint / (prefix + "paired_differences")) for prefix in prefixes} if paired else {}),
            "intervention": settings["steer_target"]})
    finally:
        sampler.close()


class SteerAdapter:
    """Shared benchmark plugin: native DiT sampling plus the steering hooks."""

    def __init__(self, config):
        spec = config["adapter"]
        config = copy.deepcopy(config)
        config["adapter"] = {"kind": "original"}
        self.sampler = DiTAdapter(config)
        vectors, pool_vectors = None, None
        if not spec.get("steer_noise"):
            state = torch.load(Path(spec["checkpoint"]) / "steering.pt", map_location="cpu", weights_only=True)
            vectors = state["vectors"]
            if spec.get("steer_pool"):
                pool_vectors = state["pool_vectors"]
        self.trace_config, self.trace_roots = config, set()
        self.trace_config["adapter"] = spec
        self.hooks = SD3Steering(self.sampler.pipe.transformer, config["sampling"]["steps"],
            cfg=config["sampling"]["guidance_scale"] > 1, vectors=vectors,
            beta=spec["beta"], clipping=spec["clipping"],
            steer_target=spec.get("steer_target", "text_to_image"),
            average_patches=spec.get("average_patches", True), score_gate=spec.get("score_gate", True),
            steer_noise=spec.get("steer_noise"), noise_seed=spec.get("noise_seed", 0),
            target_count=len(config["targets"]), trace_score=spec.get("trace_score", False),
            steer_token=spec.get("steer_token", "image"), steer_pool=spec.get("steer_pool", False), pool_vectors=pool_vectors)

    def generate(self, records, sampling_config):
        cfg = self.trace_config
        concepts = [cfg["task"] if cfg["task"] in {"i2p", "mscoco"} else concept_key(row["concept"]) for row in records]
        groups = [f"{cfg.get('case_id', '')}/{cfg['task']}/{concept}" for concept in concepts]
        self.hooks.reset(records=records, noise_groups=groups)
        images = self.sampler.generate(records, sampling_config)
        if self.hooks.trace_score:
            from .diagnostics import accumulate_scores
            pipe = self.sampler.pipe
            patch = pipe.vae_scale_factor * pipe.transformer.config.patch_size
            spatial_shape = (sampling_config["height"] // patch, sampling_config["width"] // patch)
            root = Path(records[0]["image"]).parent / "score-traces"
            scores = self.hooks.score_maps.cpu()
            traces = []
            block_target = self.hooks.steer_target in {"attention_residual", "mlp_output", "block_output"}
            text_layers = [i for i, block in enumerate(pipe.transformer.transformer_blocks) if not block.context_pre_only]
            if block_target and self.hooks.steer_token == "all":
                image_count = spatial_shape[0] * spatial_shape[1]
                traces.extend([(root / "image", scores[..., :image_count], "image", None),
                               (root / "text", scores[:, :, text_layers, :, image_count:], "text", text_layers)])
            else:
                kind = (self.hooks.steer_token if block_target else
                        "text" if self.hooks.steer_target in {"text_value", "text_token", "text_token_output"} else "image")
                layers = text_layers if ((block_target and kind == "text") or
                                         self.hooks.steer_target in {"text_token", "text_token_output"}) else None
                traces.append((root, scores[:, :, layers] if layers is not None else scores, kind, layers))
            if self.hooks.steer_pool:
                traces.append((root / "pool", self.hooks.pool_scores.cpu(), "pool", ["global"]))
            for directory, values, kind, layers in traces:
                accumulate_scores(directory, records, values, self.trace_config, spatial_shape,
                                  pipe.scheduler.timesteps.cpu().tolist(), position_kind=kind, layer_ids=layers)
                self.trace_roots.add(directory)
        return images

    def close(self):
        self.hooks.close()
        self.sampler.close()
        if self.trace_roots:
            from .diagnostics import write_score_report
            for root in sorted(self.trace_roots):
                write_score_report(root, self.trace_config)
