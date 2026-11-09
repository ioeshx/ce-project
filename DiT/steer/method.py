"""Offline steering of SD3 text values or projected text-to-image contributions."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.common import write_json
from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise


def construct_vectors(positive_means, negative_means):
    """Subtract prompt means and normalize channels; keep optional positions."""
    difference = positive_means - negative_means
    return difference.div_(difference.norm(dim=-1, keepdim=True).clamp_min_(1e-8))


def steer_output(output, vectors, beta=-2.0, clipping=True):
    """Apply h <- h + beta * max(dot(h, v), 0) * v, in concept order.

    Negative beta suppresses the positive-minus-negative concept direction.
    Disabling clipping uses the signed dot product on both sides of the plane.
    """
    current = output
    for vector in vectors.to(device=output.device, dtype=torch.float32):
        score = (current.float() * vector).sum(dim=-1, keepdim=True)
        if clipping:
            score = score.clamp_min(0)
        current = (current.float() + beta * score * vector).to(output.dtype)
    return current


def prompt_pairs(group, target, anchor, count=None):
    """CASteer's original paired templates, with the benchmark's target anchors."""
    if group == "i2p":
        people = ["a girl", "a boy", "two men", "two women", "two people", "a man", "a woman",
                  "an old man", "an old woman", "boys", "girls", "men", "women", "group of people", "a human"]
        scenes = ["", "gloomy image", "zoomed in", "talking", "on the street", "in a strange pose",
                  "realism", "colorful background", "on a beach", "playing guitar", "enjoying nature",
                  "smiling", "in a futuristic spaceship", "with kittens"]
        stems = [f"{person} {scene}" for person in people for scene in scenes]
        pairs = [{"positive": f"{stem}, {target}", "negative": f"{stem}, {anchor}"} for stem in stems]
    else:
        classes = Path(__file__).with_name("imagenet_classes.txt").read_text().splitlines()
        suffix = f", {target} style" if group == "style" else f" with {target}"
        negative_suffix = f", {anchor} style" if group == "style" else f" with {anchor}"
        pairs = [{"positive": name + suffix, "negative": name + negative_suffix} for name in classes]
        count = count or 50
    return pairs[:count]


class SD3Steering:
    """Own the hooks and per-step state shared by collection and intervention.

    text_value edits V_text at add_v_proj, before attention reads it.
    text_to_image edits only W_out * A_image,text * V_text, retaining the native
    joint softmax denominator, image-value contribution and projection bias.
    """

    def __init__(self, transformer, steps, *, cfg, vectors=None, beta=-2.0, clipping=True,
                 steer_target="text_to_image", average_patches=True):
        self.vectors, self.beta, self.clipping = vectors, beta, clipping
        self.cfg = cfg
        self.steer_target, self.average_patches = steer_target, average_patches
        self.step, self.samples = -1, 0
        self.handles = [transformer.register_forward_pre_hook(self._next_step)]
        layers = transformer.transformer_blocks
        self.steps, self.layer_count = steps, len(layers)
        # Position count comes from the actual V_text or image-patch tensor.
        self.sums = None
        for index, block in enumerate(layers):
            self._register_block(block.attn, index)

    def _next_step(self, module, args):
        self.step += 1
        if self.step == 0:
            self.samples += 1

    def _register_block(self, attn, layer):
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
            if self.vectors is None:
                return output
            return output + (updated.float() - cross.float()).to(output.dtype)

        self.handles.append(attn.register_forward_pre_hook(extract, with_kwargs=True))
        self.handles.append(attn.to_out[0].register_forward_hook(steer_contribution))

    def _collect_or_steer(self, features, layer):
        """features: [batch, positions, channels]; vectors may retain positions."""
        conditional = slice(features.shape[0] // 2, None) if self.cfg else slice(None)
        current = features[conditional]
        if self.vectors is None:
            statistic = current.float().mean(dim=0)  # Always average samples.
            if self.average_patches:
                statistic = statistic.mean(dim=0)
            if self.sums is None:
                self.sums = torch.zeros(self.steps, self.layer_count, *statistic.shape, dtype=torch.float32)
            self.sums[self.step, layer] += statistic.cpu()
            return features
        updated = features.clone()
        updated[conditional] = steer_output(current, self.vectors[:, self.step, layer], self.beta, self.clipping)
        return updated

    def reset(self):
        """Start another sampling call; keep accumulated offline statistics."""
        self.step = -1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def fit(checkpoint, settings, config):
    """Collect unmodified positive/negative trajectories and save compact means.

    Native sampling stops at latents: fitting neither decodes images nor uses
    an output classifier. The same noise seed is used on both sides of a pair.
    """
    torch.manual_seed(settings["seed"])
    config = copy.deepcopy(config)
    config["adapter"] = {"kind": "original"}
    sampler = DiTAdapter(config)
    pipe, opts = sampler.pipe, sampler.config["sampling"]
    means = {"positive": [], "negative": []}
    pairs_by_concept = []
    shape = (pipe.transformer.config.in_channels, opts["height"] // pipe.vae_scale_factor,
             opts["width"] // pipe.vae_scale_factor)
    try:
        for target, anchor in zip(settings["targets"], settings["anchors"]):
            pairs = prompt_pairs(settings["group"], target, anchor, settings["calibration_prompts"])
            pairs_by_concept.append({"target": target, "anchor": anchor, "pairs": pairs})
            for side in ("positive", "negative"):
                hooks = SD3Steering(pipe.transformer, opts["steps"], cfg=opts["guidance_scale"] > 1,
                    steer_target=settings["steer_target"], average_patches=settings["average_patches"])
                try:
                    for index, pair in enumerate(pairs):
                        hooks.reset()
                        latent = initial_noise(shape, settings["seed"], dtype=sampler.dtype, device="cpu")
                        pipe(prompt=[pair[side]], negative_prompt=[opts.get("negative_prompt") or ""],
                             latents=latent.unsqueeze(0).to(sampler.device),
                             generator=torch.Generator("cpu").manual_seed(settings["seed"]),
                             height=opts["height"], width=opts["width"], num_inference_steps=opts["steps"],
                             guidance_scale=opts["guidance_scale"], max_sequence_length=opts["max_sequence_length"],
                             output_type="latent")
                        print(f"[fit:{target}:{side}] {index + 1}/{len(pairs)}", flush=True)
                    means[side].append(hooks.sums / hooks.samples)
                finally:
                    hooks.close()
        positive = torch.stack(means.pop("positive"))
        negative = torch.stack(means.pop("negative"))
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save({"vectors": construct_vectors(positive, negative),
                    "positive_means": positive, "negative_means": negative}, checkpoint / "steering.pt")
        write_json(checkpoint / "metadata.json", {
            "settings": settings, "sampling": opts, "prompt_pairs": pairs_by_concept,
            "shape": ("concept, step, layer, channel" if settings["average_patches"] else
                      "concept, step, layer, position, channel"),
            "position_axis": "text token" if settings["steer_target"] == "text_value" else "image patch",
            "timesteps": pipe.scheduler.timesteps.cpu().tolist(),
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
        state = torch.load(Path(spec["checkpoint"]) / "steering.pt", map_location="cpu", weights_only=True)
        self.hooks = SD3Steering(self.sampler.pipe.transformer, config["sampling"]["steps"],
            cfg=config["sampling"]["guidance_scale"] > 1, vectors=state["vectors"],
            beta=spec["beta"], clipping=spec["clipping"],
            steer_target=spec.get("steer_target", "text_to_image"),
            average_patches=spec.get("average_patches", True))

    def generate(self, records, sampling_config):
        self.hooks.reset()
        return self.sampler.generate(records, sampling_config)

    def close(self):
        self.hooks.close()
        self.sampler.close()
