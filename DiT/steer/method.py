"""Offline mean-difference steering on SD3 joint-attention text contributions."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark.common import write_json
from benchmark.dit_adapter import DiTAdapter
from benchmark.sampling_profile import initial_noise


def construct_vectors(positive_means, negative_means):
    """[concept, step, layer, channel]: average first, subtract, then normalize."""
    difference = positive_means - negative_means
    return F.normalize(difference, dim=-1, eps=1e-8)


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

    We extract W_out * Attention(Q_image, [K_image, K_text], [0, V_text]).
    Keeping image keys preserves the native JOINT softmax denominator. Native
    attention still computes its full output; only our steering delta is added.
    """

    def __init__(self, transformer, steps, *, cfg, vectors=None, beta=-2.0, clipping=True):
        self.vectors, self.beta, self.clipping = vectors, beta, clipping
        self.cfg = cfg
        self.step, self.samples = -1, 0
        self.handles = [transformer.register_forward_pre_hook(self._next_step)]
        layers = transformer.transformer_blocks
        channels = layers[0].attn.to_out[0].out_features
        self.sums = torch.zeros(steps, len(layers), channels) if vectors is None else None
        for index, block in enumerate(layers):
            self._register_block(block.attn, index)

    def _next_step(self, module, args):
        self.step += 1
        if self.step == 0:
            self.samples += 1

    def _register_block(self, attn, layer):
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

        def collect_or_steer(module, args, output):
            # Exclude projection bias: it belongs to the full native output.
            cross = F.linear(state.pop("cross"), module.weight, bias=None)
            conditional = slice(cross.shape[0] // 2, None) if self.cfg else slice(None)
            current = cross[conditional]
            if self.vectors is None:
                self.sums[self.step, layer] += current.float().mean(dim=(0, 1)).cpu()
                return output
            updated = steer_output(current, self.vectors[:, self.step, layer], self.beta, self.clipping)
            delta = updated.float() - current.float()
            result = output.clone()
            result[conditional] += delta.to(output.dtype)
            return result

        self.handles.append(attn.register_forward_pre_hook(extract, with_kwargs=True))
        self.handles.append(attn.to_out[0].register_forward_hook(collect_or_steer))

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
                hooks = SD3Steering(pipe.transformer, opts["steps"], cfg=opts["guidance_scale"] > 1)
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
        positive, negative = (torch.stack(means[side]) for side in ("positive", "negative"))
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        torch.save({"vectors": construct_vectors(positive, negative),
                    "positive_means": positive, "negative_means": negative}, checkpoint / "steering.pt")
        write_json(checkpoint / "metadata.json", {
            "settings": settings, "sampling": opts, "prompt_pairs": pairs_by_concept,
            "shape": "concept, step, layer, channel", "timesteps": pipe.scheduler.timesteps.cpu().tolist(),
            "intervention": "projected text-value contribution to image tokens, joint softmax retained"})
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
            beta=spec["beta"], clipping=spec["clipping"])

    def generate(self, records, sampling_config):
        self.hooks.reset()
        return self.sampler.generate(records, sampling_config)

    def close(self):
        self.hooks.close()
        self.sampler.close()
