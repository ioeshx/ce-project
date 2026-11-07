"""Projected text-message grafting, native output preservation and full user replay."""
from functools import partial
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
import torch.nn.functional as F

from benchmark.tests import test_query_graft as fixtures
from DiT.queryGraft.method import resolve_settings
from DiT.queryGraft.source import text_message_attention


class TextMessageAttentionTests(unittest.TestCase):
    def test_matches_dense_joint_attention_with_masks_and_source_controls(self):
        torch.manual_seed(12)
        query, key, value = (torch.randn(2, 2, 5, 4) for _ in range(3))
        user = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
        anchor = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        region = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        allowed = torch.ones(2, 1, 5, 5, dtype=torch.bool)
        allowed[..., -1] = False
        for flux in (False, True):
            image = slice(3, None) if flux else slice(None, 2)
            text = slice(None, 3) if flux else slice(2, None)
            for dtype in (torch.float32, torch.bfloat16):
                q, k, v = (tensor.to(dtype) for tensor in (query, key, value))
                scores = (q.float() @ k.float().transpose(-1, -2) / 2).masked_fill(~allowed, -torch.inf)
                native = scores[:, :, image].softmax(-1)[..., text] @ v[:, :, text].float()
                for mode in (None, "logits", "values"):
                    with self.subTest(flux=flux, dtype=dtype, mode=mode):
                        controls = {}
                        expected = native
                        if mode:
                            controls = dict(user_tokens=user, anchor_tokens=anchor, region=region,
                                            mode=mode, user_weight=0.5, anchor_weight=2)
                            weights = torch.ones(2, 5)
                            weights[:, text] = torch.where(user, 0.5, torch.where(anchor, 2., 1.))
                            if mode == "logits":
                                edited_scores = scores[:, :, image] + weights.log()[:, None, None]
                                edited = edited_scores.softmax(-1)[..., text] @ v[:, :, text].float()
                            else:
                                edited = scores[:, :, image].softmax(-1)[..., text] @ (
                                    v[:, :, text].float() * weights[:, None, text, None])
                            expected = torch.where(region[:, None, :, None], edited, native)
                        actual = text_message_attention(q, k, v, allowed,
                                                        text_count=3, flux=flux, **controls)
                        torch.testing.assert_close(actual.float(), expected,
                                                   rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                                                   atol=0.01 if dtype == torch.bfloat16 else 1e-6)


class ProjectedTextOutputTests(unittest.TestCase):
    adapter = fixtures.QueryGraftTests.adapter

    def force_mask(self, graft):
        finish = graft.finish_mask

        def fixed_votes():
            if len(graft.settings["pairs"]) == 1:
                values = [[[1, 0, 0, 1]], [[0, 1, 1, 0]]]
            else:
                values = [[[1, 0, 0, 0], [0, 1, 0, 0]],
                          [[0, 0, 1, 0], [0, 0, 0, 1]]]
            graft.votes = torch.tensor(values) * len(graft.localizers)
            finish()
        return patch.object(graft, "finish_mask", side_effect=fixed_votes)

    def dense_message(self, graft, block, single, module, args, kwargs):
        """Independent dense oracle from the normalized, actual native block inputs."""
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        context = kwargs.get("encoder_hidden_states")
        attn = block.attn

        def heads(tensor):
            return tensor.unflatten(-1, (attn.heads, -1)).transpose(1, 2)

        query, key, value = (heads(projection(hidden))
                             for projection in (attn.to_q, attn.to_k, attn.to_v))
        if attn.norm_q is not None:
            query, key = attn.norm_q(query), attn.norm_k(key)
        if not single:
            tq, tk, tv = (heads(projection(context))
                          for projection in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
            if attn.norm_added_q is not None:
                tq, tk = attn.norm_added_q(tq), attn.norm_added_k(tk)
            if graft.flux:
                query, key, value = (torch.cat(parts, 2)
                                     for parts in ((tq, query), (tk, key), (tv, value)))
            else:
                query, key, value = (torch.cat(parts, 2)
                                     for parts in ((query, tq), (key, tk), (value, tv)))
        rotary = kwargs.get("image_rotary_emb")
        if rotary is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query, key = (apply_rotary_emb(tensor, rotary, sequence_dim=2) for tensor in (query, key))
        image = slice(graft.text_count, None) if graft.flux else slice(None, hidden.shape[1])
        text = slice(None, graft.text_count) if graft.flux else slice(hidden.shape[1], None)
        weights = (query[:, :, image].float() @ key.float().transpose(-1, -2)
                   / query.shape[-1] ** 0.5).softmax(-1)
        raw = (weights[..., text] @ value[:, :, text].float()).transpose(1, 2).flatten(2)
        projection = block.proj_out if single else attn.to_out[0]
        return F.linear(raw, projection.weight[:, :raw.shape[-1]], bias=None)

    def check_replay(self, flux, rule="replace", multiple=False):
        pairs = ([{"target": "nudity", "anchor": "dressed"}, {"target": "blood", "anchor": "water"}]
                 if multiple else [{"target": "nudity", "anchor": "dressed"}])
        graft, blocks, call, _, _ = self.adapter(
            flux, "text_output", "all", update_rule=rule, strength=0.25, pairs=pairs)
        self.assertFalse(graft.inline)
        self.assertEqual(graft.editors, list(range(len(blocks))))
        messages, edited_layers, calls = {}, [], []

        def read(layer, block, single, module, args, kwargs):
            if graft.mode in {"anchor", "edit"}:
                tag = graft.current_anchor if graft.mode == "anchor" else None
                messages[graft.mode, tag, layer] = self.dense_message(
                    graft, block, single, module, args, kwargs).detach().clone()

        for layer, block in enumerate(blocks):
            projection = block.proj_out if flux and layer >= len(graft.pipe.transformer.transformer_blocks) else block.attn.to_out[0]
            with torch.no_grad():
                # Bias is deliberately large: replacing the whole projected output must fail this oracle.
                projection.bias.fill_(3 + layer)
            handle = block.attn.register_forward_pre_hook(
                partial(read, layer, block, flux and layer >= len(graft.pipe.transformer.transformer_blocks)),
                with_kwargs=True)
            self.addCleanup(handle.remove)

        native = graft.native_forward
        projected = graft.projected_text_output

        def capture(**inputs):
            calls.append((graft.mode, inputs))
            return native(**inputs)

        def check(layer, output, message):
            tag = graft.current_anchor if graft.mode == "anchor" else None
            torch.testing.assert_close(message, messages[graft.mode, tag, layer], rtol=1e-5, atol=1e-6)
            result = projected(layer, output, message)
            if graft.mode == "edit":
                before = message[-graft.batch:]
                expected = output.clone()
                for pair, definition in enumerate(pairs):
                    mask = (graft.winner == pair)[..., None]
                    if rule == "zero":
                        after = torch.zeros_like(before)
                    else:
                        source = messages["anchor", definition["anchor"], layer]
                        after = (source if rule == "replace" else before + 0.25 * (source - before)
                                 if rule == "blend" else before + 0.25 * source)
                    expected[-graft.batch:] = torch.where(
                        mask, output[-graft.batch:] + (after - before), expected[-graft.batch:])
                torch.testing.assert_close(result, expected, rtol=1e-5, atol=2e-6)
                untouched = graft.winner < 0
                torch.testing.assert_close(result[-graft.batch:][untouched],
                                           output[-graft.batch:][untouched], rtol=0, atol=0)
                if not flux:
                    torch.testing.assert_close(result[:graft.batch], output[:graft.batch], rtol=0, atol=0)
                edited_layers.append(layer)
            return result

        with torch.inference_mode():
            original = graft.pipe.transformer(**call)[0]
            graft.active = True
            with self.force_mask(graft), patch.object(graft, "native_forward", side_effect=capture), \
                    patch.object(graft, "projected_text_output", side_effect=check):
                result = graft.forward(**call)[0]
        self.assertEqual(edited_layers, graft.editors)
        self.assertEqual([mode for mode, _ in calls],
                         ["probe", *([] if rule == "zero" else ["anchor"] * len(pairs)), "edit"])
        self.assertEqual(graft.forward_calls, 2 if rule == "zero" else 2 + len(pairs))
        self.assertFalse(torch.allclose(result[-graft.batch:], original[-graft.batch:]))
        for mode, inputs in calls:
            expected = call["hidden_states"][-graft.batch:] if mode == "anchor" else call["hidden_states"]
            torch.testing.assert_close(inputs["hidden_states"], expected, rtol=0, atol=0)
        if not flux:
            torch.testing.assert_close(result[:graft.batch], original[:graft.batch], rtol=1e-5, atol=1e-6)
        self.assertFalse(graft.single_text_messages)

    def test_all_layers_receive_only_the_projected_text_delta(self):
        for flux in (False, True):
            for multiple in (False, True):
                with self.subTest(flux=flux, multiple=multiple):
                    self.check_replay(flux, multiple=multiple)

    def test_blend_add_and_zero_keep_the_native_output_remainder(self):
        for flux in (False, True):
            for rule in ("blend", "add", "zero"):
                with self.subTest(flux=flux, rule=rule):
                    self.check_replay(flux, rule)

    def test_default_configuration_and_velocity_exception(self):
        config = dict(model_name="sd35_medium", targets=["nudity"], adapter={}, sampling={})
        settings = resolve_settings(config)
        self.assertEqual((settings["representation"], settings["edit_layers"]), ("text_output", "all"))
        self.assertEqual(resolve_settings(config, {"representation": "velocity"})["edit_layers"], "last")
        with self.assertRaisesRegex(ValueError, "one final Transformer output site"):
            resolve_settings(config, {"representation": "velocity", "edit_layers": "all"})

    def test_bfloat16_replay_preserves_dtype_and_runs_every_layer(self):
        for flux in (False, True):
            with self.subTest(flux=flux):
                graft, _, call, _, _ = self.adapter(flux, "text_output", "all")
                graft.pipe.transformer.to(torch.bfloat16)
                graft.dtype = torch.bfloat16
                for name in ("hidden_states", "encoder_hidden_states", "pooled_projections"):
                    call[name] = call[name].to(torch.bfloat16)
                sites = []
                original = graft.projected_text_output

                def capture(layer, output, message):
                    result = original(layer, output, message)
                    if graft.mode == "edit":
                        self.assertEqual(message.dtype, torch.bfloat16)
                        self.assertEqual(result.dtype, torch.bfloat16)
                        sites.append(layer)
                    return result

                graft.active = True
                with self.force_mask(graft), patch.object(graft, "projected_text_output", side_effect=capture), \
                        torch.inference_mode():
                    result = graft.forward(**call)[0]
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(sites, graft.editors)
                self.assertEqual(graft.forward_calls, 3)

    def test_sampling_updates_scheduler_once_per_step_and_traces_projected_messages(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from matplotlib.figure import Figure
        from PIL import Image
        from safetensors.torch import load_file

        graft, _, call, _, _ = self.adapter(False, "text_output", "all", trace=True)
        scheduler_steps = []

        class SamplingPipe(SimpleNamespace):
            def __call__(self, **kwargs):
                scheduler = FlowMatchEulerDiscreteScheduler()
                scheduler.set_timesteps(kwargs["num_inference_steps"])
                latents = kwargs["latents"]
                for step, timestep in enumerate(scheduler.timesteps):
                    graft.settings["rho"] = 1 if step == 1 else 0
                    current = {**call, "hidden_states": torch.cat((latents, latents)),
                               "timestep": timestep.expand(2 * graft.batch) / 1000}
                    negative, positive = self.transformer(**current)[0].chunk(2)
                    prediction = negative + kwargs["guidance_scale"] * (positive - negative)
                    latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                    scheduler_steps.append(step)
                    kwargs["callback_on_step_end"](self, step, timestep, {})
                return SimpleNamespace(images=[Image.new("RGB", (4, 4)) for _ in kwargs["prompt"]])

        graft.pipe = SamplingPipe(**vars(graft.pipe))
        sampling = {**graft.config["sampling"], "steps": 3, "guidance_scale": 4.5}
        with tempfile.TemporaryDirectory() as directory:
            records = [{"sample_id": f"sample_{i}", "prompt": "person", "seed": 42 + i,
                        "image": str(Path(directory) / "images" / "run" / f"sample_{i}.png")}
                       for i in range(2)]
            with self.force_mask(graft), patch.object(Figure, "savefig", autospec=True):
                graft.generate(records, sampling)
            self.assertEqual(scheduler_steps, [0, 1, 2])
            self.assertEqual(graft.forward_calls, 7)  # Two triggered steps x 3, one untouched step x 1.
            for i, destination in enumerate(graft.trace_dirs):
                self.assertEqual([info["representation"] for info in graft.step_traces[i]],
                                 ["text_output"] * 3)
                for step in range(3):
                    tensors = load_file(destination / "tensors" / f"step_{step:04d}.safetensors")
                    self.assertEqual(tensors["edit_layer_indices"].tolist(), graft.editors)
                    count = tensors["activation_value_count"]
                    self.assertEqual(count.tolist(), [[0., 0.]] if step == 1 else [[32., 32.]])
                    if step != 1:
                        self.assertTrue((tensors["activation_delta_rms"] > 0).all())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
