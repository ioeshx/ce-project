"""Source modes, exact attention controls, and same-step extrapolation."""
from functools import partial
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from benchmark.tests import test_query_graft as fixtures
from benchmark.tests.test_query_graft import Tokenizer
from DiT.queryGraft.source import controlled_attention, joint_prompts, token_groups


class SourceAttentionTests(unittest.TestCase):
    def test_bfloat16_weights_match_known_attention_averages(self):
        query = key = torch.zeros(1, 1, 3, 2, dtype=torch.bfloat16)
        value = torch.tensor([[[[1, 0], [0, 1], [1, 1]]]], dtype=torch.bfloat16)
        for mode, expected in (("logits", [6 / 7, 5 / 7]), ("values", [1, 5 / 6])):
            actual = controlled_attention(
                query, key, value, None, text_count=2, flux=False,
                user_tokens=torch.tensor([[1, 0]], dtype=torch.bool),
                anchor_tokens=torch.tensor([[0, 1]], dtype=torch.bool),
                region=torch.tensor([[1]], dtype=torch.bool), mode=mode, user_weight=0.5, anchor_weight=2)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual[0, 0, 0].float(), torch.tensor(expected), rtol=0.01, atol=0.005)
            torch.testing.assert_close(actual[0, 0, 1:].float(), torch.full((2, 2), 2 / 3), rtol=0.01, atol=0.005)

    def test_appended_token_groups_exclude_padding_and_specials(self):
        prompts = ["man", "a woman"]
        user, anchor = token_groups(Tokenizer(), prompts, joint_prompts(prompts, "clothed"), 8)
        self.assertEqual(user.sum(-1).tolist(), [2, 3])  # Includes the separator comma.
        self.assertEqual(anchor.sum(-1).tolist(), [1, 1])
        self.assertFalse((user & anchor).any())
        self.assertFalse(user[:, 0].any() or anchor[:, 0].any())
        self.assertFalse(user[:, -1].any() or anchor[:, -1].any())

    def test_controls_match_explicit_logits_and_values(self):
        torch.manual_seed(4)
        query, key, value = (torch.randn(2, 2, 5, 4) for _ in range(3))
        user = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.bool)
        anchor = torch.tensor([[0, 1, 0], [0, 0, 1]], dtype=torch.bool)
        region = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        allowed = torch.ones(2, 1, 5, 5, dtype=torch.bool)
        allowed[..., 0] = False
        for flux in (False, True):
            image = slice(3, None) if flux else slice(None, 2)
            text = slice(None, 3) if flux else slice(2, None)
            for mode in ("logits", "values"):
                with self.subTest(flux=flux, mode=mode):
                    weights = torch.ones(2, 5)
                    weights[:, text] = torch.where(user, 0.5, torch.where(anchor, 2.0, 1.0))
                    scores = query @ key.transpose(-1, -2) / 2
                    scores = scores.masked_fill(~allowed, -torch.inf)
                    native = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
                    if mode == "logits":
                        edited = (scores[:, :, image] + weights.log()[:, None, None]).softmax(-1) @ value
                    else:
                        edited = scores[:, :, image].softmax(-1) @ (value * weights[:, None, :, None])
                    expected = native.clone()
                    expected[:, :, image] = torch.where(region[:, None, :, None], edited, native[:, :, image])
                    actual = controlled_attention(
                        query, key, value, allowed, text_count=3, flux=flux, user_tokens=user,
                        anchor_tokens=anchor, region=region, mode=mode, user_weight=0.5, anchor_weight=2)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    neutral = controlled_attention(
                        query, key, value, allowed, text_count=3, flux=flux, user_tokens=user,
                        anchor_tokens=anchor, region=region, mode=mode, user_weight=1, anchor_weight=1)
                    torch.testing.assert_close(neutral, native, rtol=0, atol=0)


class SourcePlanTests(unittest.TestCase):
    def test_modes_prepare_distinct_immutable_benchmark_plans(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from benchmark.data import load_manifest
        from DiT.queryGraft.run import main

        def schedule(config):
            config["sampling"]["scheduler_config"] = dict(FlowMatchEulerDiscreteScheduler().config)

        methods = []
        with tempfile.TemporaryDirectory() as directory, patch("benchmark.model_profiles.resolve_scheduler", schedule):
            for mode, options in (
                    ("anchor", []), ("joint", ["--source-mode", "joint"]),
                    ("logits", ["--source-mode", "logits", "--source-user-weight", "0.5", "--source-anchor-weight", "2"]),
                    ("values", ["--source-mode", "values", "--source-user-weight", "0.5", "--source-anchor-weight", "2"]),
                    ("extrapolate", ["--source-mode", "extrapolate", "--source-scale", "2"])):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(["benchmark", "--cases", "i2p", "--tasks", "i2p", "--max-records", "2",
                                           "--representation", "velocity",
                                           "--phase", "prepare", "--offline", "--output-dir", directory,
                                           "--result-root", str(Path(directory) / "images"), *options]), 0)
                path = Path(output.getvalue().strip().split("[manifest] ")[-1])
                manifest = load_manifest(path)
                settings = manifest["config"]["adapter"]["query_graft"]
                self.assertEqual(settings["source_mode"], mode)
                self.assertEqual(settings["representation"], "velocity")
                self.assertIn("velocity", manifest["method"])
                self.assertEqual(len(manifest["records"]), 2)
                self.assertIn("source.py", manifest["config"]["adapter"]["source_files"])
                if mode != "anchor":
                    self.assertIn(f"source-{mode}", manifest["method"])
                else:
                    self.assertNotIn("source-", manifest["method"])
                methods.append(manifest["method"])
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    main(["benchmark", "--manifest", str(path), "--source-mode", "joint", "--phase", "sample"])
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(len(set(methods)), 5)


class SourceModeTests(unittest.TestCase):
    adapter = fixtures.QueryGraftTests.adapter

    def prepare(self, graft):
        with torch.inference_mode():
            graft.prepare_sources([{"prompt": "man"}, {"prompt": "woman"}])

    def force_mask(self, graft):
        finish = graft.finish_mask

        def finish_with_votes():
            graft.votes = torch.tensor([[[1, 0, 0, 1]], [[0, 1, 1, 0]]]) * len(graft.localizers)
            finish()
        return patch.object(graft, "finish_mask", side_effect=finish_with_votes)

    def test_joint_conditions_are_per_sample_and_keep_native_pooled_encoding(self):
        for flux in (False, True):
            graft, _, _, _, _ = self.adapter(flux, "block_output", source_mode="joint")
            with patch.object(graft.pipe, "encode_prompt", wraps=graft.pipe.encode_prompt) as encode:
                self.prepare(graft)
            anchor = graft.settings["pairs"][0]["anchor"]
            self.assertEqual(encode.call_args.kwargs["prompt"], [f"man, {anchor}", f"woman, {anchor}"])
            sequence, pooled, _ = graft.anchor_conditions[anchor]
            self.assertFalse(torch.equal(sequence[0], sequence[1]))
            self.assertFalse(torch.equal(pooled[0], pooled[1]))

    def test_sd3_mixed_clip_rows_stay_unweighted(self):
        graft, _, _, _, _ = self.adapter(False, "block_output", source_mode="logits")
        groups = [
            (torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool).expand(2, -1),
             torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool).expand(2, -1)),
            (torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.bool).expand(2, -1),
             torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.bool).expand(2, -1)),
            (torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool).expand(2, -1),
             torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool).expand(2, -1))]
        with patch("DiT.queryGraft.method.token_groups", side_effect=groups):
            self.prepare(graft)
        user, anchor = next(iter(graft.source_groups.values()))
        self.assertFalse(user[:, 2].any() or anchor[:, 2].any())
        self.assertTrue(user[:, 1].all())
        self.assertTrue(anchor[:, 3].all())

    def test_neutral_weight_modes_equal_joint_and_weighted_modes_change_source(self):
        for flux in (False, True):
            for representation in ("text_output", "av", "residual", "block_output", "velocity"):
                for edit_layers in (("last",) if representation == "velocity" else ("last", "all")):
                    with self.subTest(flux=flux, representation=representation, edit_layers=edit_layers):
                        results = {}
                        votes = {}
                        for mode, user_weight, anchor_weight in (
                                ("joint", 1, 1), ("logits", 1, 1), ("values", 1, 1), ("extrapolate", 1, 1),
                                ("logits", 0.5, 2), ("values", 0.5, 2)):
                            graft, _, call, _, _ = self.adapter(
                                flux, representation, edit_layers, source_mode=mode,
                                source_user_weight=user_weight, source_anchor_weight=anchor_weight)
                            self.prepare(graft)
                            graft.active = True
                            # Save actual localization votes before applying the same test edit mask.
                            record = graft.finish_mask

                            def capture_votes():
                                votes[mode, user_weight] = graft.votes.clone()
                                graft.votes = torch.tensor([[[1, 0, 0, 1]], [[0, 1, 1, 0]]]) * len(graft.localizers)
                                record()

                            with patch.object(graft, "finish_mask", side_effect=capture_votes), torch.inference_mode():
                                results[mode, user_weight] = graft.forward(**call)[0]
                        for mode in ("logits", "values", "extrapolate"):
                            torch.testing.assert_close(results[mode, 1], results["joint", 1], rtol=0, atol=0)
                        for mode in ("logits", "values"):
                            torch.testing.assert_close(votes[mode, 0.5], votes["joint", 1], rtol=0, atol=0)
                            self.assertTrue(torch.isfinite(results[mode, 0.5]).all())
                            self.assertFalse(torch.equal(results[mode, 0.5], results["joint", 1]))

    def test_zero_extrapolation_matches_native_for_all_edit_sites(self):
        for flux in (False, True):
            for representation in ("text_output", "av", "residual", "block_output", "velocity"):
                for edit_layers in (("last",) if representation == "velocity" else ("last", "all")):
                    with self.subTest(flux=flux, representation=representation, edit_layers=edit_layers):
                        graft, _, call, _, _ = self.adapter(
                            flux, representation, edit_layers, source_mode="extrapolate", source_scale=0)
                        self.prepare(graft)
                        with torch.inference_mode():
                            native = graft.pipe.transformer(**call)[0]
                            graft.active = True
                            with self.force_mask(graft):
                                actual = graft.forward(**call)[0]
                        torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
                        self.assertEqual(graft.forward_calls,
                                         2 if edit_layers == "last" and representation != "text_output" else 3)

    def test_extrapolation_uses_original_user_states_during_multilayer_replay(self):
        for flux in (False, True):
            for edit_layers in ("last", "all"):
                with self.subTest(flux=flux, edit_layers=edit_layers):
                    graft, blocks, call, frames, _ = self.adapter(
                        flux, "block_output", edit_layers, source_mode="extrapolate", source_scale=2)
                    self.prepare(graft)
                    outputs = {}

                    def capture(layer, module, args, output):
                        outputs[graft.mode, layer] = output[1].detach().clone()

                    for layer, block in enumerate(blocks):
                        self.addCleanup(block.register_forward_hook(partial(capture, layer)).remove)
                    graft.active = True
                    with self.force_mask(graft), torch.inference_mode():
                        graft.forward(**call)
                    mode = "user" if edit_layers == "last" else "edit"
                    reference = "user" if edit_layers == "last" else "probe"
                    mask = (graft.winner >= 0)[..., None]
                    for layer in graft.editors:
                        original = frames[reference, layer][1][-graft.batch:]
                        source = frames["anchor", layer][1]
                        extrapolated = original + 2 * (source - original)
                        before = frames[mode, layer][1]
                        expected = before.clone()
                        expected[-graft.batch:] = torch.where(mask, extrapolated, before[-graft.batch:])
                        torch.testing.assert_close(outputs[mode, layer], expected, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
