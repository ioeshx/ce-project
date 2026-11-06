"""Independent score and attention checks for queryGraft's saliency diagnostics."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import product
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.models.embeddings import apply_rotary_emb

from DiT.queryGraft.method import QueryAttention, QueryGraft, resolve_settings


class ConceptScoreTests(unittest.TestCase):
    def test_known_scores_preserve_pair_channel_cfg_and_spatial_order(self):
        # Two CFG-negative rows must never leak into either positive sample.
        positive = torch.tensor([
            [[3, 1, 0], [0, 3, 1], [1, 0, 3], [1, 1, 0], [0, 0, 0], [-3, 0, -1]],
            [[0, 3, 1], [1, 0, 3], [3, 1, 0], [0, 1, 1], [0, 0, 0], [0, -3, -1]],
        ], dtype=torch.float32)
        basis = torch.tensor([[2, 0, 0], [0, 5, 0], [0, 0, 9],
                              [0, 7, 0], [0, 0, 3], [4, 0, 0]], dtype=torch.float32)
        expected_logits = torch.stack((positive * torch.tensor([2, 5, 9]),
                                       positive[..., [1, 2, 0]] * torch.tensor([7, 3, 4])), dim=1)
        expected_votes = torch.tensor([
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1]],
            [[0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0]],
        ], dtype=torch.int32)
        anchor_min_votes = torch.tensor([
            [[0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]],
            [[0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]],
        ], dtype=torch.int32)
        # Scalar Python arithmetic serves as an independent softmax oracle.
        expected_probabilities = expected_logits.clone()
        for b in range(2):
            for k in range(2):
                for p in range(6):
                    weights = [math.exp(float(v)) for v in expected_logits[b, k, p]]
                    expected_probabilities[b, k, p] = torch.tensor([v / sum(weights) for v in weights])
        for dtype, mask_rule in product((torch.float32, torch.bfloat16), ("target_max", "anchor_min")):
            with self.subTest(dtype=dtype, mask_rule=mask_rule):
                graft = QueryGraft.__new__(QueryGraft)
                graft.batch, graft.localizers = 2, [3]
                graft.settings = {"trace": True, "mask_rule": mask_rule,
                                  "my_layer_aggre": True, "my_layer_policy": True}
                graft.votes = None
                graft.layer_ratios = {}
                graft.diagnostic_mask_sum = graft.diagnostic_score_sum = graft.diagnostic_logit_sum = None
                graft.diagnostic_layers = []
                image = torch.cat((torch.full_like(positive, 1000), positive)).to(dtype)
                concepts = torch.cat((-basis.expand(2, -1, -1), basis.expand(2, -1, -1))).to(dtype)
                graft.record_concepts(3, image, concepts)
                torch.testing.assert_close(graft.diagnostic_logit_sum, expected_logits)
                torch.testing.assert_close(graft.diagnostic_score_sum, expected_probabilities)
                votes = anchor_min_votes if mask_rule == "anchor_min" else expected_votes
                torch.testing.assert_close(graft.votes, votes)
                torch.testing.assert_close(graft.diagnostic_mask_sum, votes.float())
                # Diagnostics include a non-localizer without adding another vote.
                graft.record_concepts(9, image, concepts)
                torch.testing.assert_close(graft.diagnostic_logit_sum / 2, expected_logits)
                torch.testing.assert_close(graft.diagnostic_score_sum / 2, expected_probabilities)
                torch.testing.assert_close(graft.votes, votes)
                torch.testing.assert_close(graft.diagnostic_mask_sum / 2, votes.float())
                self.assertEqual(graft.diagnostic_layers, [3, 9])

    def test_anchor_min_ties_layer_votes_and_area_gate(self):
        graft = QueryGraft.__new__(QueryGraft)
        graft.batch, graft.localizers = 1, [0, 1, 2]
        graft.settings = {"trace": False, "mask_rule": "anchor_min", "eta": .5, "rho": .25,
                          "my_layer_aggre": True, "my_layer_policy": True}
        graft.votes = None
        layers = [
            [[1, 0, 2], [3, 0, 1], [2, 2, 3], [3, 1, 1]],
            [[1, 1, 0], [3, 0, 1], [1, 0, 3], [3, 1, 1]],
            [[1, 0, 2], [3, 0, 1], [2, 2, 3], [0, 0, 0]],
        ]
        for layer, values in enumerate(layers):
            graft.record_concepts(layer, torch.tensor([values], dtype=torch.float32), torch.eye(3)[None])
        torch.testing.assert_close(graft.votes, torch.tensor([[[2, 3, 1, 0]]], dtype=torch.int32))
        graft.finish_mask()
        torch.testing.assert_close(graft.winner, torch.tensor([[0, 0, -1, -1]]))
        self.assertTrue(graft.triggered.item())
        # Both vote and area thresholds remain strict under the new rule.
        graft.settings["eta"] = 2 / 3
        graft.finish_mask()
        torch.testing.assert_close(graft.masks, torch.tensor([[[False, True, False, False]]]))
        self.assertFalse(graft.triggered.item())
        self.assertTrue((graft.winner == -1).all())

    def test_readout_matches_full_dense_masked_attention(self):
        # Compare the optimized split implementation with one explicit full
        # attention matrix. Include two concept pairs to catch cross-pair leaks.
        for (flux, single), include_prompt, mask_kind in product(
                ((False, False), (True, False), (True, True)), (None, False, True), (None, "bool", "float")):
            with self.subTest(flux=flux, single=single, include_prompt=include_prompt,
                              mask_kind=mask_kind), torch.inference_mode():
                torch.manual_seed(31)
                if flux:
                    model = FluxTransformer2DModel(
                        in_channels=4, num_layers=1, num_single_layers=1,
                        attention_head_dim=8, num_attention_heads=2,
                        joint_attention_dim=16, pooled_projection_dim=8,
                        axes_dims_rope=(2, 2, 4))
                    blocks = model.single_transformer_blocks if single else model.transformer_blocks
                else:
                    model = SD3Transformer2DModel(
                        sample_size=4, patch_size=2, in_channels=4, out_channels=4,
                        num_layers=2, attention_head_dim=8, num_attention_heads=2,
                        joint_attention_dim=16, caption_projection_dim=16,
                        pooled_projection_dim=8, pos_embed_max_size=4)
                    blocks = model.transformer_blocks
                attn = blocks[0].attn
                captured = []
                owner = SimpleNamespace(
                    active=True, mode="probe", flux=flux, text_count=3, concept_count=6,
                    settings={"pairs": [{}, {}]},
                    record_concepts=lambda layer, image, concept: captured.append((image, concept)),
                    attention_output=lambda layer, image: image)
                if include_prompt is not None:
                    owner.settings["concept_include_prompt"] = include_prompt
                image = torch.randn(2, 6, 16)
                context = torch.randn(2, 9, 16)
                hidden = torch.cat((context, image), 1) if single else image

                def project(module, data):
                    return module(data).unflatten(-1, (attn.heads, -1)).transpose(1, 2)

                q, k, v = (project(module, hidden) for module in (attn.to_q, attn.to_k, attn.to_v))
                if attn.norm_q is not None:
                    q = attn.norm_q(q)
                if attn.norm_k is not None:
                    k = attn.norm_k(k)
                if not single:
                    cq, ck, cv = (project(module, context) for module in
                                  (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
                    if attn.norm_added_q is not None:
                        cq = attn.norm_added_q(cq)
                    if attn.norm_added_k is not None:
                        ck = attn.norm_added_k(ck)
                    q, k, v = (torch.cat((c, x) if flux else (x, c), dim=2)
                               for x, c in ((q, cq), (k, ck), (v, cv)))
                rotary = None
                if flux:
                    ids = torch.zeros(15, 3)
                    ids[9:, 1] = torch.arange(2).repeat_interleave(3)
                    ids[9:, 2] = torch.arange(3).repeat(2)
                    rotary = model.pos_embed(ids)
                    q = apply_rotary_emb(q, rotary, sequence_dim=2)
                    k = apply_rotary_emb(k, rotary, sequence_dim=2)
                concept_indices = list(range(3, 9)) if flux else list(range(9, 15))
                base_indices = [i for i in range(15) if i not in concept_indices]
                image_indices = list(range(9, 15)) if flux else list(range(6))
                prompt_indices = list(range(3)) if flux else list(range(6, 9))
                allowed = torch.zeros(15, 15, dtype=torch.bool)
                for row in base_indices:
                    allowed[row, base_indices] = True
                for pair in range(2):
                    group = concept_indices[3 * pair:3 * pair + 3]
                    for row in group:
                        allowed[row, base_indices if include_prompt else image_indices] = True
                        allowed[row, group] = True
                logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                attention_mask = None
                if mask_kind is not None:
                    native_allowed = torch.ones(2, 1, 15, 15, dtype=torch.bool)
                    native_allowed[..., prompt_indices[1]] = False
                    native_allowed[0, ..., image_indices[2]] = False
                    native_allowed[..., concept_indices[0], concept_indices[1]] = False
                    if mask_kind == "bool":
                        attention_mask = native_allowed
                        logits = logits.masked_fill(~native_allowed, -torch.inf)
                    else:
                        attention_mask = (torch.randn(2, 1, 15, 15) * .2).masked_fill(
                            ~native_allowed, -torch.inf)
                        logits = logits + attention_mask
                dense_av = logits.masked_fill(~allowed, -torch.inf).softmax(-1) @ v
                dense_av = dense_av.transpose(1, 2).flatten(2)
                expected_image = dense_av[:, 9:] if flux else dense_av[:, :6]
                expected_concepts = dense_av[:, concept_indices]
                processor = QueryAttention(owner, 0, single, attn.processor)
                processor(attn, hidden, encoder_hidden_states=None if single else context,
                          image_rotary_emb=rotary, attention_mask=attention_mask)
                self.assertEqual(len(captured), 1)
                torch.testing.assert_close(captured[0][0], expected_image, atol=5e-7, rtol=1e-5)
                torch.testing.assert_close(captured[0][1], expected_concepts, atol=5e-7, rtol=1e-5)
                if mask_kind is None:
                    # Perturb only prompt tokens at this layer. Default concept
                    # readouts must stay fixed; opt-in readouts must respond.
                    changed_context = context.clone()
                    changed_context[:, :3] = torch.randn_like(context[:, :3]) * 4
                    changed_hidden = torch.cat((changed_context, image), 1) if single else image
                    processor(attn, changed_hidden, encoder_hidden_states=None if single else changed_context,
                              image_rotary_emb=rotary)
                    self.assertFalse(torch.allclose(captured[0][0], captured[1][0]))
                    if include_prompt:
                        self.assertFalse(torch.allclose(captured[0][1], captured[1][1]))
                    else:
                        torch.testing.assert_close(captured[0][1], captured[1][1], atol=0, rtol=0)

    def test_trace_preserves_outputs_masks_and_forward_count(self):
        from benchmark.tests import test_query_graft as fixtures

        for flux, mask_rule in product((False, True), ("target_max", "anchor_min")):
            with self.subTest(flux=flux, mask_rule=mask_rule):
                fixture = fixtures.QueryGraftTests()
                try:
                    graft, _, call, _, _ = fixture.adapter(flux, "av", mask_rule=mask_rule)
                    graft.localizers = [0]  # Trace additionally scores the other layer.
                    graft.active = True
                    with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
                        without_trace = graft.forward(**call)[0].clone()
                        votes, winner = graft.votes.clone(), graft.winner.clone()
                        calls = graft.forward_calls
                        graft.trace_dirs = [Path(directory) / str(i) for i in range(graft.batch)]
                        graft.settings["trace"] = True
                        with_trace = graft.forward(**call)[0]
                        torch.testing.assert_close(with_trace, without_trace, rtol=0, atol=0)
                        torch.testing.assert_close(graft.votes, votes, rtol=0, atol=0)
                        torch.testing.assert_close(graft.winner, winner, rtol=0, atol=0)
                        self.assertEqual(graft.forward_calls, calls * 2)
                        self.assertEqual(graft.diagnostic_layers, [0, 1])
                        self.assertEqual(graft.step_traces[0][0]["mask_rule"], mask_rule)
                finally:
                    fixture.doCleanups()


class ConceptAttentionSettingsTests(unittest.TestCase):
    def test_prompt_attention_is_opt_in_and_requires_a_boolean(self):
        config = {"model_name": "sd35_medium", "targets": ["nudity"], "adapter": {}, "sampling": {}}
        self.assertIs(resolve_settings(config)["concept_include_prompt"], False)
        self.assertIs(resolve_settings(config, {"concept_include_prompt": True})["concept_include_prompt"], True)
        config["adapter"]["query_graft"] = {"concept_include_prompt": True}
        self.assertIs(resolve_settings(config, {"concept_include_prompt": False})["concept_include_prompt"], False)
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "concept_include_prompt"):
                resolve_settings(config, {"concept_include_prompt": invalid})

    def test_cli_persists_prompt_attention_choice_and_separates_runs(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from benchmark.data import load_manifest
        from DiT.queryGraft.run import main

        def schedule(config):
            config["sampling"]["scheduler_config"] = dict(FlowMatchEulerDiscreteScheduler().config)

        saved = []
        with tempfile.TemporaryDirectory() as directory, \
                patch("benchmark.model_profiles.resolve_scheduler", schedule), \
                patch("benchmark.layout.timestamp", return_value="20261106-000000"):
            for include_prompt, options in ((False, []), (True, ["--concept-include-prompt"]),
                                            (False, ["--no-concept-include-prompt"])):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main([
                        "benchmark", "--cases", "i2p", "--tasks", "i2p", "--max-records", "1",
                        "--phase", "prepare", "--offline", "--output-dir", directory,
                        "--result-root", str(Path(directory) / "images"), *options]), 0)
                path = Path(output.getvalue().strip().split("[manifest] ")[-1])
                manifest = load_manifest(path)
                settings = manifest["config"]["adapter"]["query_graft"]
                self.assertIs(settings["concept_include_prompt"], include_prompt)
                self.assertEqual("concept-image-prompt-" in manifest["method"], include_prompt)
                saved.append((path, path.read_bytes(), manifest["method"]))
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    main(["benchmark", "--manifest", str(path), "--concept-include-prompt", "--phase", "sample"])
                self.assertEqual(error.exception.code, 2)
            self.assertNotEqual(saved[0][2], saved[1][2])
            for path, original, _ in saved:
                self.assertEqual(path.read_bytes(), original)


class MaskRuleSettingsTests(unittest.TestCase):
    def test_default_explicit_rule_and_invalid_rule(self):
        config = {"model_name": "sd35_medium", "targets": ["nudity"], "adapter": {}, "sampling": {}}
        self.assertEqual(resolve_settings(config)["mask_rule"], "target_max")
        self.assertEqual(resolve_settings(config, {"mask_rule": "anchor_min"})["mask_rule"], "anchor_min")
        with self.assertRaisesRegex(ValueError, "mask_rule"):
            resolve_settings(config, {"mask_rule": "unknown"})

    def test_cli_saves_rule_in_distinct_immutable_plans(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from benchmark.data import load_manifest
        from DiT.queryGraft.run import main

        def schedule(config):
            config["sampling"]["scheduler_config"] = dict(FlowMatchEulerDiscreteScheduler().config)

        saved = []
        with tempfile.TemporaryDirectory() as directory, \
                patch("benchmark.model_profiles.resolve_scheduler", schedule), \
                patch("benchmark.layout.timestamp", return_value="20261106-000000"):
            for rule, options in (("target_max", []), ("anchor_min", ["--mask-rule", "anchor_min"])):
                options = ["--my-layer-policy", *options]
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main([
                        "benchmark", "--cases", "i2p", "--tasks", "i2p", "--max-records", "1",
                        "--phase", "prepare", "--offline", "--output-dir", directory,
                        "--result-root", str(Path(directory) / "images"), *options]), 0)
                path = Path(output.getvalue().strip().split("[manifest] ")[-1])
                manifest = load_manifest(path)
                self.assertEqual(manifest["config"]["adapter"]["query_graft"]["mask_rule"], rule)
                self.assertEqual("mask-anchor_min" in manifest["method"], rule == "anchor_min")
                saved.append((path, path.read_bytes(), manifest["method"]))
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                    main(["benchmark", "--manifest", str(path), "--mask-rule", "anchor_min", "--phase", "sample"])
                self.assertEqual(error.exception.code, 2)
            self.assertNotEqual(saved[0][2], saved[1][2])
            for path, original, _ in saved:
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
