"""Known soft maps distinguish ConceptAttention defaults from opt-in voting."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from DiT.queryGraft.method import DEFAULTS, QueryGraft, resolve_settings
from benchmark.tests import test_query_graft as fixtures


class LayerPolicyTests(unittest.TestCase):
    def reader(self, pairs=1, **settings):
        graft = QueryGraft.__new__(QueryGraft)
        graft.settings = {**DEFAULTS, "pairs": [{} for _ in range(pairs)], **settings}
        graft.batch, graft.localizers = 2, [0, 1]
        graft.votes = graft.score_sum = None
        graft.layer_ratios = {}
        graft.diagnostic_mask_sum = graft.diagnostic_score_sum = graft.diagnostic_logit_sum = None
        graft.diagnostic_layers = []
        return graft

    def record(self, graft, layer, probabilities):
        # Identity concept vectors make the supplied image rows the exact logits.
        dimensions = probabilities.shape[-1]
        concepts = torch.eye(dimensions).expand(graft.batch, -1, -1)
        graft.record_concepts(layer, probabilities.log(), concepts)

    def test_four_flag_combinations_have_distinct_known_masks(self):
        layers = torch.tensor([
            [[.60, .25, .15], [.35, .40, .25], [.05, .80, .15]],
            [[.10, .65, .25], [.35, .40, .25], [.45, .30, .25]],
        ])
        expected = {
            (False, False): [True, True, False],
            (True, False): [True, True, True],
            (False, True): [False, False, False],
            (True, True): [True, False, True],
        }
        for aggre, policy in product((False, True), repeat=2):
            with self.subTest(aggre=aggre, policy=policy):
                graft = self.reader(my_layer_aggre=aggre, my_layer_policy=policy)
                for layer, values in enumerate(layers):
                    self.record(graft, layer, torch.stack((values, values.flip(0))))
                graft.finish_mask()
                mask = torch.tensor(expected[aggre, policy])
                torch.testing.assert_close(graft.masks[:, 0], torch.stack((mask, mask.flip(0))))
                mean = layers.mean(0)
                torch.testing.assert_close(graft.localization_scores[:, 0],
                                           torch.stack((mean, mean.flip(0))))
                if not aggre and not policy:
                    # eta and mask_rule have no effect without their opt-in flags.
                    graft.settings.update(eta=1, mask_rule="anchor_min")
                    graft.finish_mask()
                    torch.testing.assert_close(graft.masks[:, 0], torch.stack((mask, mask.flip(0))))
                if aggre:
                    graft.settings["eta"] = .5
                    graft.finish_mask()
                    strict = torch.tensor([False, not policy, False])
                    torch.testing.assert_close(graft.masks[:, 0], torch.stack((strict, strict.flip(0))))

    def test_multiclass_uses_global_concepts_and_preserves_area_gate(self):
        graft = self.reader(pairs=2)
        values = torch.tensor([
            [.50, .10, .10, .20, .05, .05],  # target 0, even though target 1 wins its own triple
            [.10, .05, .05, .60, .10, .10],  # target 1
            [.15, .35, .10, .20, .10, .10],  # anchor 0: no erasure
            [.10, .05, .05, .10, .10, .60],  # background 1: no erasure
        ]).expand(graft.batch, -1, -1)
        for layer in graft.localizers:
            self.record(graft, layer, values)
        graft.finish_mask()
        torch.testing.assert_close(graft.winner, torch.tensor([[0, 1, -1, -1]]).expand(2, -1))
        torch.testing.assert_close(graft.localization_scores.transpose(1, 2).flatten(2), values)
        self.assertTrue(graft.triggered.all())
        graft.settings["rho"] = .25
        graft.finish_mask()
        self.assertFalse(graft.triggered.any())
        self.assertTrue((graft.winner == -1).all())

    def test_constant_single_map_is_empty_and_multiclass_ties_follow_argmax(self):
        for pairs in (1, 2):
            graft = self.reader(pairs=pairs)
            values = torch.full((2, 4, 3 * pairs), 1 / (3 * pairs))
            for layer in graft.localizers:
                self.record(graft, layer, values)
            graft.finish_mask()
            if pairs == 1:
                self.assertFalse(graft.masks.any())
            else:
                torch.testing.assert_close(graft.winner, torch.zeros(2, 4, dtype=torch.int64))

    def test_extra_trace_layers_do_not_change_selected_soft_average(self):
        graft = self.reader(trace=True)
        selected = torch.tensor([[[.6, .2, .2], [.2, .4, .4]]]).expand(2, -1, -1)
        for layer in graft.localizers:
            self.record(graft, layer, selected)
        self.record(graft, 9, selected.flip(1))
        graft.finish_mask()
        torch.testing.assert_close(graft.localization_scores[:, 0], selected)
        self.assertFalse(torch.allclose(graft.diagnostic_score_sum[:, 0] / 3, selected))
        torch.testing.assert_close(graft.winner, torch.tensor([[0, -1]]).expand(2, -1))

    def test_default_models_and_flags_preserve_native_output_when_area_gate_is_off(self):
        for flux, aggre, policy in product((False, True), repeat=3):
            with self.subTest(flux=flux, aggre=aggre, policy=policy):
                fixture = fixtures.QueryGraftTests()
                try:
                    graft, _, call, _, _ = fixture.adapter(
                        flux, "av", trace=True, rho=1,
                        my_layer_aggre=aggre, my_layer_policy=policy)
                    with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
                        graft.trace_dirs = [Path(directory) / str(i) for i in range(graft.batch)]
                        native = graft.pipe.transformer(**call)[0]
                        graft.active = True
                        actual = graft.forward(**call)[0]
                        torch.testing.assert_close(actual, native, atol=1e-6, rtol=1e-5)
                        self.assertEqual(graft.forward_calls, 1)
                        self.assertFalse(graft.triggered.any())
                        self.assertEqual(graft.step_traces[0][0]["my_layer_aggre"], aggre)
                        self.assertEqual(graft.step_traces[0][0]["my_layer_policy"], policy)
                finally:
                    fixture.doCleanups()


class LayerPolicySettingsTests(unittest.TestCase):
    def test_defaults_and_strict_boolean_settings(self):
        config = {"model_name": "sd35_medium", "targets": ["nudity"], "adapter": {}, "sampling": {}}
        settings = resolve_settings(config)
        for name in ("my_layer_aggre", "my_layer_policy"):
            self.assertIs(settings[name], False)
            self.assertIs(resolve_settings(config, {name: True})[name], True)
            for invalid in ("false", 0, 1, None):
                with self.subTest(name=name, invalid=invalid), self.assertRaisesRegex(ValueError, name):
                    resolve_settings(config, {name: invalid})

    def test_cli_persists_four_distinct_plans_and_requires_new_plan_for_changes(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from benchmark.data import load_manifest
        from DiT.queryGraft.run import main

        def schedule(config):
            config["sampling"]["scheduler_config"] = dict(FlowMatchEulerDiscreteScheduler().config)

        names, saved = set(), []
        with tempfile.TemporaryDirectory() as directory, \
                patch("benchmark.model_profiles.resolve_scheduler", schedule), \
                patch("benchmark.layout.timestamp", return_value="20261106-000000"):
            for aggre, policy in product((False, True), repeat=2):
                options = (["--my-layer-aggre"] if aggre else []) + (["--my-layer-policy"] if policy else [])
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main([
                        "benchmark", "--cases", "i2p", "--tasks", "i2p", "--max-records", "1",
                        "--phase", "prepare", "--offline", "--output-dir", directory,
                        "--result-root", str(Path(directory) / "images"), *options]), 0)
                path = Path(output.getvalue().strip().split("[manifest] ")[-1])
                manifest = load_manifest(path)
                settings = manifest["config"]["adapter"]["query_graft"]
                self.assertIs(settings["my_layer_aggre"], aggre)
                self.assertIs(settings["my_layer_policy"], policy)
                names.add(manifest["method"])
                saved.append((path, path.read_bytes()))
                for flag in ("--my-layer-aggre", "--my-layer-policy", "--no-my-layer-aggre", "--no-my-layer-policy"):
                    with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                        main(["benchmark", "--manifest", str(path), flag, "--phase", "sample"])
            self.assertEqual(len(names), 4)
            for path, original in saved:
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
