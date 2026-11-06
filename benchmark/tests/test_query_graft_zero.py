"""Zero updates must edit the chosen feature site without computing a source."""
from contextlib import redirect_stdout
from functools import partial
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from DiT.queryGraft.method import QueryGraft
from benchmark.tests import test_query_graft as fixtures
from benchmark.tests import test_query_graft_velocity as velocity_fixtures


class ZeroUpdateTests(unittest.TestCase):
    adapter = fixtures.QueryGraftTests.adapter
    force_mask = velocity_fixtures.VelocityTests.force_mask
    capture_native = velocity_fixtures.VelocityTests.capture_native
    output_mask = velocity_fixtures.VelocityTests.output_mask

    def test_zeroes_all_feature_sites_and_edit_layers_with_multiple_pairs(self):
        pairs = [{"target": "Pikachu", "anchor": "animal"},
                 {"target": "Snoopy", "anchor": "cartoon"}]
        for flux in (False, True):
            for representation in ("av", "residual", "block_output"):
                for edit_layers in ("last", "all"):
                    with self.subTest(flux=flux, representation=representation, layers=edit_layers):
                        graft, blocks, call, frames, _ = self.adapter(
                            flux, representation, edit_layers, update_rule="zero", pairs=pairs)
                        edits, outputs = {}, {}
                        original_patch, finish = graft.patch, graft.finish_mask

                        def capture_patch(layer, image):
                            result = original_patch(layer, image)
                            edits[layer] = image.clone(), result.clone()
                            return result

                        def finish_with_votes():
                            graft.votes = torch.tensor([
                                [[1, 0, 0, 1], [0, 1, 0, 0]],
                                [[0, 1, 0, 0], [1, 0, 1, 0]],
                            ]) * len(graft.localizers)
                            finish()

                        def capture_block(layer, module, args, output):
                            outputs[graft.mode, layer] = tuple(
                                None if value is None else value.clone() for value in output)

                        for layer, block in enumerate(blocks):
                            handle = block.register_forward_hook(partial(capture_block, layer))
                            self.addCleanup(handle.remove)
                        with torch.inference_mode():
                            native = graft.pipe.transformer(**call)[0]
                            graft.active = True
                            with patch.object(graft, "finish_mask", side_effect=finish_with_votes), \
                                    patch.object(graft, "patch", side_effect=capture_patch):
                                actual = graft.forward(**call)[0]
                        self.assertEqual(set(edits), set(graft.editors))
                        for before, after in edits.values():
                            mask = (graft.winner >= 0)[..., None].expand_as(before)
                            self.assertTrue(torch.count_nonzero(before[mask]).item())
                            self.assertEqual(torch.count_nonzero(after[mask]).item(), 0)
                            torch.testing.assert_close(after[~mask], before[~mask], rtol=0, atol=0)
                        mode = "user" if edit_layers == "last" else "edit"
                        if representation == "block_output":
                            for layer in graft.editors:
                                context, before = frames[mode, layer]
                                actual_context, after = outputs[mode, layer]
                                expected = before.clone()
                                expected[-graft.batch:] = before[-graft.batch:].masked_fill(
                                    (graft.winner >= 0)[..., None], 0)
                                torch.testing.assert_close(after, expected, rtol=0, atol=0)
                                if context is not None:
                                    torch.testing.assert_close(actual_context, context, rtol=0, atol=0)
                        if not flux:
                            torch.testing.assert_close(actual[:graft.batch], native[:graft.batch],
                                                       rtol=1e-5, atol=1e-6)
                        self.assertFalse(torch.allclose(actual[-graft.batch:], native[-graft.batch:]))
                        self.assertEqual(graft.forward_calls, 1 if edit_layers == "last" else 2)
                        self.assertFalse(graft.anchor_conditions)
                        self.assertFalse(graft.sources)

    def test_zero_velocity_preserves_cfg_and_unselected_latent_values(self):
        for flux in (False, True):
            for return_dict in (False, True):
                with self.subTest(flux=flux, return_dict=return_dict):
                    graft, _, call, _, _ = self.adapter(flux, "velocity", update_rule="zero", strength=0)
                    call["return_dict"] = return_dict
                    raw = {}
                    graft.active = True
                    with self.force_mask(graft), self.capture_native(graft, raw, {}), torch.inference_mode():
                        actual = graft.forward(**call)[0]
                    expected = raw["user"].clone()
                    expected[-graft.batch:] = expected[-graft.batch:].masked_fill(self.output_mask(graft), 0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(set(raw), {"user"})
                    self.assertEqual(graft.forward_calls, 1)

    def test_empty_region_skips_zero_update_and_matches_native(self):
        for flux in (False, True):
            for representation in ("av", "residual", "block_output", "velocity"):
                with self.subTest(flux=flux, representation=representation):
                    graft, _, call, _, _ = self.adapter(
                        flux, representation, "last" if representation == "velocity" else "all",
                        update_rule="zero", rho=1, source_mode="extrapolate")
                    with torch.inference_mode():
                        native = graft.pipe.transformer(**call)[0]
                        graft.active = True
                        actual = graft.forward(**call)[0]
                    torch.testing.assert_close(actual, native, rtol=1e-5, atol=1e-6)
                    self.assertFalse(graft.triggered.any())
                    self.assertEqual(graft.forward_calls, 1)
                    self.assertFalse(graft.user_sources)

    def test_source_options_are_unused_and_default_mask_logic_is_unchanged(self):
        for mode in ("anchor", "joint", "logits", "values", "extrapolate"):
            with self.subTest(mode=mode):
                with patch.object(QueryGraft, "encode_source", side_effect=AssertionError("source encoding")):
                    graft, _, call, _, _ = self.adapter(
                        False, "av", update_rule="zero", source_mode=mode,
                        my_layer_aggre=False, my_layer_policy=False)
                    graft.prepare_sources([{"prompt": "person " * 50}] * graft.batch)
                    graft.active = True
                    with torch.inference_mode():
                        graft.forward(**call)
                reference, _, reference_call, _, _ = self.adapter(
                    False, "av", my_layer_aggre=False, my_layer_policy=False)
                reference.active = True
                with torch.inference_mode():
                    reference.forward(**reference_call)
                torch.testing.assert_close(graft.localization_scores, reference.localization_scores, rtol=0, atol=0)
                torch.testing.assert_close(graft.masks, reference.masks, rtol=0, atol=0)
                torch.testing.assert_close(graft.winner, reference.winner, rtol=0, atol=0)
                self.assertFalse(graft.user_sources)
                self.assertFalse(graft.source_groups)
                self.assertEqual(graft.forward_calls, 1)
                self.assertEqual(reference.settings["update_rule"], "replace")

    def test_trace_reports_zero_activations_and_correct_operation(self):
        from matplotlib.figure import Figure
        from DiT.queryGraft.diagnostics import save_sample_diagnostics

        graft, _, call, _, _ = self.adapter(False, "velocity", update_rule="zero", trace=True)
        figures = {}

        def capture_figure(figure, path, **kwargs):
            figures[str(path)] = (figure._suptitle.get_text(),
                                  [axis.get_title() for axis in figure.axes])

        with tempfile.TemporaryDirectory() as directory:
            graft.trace_dirs = [Path(directory) / str(i) for i in range(graft.batch)]
            graft.active = True
            with self.force_mask(graft), torch.inference_mode(), \
                    patch.object(Figure, "savefig", autospec=True, side_effect=capture_figure):
                graft.forward(**call)
                for i, destination in enumerate(graft.trace_dirs):
                    tensors = load_file(destination / "tensors" / "step_0000.safetensors")
                    self.assertEqual(torch.count_nonzero(tensors["activation_after_rms"]).item(), 0)
                    self.assertTrue((tensors["activation_before_rms"] > 0).all())
                    torch.testing.assert_close(tensors["activation_delta_rms"], tensors["activation_before_rms"])
                    self.assertEqual(graft.step_traces[i][0]["update_rule"], "zero")
                    title, panels = figures[str(destination / "timesteps" / "step_0000.png")]
                    self.assertIn("update=zero", title)
                    self.assertIn("nudity -> zero", panels[0])
                    save_sample_diagnostics(destination, {
                        "applied_mask_mean": tensors["applied_mask"].float(),
                        "localization_scores_mean": tensors["localization_scores_mean"],
                    }, graft.settings["pairs"], 1, update_rule=graft.settings["update_rule"])
                    title, panels = figures[str(destination / "summary.png")]
                    self.assertIn("update=zero", title)
                    self.assertIn("nudity -> zero", panels[0])

    def test_cli_requires_explicit_zero_and_saves_a_separate_plan(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from benchmark.data import load_manifest
        from DiT.queryGraft.run import main

        def schedule(config):
            config["sampling"]["scheduler_config"] = dict(FlowMatchEulerDiscreteScheduler().config)

        names = set()
        with tempfile.TemporaryDirectory() as directory, \
                patch("benchmark.model_profiles.resolve_scheduler", schedule):
            for rule in ("replace", "zero"):
                output = StringIO()
                with redirect_stdout(output):
                    main(["benchmark", "--cases", "i2p", "--tasks", "i2p", "--max-records", "1",
                          "--phase", "prepare", "--offline", "--output-dir", directory,
                          "--result-root", str(Path(directory) / "images"),
                          *(["--update-rule", "zero", "--source-mode", "joint", "--strength", "0"]
                            if rule == "zero" else [])])
                manifest = load_manifest(Path(output.getvalue().strip().split("[manifest] ")[-1]))
                self.assertEqual(manifest["config"]["adapter"]["query_graft"]["update_rule"], rule)
                names.add(manifest["method"])
                if rule == "zero":
                    self.assertIn("-zero-", manifest["method"])
                    self.assertNotIn("-zero-0-", manifest["method"])
                    self.assertNotIn("source-joint", manifest["method"])
            self.assertEqual(len(names), 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
