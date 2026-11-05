"""Check completed velocity replacement, CFG isolation and latent patch alignment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from benchmark.tests import test_query_graft as fixtures


class VelocityTests(unittest.TestCase):
    adapter = fixtures.QueryGraftTests.adapter

    def force_mask(self, graft):
        finish = graft.finish_mask

        def finish_with_votes():
            graft.votes = torch.tensor([[[1, 1, 0, 0]], [[0, 1, 0, 1]]]) * len(graft.localizers)
            finish()
        return patch.object(graft, "finish_mask", side_effect=finish_with_votes)

    def capture_native(self, graft, outputs, inputs):
        native = graft.native_forward

        def capture(**call):
            result = native(**call)
            outputs[graft.mode] = result[0].detach().clone()
            inputs[graft.mode] = call
            return result
        return patch.object(graft, "native_forward", side_effect=capture)

    def output_mask(self, graft):
        selected = graft.winner >= 0
        if graft.flux:
            return selected[..., None]
        # Independent latent-space reference: each image token covers 2 x 2 values.
        return selected.reshape(graft.batch, 1, 2, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)

    def test_replaces_completed_source_velocity_before_cfg_for_all_source_modes(self):
        for flux, cfg in ((False, True), (False, False), (True, False)):
            for return_dict in (False, True):
                for mode in ("anchor", "joint", "logits", "values", "extrapolate"):
                    with self.subTest(flux=flux, cfg=cfg, return_dict=return_dict, mode=mode):
                        graft, _, call, _, _ = self.adapter(
                            flux, "velocity", source_mode=mode, source_scale=2,
                            source_user_weight=0.5, source_anchor_weight=2)
                        if not flux and not cfg:
                            for name in ("hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"):
                                call[name] = call[name][-graft.batch:]
                        call["return_dict"] = return_dict
                        raw, inputs = {}, {}
                        with torch.inference_mode():
                            graft.prepare_sources([{"prompt": "man"}, {"prompt": "woman"}])
                            graft.active = True
                            with self.force_mask(graft), self.capture_native(graft, raw, inputs):
                                output = graft.forward(**call)
                        self.assertEqual(isinstance(output, tuple), not return_dict)
                        actual = output[0]
                        user = raw["user"][-graft.batch:]
                        source = raw["anchor"]
                        if mode == "extrapolate":
                            source = user + 2 * (source - user)
                        expected = raw["user"].clone()
                        expected[-graft.batch:] = torch.where(self.output_mask(graft), source, user)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if return_dict:
                            self.assertIs(output.sample, output["sample"])
                        # Source uses the current positive x_t/t, with its own pooled condition.
                        for name in ("hidden_states", "timestep"):
                            torch.testing.assert_close(inputs["anchor"][name], call[name][-graft.batch:], rtol=0, atol=0)
                        anchor = graft.settings["pairs"][0]["anchor"]
                        pooled = graft.anchor_conditions[anchor][1].expand(graft.batch, -1)
                        torch.testing.assert_close(inputs["anchor"]["pooled_projections"], pooled, rtol=0, atol=0)
                        self.assertEqual(graft.forward_calls, 2)
                        if cfg:
                            torch.testing.assert_close(actual[:graft.batch], raw["user"][:graft.batch], rtol=0, atol=0)
                            negative, positive = actual.chunk(2)
                            guided = negative + 4.5 * (positive - negative)
                            native_guided = raw["user"][:graft.batch] + 4.5 * (
                                user - raw["user"][:graft.batch])
                            torch.testing.assert_close(guided - native_guided, 4.5 * (positive - user),
                                                       rtol=1e-5, atol=1e-6)

    def test_velocity_mask_follows_rectangular_latent_patch_grid(self):
        for flux in (False, True):
            with self.subTest(flux=flux):
                graft, _, call, _, _ = self.adapter(flux, "velocity")
                graft.grid = (2, 3)
                if flux:
                    call["hidden_states"] = torch.randn(2, 6, 4)
                    call["img_ids"] = torch.tensor([[0, row, column] for row in range(2) for column in range(3)])
                else:
                    call["hidden_states"] = torch.randn(4, 4, 4, 6)
                finish = graft.finish_mask

                def finish_with_votes():
                    graft.votes = torch.tensor([[[0, 1, 0, 0, 0, 1]], [[1, 0, 1, 0, 1, 0]]]) * len(graft.localizers)
                    finish()

                raw = {}
                graft.active = True
                with patch.object(graft, "finish_mask", side_effect=finish_with_votes), \
                        self.capture_native(graft, raw, {}), torch.inference_mode():
                    actual = graft.forward(**call)[0]
                mask = graft.winner >= 0
                mask = (mask[..., None] if flux else mask.reshape(2, 1, 2, 3)
                        .repeat_interleave(2, -2).repeat_interleave(2, -1))
                expected = raw["user"].clone()
                expected[-graft.batch:] = torch.where(mask, raw["anchor"], raw["user"][-graft.batch:])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_blend_and_add_apply_to_velocity_with_no_change_outside_mask(self):
        for flux in (False, True):
            for rule in ("blend", "add"):
                with self.subTest(flux=flux, rule=rule):
                    graft, _, call, _, _ = self.adapter(
                        flux, "velocity", update_rule=rule, strength=0.25)
                    raw = {}
                    graft.active = True
                    with self.force_mask(graft), self.capture_native(graft, raw, {}), torch.inference_mode():
                        actual = graft.forward(**call)[0]
                    user, source = raw["user"][-graft.batch:], raw["anchor"]
                    changed = user + 0.25 * (source - user) if rule == "blend" else user + 0.25 * source
                    expected = raw["user"].clone()
                    expected[-graft.batch:] = torch.where(self.output_mask(graft), changed, user)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_velocity_diagnostics_measure_actual_final_output_values(self):
        from safetensors.torch import load_file
        graft, _, call, _, _ = self.adapter(False, "velocity", trace=True)
        with tempfile.TemporaryDirectory() as directory:
            graft.trace_dirs = [Path(directory) / str(i) for i in range(graft.batch)]
            raw = {}
            graft.active = True
            with self.force_mask(graft), self.capture_native(graft, raw, {}), torch.inference_mode():
                actual = graft.forward(**call)[0]
            for i, destination in enumerate(graft.trace_dirs):
                tensors = load_file(destination / "step_0000.safetensors")
                mask = self.output_mask(graft)[i].expand_as(actual[-graft.batch:][i])
                delta = (actual[-graft.batch:][i] - raw["user"][-graft.batch:][i])[mask]
                torch.testing.assert_close(tensors["activation_delta_rms"][0, 0], delta.square().mean().sqrt())
                self.assertEqual(tensors["activation_value_count"][0, 0].item(), mask.sum().item())
                self.assertEqual(tensors["activation_token_count"][0, 0].item(), 2)
                self.assertEqual(tensors["edit_layer_indices"].tolist(), [-1])
                self.assertTrue((destination / "step_0000.png").is_file())
                self.assertEqual(graft.step_traces[i][0]["representation"], "velocity")

    def test_velocity_has_a_single_output_site(self):
        for selection in ("all", [0]):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "one final Transformer output site"):
                self.adapter(False, "velocity", edit_layers=selection)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
