"""Trace plots and saved values must reconstruct the actual mask decisions."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from benchmark.tests import test_query_graft as fixtures


class TraceDecisionTests(unittest.TestCase):
    adapter = fixtures.QueryGraftTests.adapter

    def capture_figure(self, figures, figure, path, **kwargs):
        # Inspect plotted data before save_step_diagnostics clears the figure.
        figures[str(path)] = {
            "title": figure._suptitle.get_text(),
            "maps": [np.asarray(axis.images[0].get_array()).copy()
                     for axis in figure.axes if axis.images],
            "labels": [axis.get_xlabel() for axis in figure.axes],
        }

    def test_step_plots_show_actual_selected_input_and_candidate_mask(self):
        from matplotlib.figure import Figure

        cases = ((False, False, False, 1), (True, False, False, 2),
                 (True, True, False, 1), (False, False, True, 1), (False, True, True, 2))
        for flux, aggre, policy, pairs in cases:
            with self.subTest(flux=flux, aggre=aggre, policy=policy, pairs=pairs):
                options = dict(my_layer_aggre=aggre, my_layer_policy=policy, mask_rule="anchor_min")
                if pairs == 2:
                    options["pairs"] = [{"target": "Pikachu", "anchor": "animal"},
                                        {"target": "Snoopy", "anchor": "cartoon"}]
                graft, _, call, _, _ = self.adapter(flux, "av", **options)
                graft.localizers = [0]
                graft.active = True
                with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
                    before = graft.forward(**call)[0].clone()
                    masks, winner = graft.masks.clone(), graft.winner.clone()
                    calls = graft.forward_calls
                    graft.trace_dirs = [Path(directory) / str(i) for i in range(graft.batch)]
                    graft.settings["trace"] = True
                    figures = {}
                    with patch.object(Figure, "savefig", autospec=True,
                                      side_effect=lambda *args, **kwargs: self.capture_figure(figures, *args, **kwargs)):
                        after = graft.forward(**call)[0]
                    torch.testing.assert_close(after, before, atol=0, rtol=0)
                    torch.testing.assert_close(graft.masks, masks, atol=0, rtol=0)
                    torch.testing.assert_close(graft.winner, winner, atol=0, rtol=0)
                    self.assertEqual(graft.forward_calls, 2 * calls)
                    for i, destination in enumerate(graft.trace_dirs):
                        data = load_file(destination / "tensors" / "step_0000.safetensors")
                        info = graft.step_traces[i][0]
                        figure = figures[str(destination / "timesteps" / "step_0000.png")]
                        soft = data["localization_scores_mean"]
                        self.assertFalse(torch.allclose(soft, data["concept_scores_mean"]))
                        self.assertEqual(info["aggregation"], "layer_vote" if aggre else "soft_mean")
                        self.assertEqual(info["softmax_scope"], "per_pair" if policy else "all_concepts")
                        if aggre:
                            reconstructed = data["vote_fraction"] > data["decision_threshold"][:, None, None]
                            input_map = data["vote_fraction"]
                            self.assertIn("eta=", figure["title"])
                        elif policy:
                            reconstructed = (soft[..., 1] < soft[..., 0]) & (soft[..., 1] < soft[..., 2])
                            input_map = soft[..., 0]
                            self.assertNotIn("decision_threshold", data)
                            self.assertIsNone(info["decision_threshold"])
                        elif pairs == 1:
                            cutoff = soft[..., 0].mean((-2, -1))
                            torch.testing.assert_close(data["decision_threshold"], cutoff)
                            reconstructed = soft[..., 0] > data["decision_threshold"][:, None, None]
                            input_map = soft[..., 0]
                        else:
                            labels = soft.reshape(pairs, -1, 3).transpose(0, 1).flatten(1).argmax(-1)
                            labels = labels.reshape(*graft.grid)
                            torch.testing.assert_close(data["concept_argmax"], labels)
                            reconstructed = labels[None] == (3 * torch.arange(pairs))[:, None, None]
                            input_map = soft[..., 0]
                            self.assertIsNone(info["decision_threshold"])
                        torch.testing.assert_close(data["mask"], reconstructed)
                        for pair in range(pairs):
                            np.testing.assert_array_equal(figure["maps"][3 * pair], data["applied_mask"][pair])
                            np.testing.assert_array_equal(figure["maps"][3 * pair + 1], input_map[pair])
                            np.testing.assert_array_equal(figure["maps"][3 * pair + 2], data["mask"][pair])
                        if not aggre:
                            self.assertNotIn("eta=", figure["title"])

    def test_generate_summary_uses_selected_layers_and_keeps_untriggered_steps(self):
        from matplotlib.figure import Figure

        graft, _, call, _, _ = self.adapter(False, "av", trace=True,
                                           my_layer_aggre=False, my_layer_policy=False)
        graft.localizers = [0]

        class SamplingPipe(SimpleNamespace):
            def __call__(self, **kwargs):
                for step in range(kwargs["num_inference_steps"]):
                    current = dict(call)
                    current["timestep"] = torch.full_like(call["timestep"], 1 - step / 3)
                    graft.settings["rho"] = 1 if step == 1 else 0
                    self.transformer(**current)
                    kwargs["callback_on_step_end"](self, step, current["timestep"][0], {})
                return SimpleNamespace(images=[Image.new("RGB", (4, 4)) for _ in kwargs["prompt"]])

        graft.pipe = SamplingPipe(**vars(graft.pipe))
        sampling = {**graft.config["sampling"], "steps": 3, "guidance_scale": 4.5}
        with tempfile.TemporaryDirectory() as directory:
            records = [{"sample_id": f"sample_{i}", "prompt": "person", "seed": 42 + i,
                        "image": str(Path(directory) / "images" / "generated" / f"sample_{i}.png")}
                       for i in range(2)]
            figures = {}
            with patch.object(Figure, "savefig", autospec=True,
                              side_effect=lambda *args, **kwargs: self.capture_figure(figures, *args, **kwargs)):
                graft.generate(records, sampling)
            for destination in graft.trace_dirs:
                steps = [load_file(destination / "tensors" / f"step_{step:04d}.safetensors") for step in range(3)]
                summary = load_file(destination / "summary.safetensors")
                self.assertFalse(steps[1]["applied_mask"].any())
                for name, source in (("applied_mask_mean", "applied_mask"),
                                     ("concept_scores_mean", "concept_scores_mean"),
                                     ("localization_scores_mean", "localization_scores_mean")):
                    expected = torch.stack([step[source].float() for step in steps]).mean(0)
                    torch.testing.assert_close(summary[name], expected)
                figure = figures[str(destination / "summary.png")]
                np.testing.assert_array_equal(figure["maps"][1], summary["localization_scores_mean"][0, ..., 0])
                self.assertFalse(torch.allclose(summary["localization_scores_mean"], summary["concept_scores_mean"]))
                self.assertIn("no new threshold", figure["title"])
                self.assertIn("1 selected blocks", figure["title"])
                metadata = json.loads((destination / "metadata.json").read_text())
                self.assertEqual(len(metadata["steps"]), 3)
                self.assertTrue(all(step["region_policy"] == "spatial_mean" for step in metadata["steps"]))
            # A new sampling call clears the sums and creates no trace files when disabled.
            graft.settings["trace"] = False
            records = copy.deepcopy(records)
            for record in records:
                record["image"] = str(Path(directory) / "disabled" / "images" / "generated" / "sample.png")
            graft.generate(records, sampling)
            self.assertFalse(graft.sample_trace_sums)
            self.assertFalse((Path(directory) / "disabled" / "traces").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
