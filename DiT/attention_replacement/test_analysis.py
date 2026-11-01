"""Analytical fixtures for aggregation axes, cohorts, branches, and mask checks."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from DiT.attention_replacement.analyze import (
    collect_statistics, discover_runs, reduce_maps, render_group, Reservoir, write_overview,
    analyze_results, sample_figure, sample_figures,
)


def fixture(root, missing=False, labels=True):
    run = root / "results" / "traces" / "run"
    sites = [{"id": f"layer_{layer:03d}_{branch}", "layer": layer, "branch": branch,
              "grid": [2, 2]} for layer, branch in ((0, "attn"), (0, "attn2"), (1, "attn"))]
    for sample, concept in enumerate(("Pikachu", "Mickey")):
        folder = run / f"sample_{sample}"
        folder.mkdir(parents=True)
        metadata = {"prototype_means": {"Pikachu": {}}, "sites": sites,
                    "settings": {"pairs": [{"target": "Pikachu", "anchor": "Cartoon Character"}],
                                 "step_start": 0, "step_end": 2, "tau": 0., "tau_by_concept": {}},
                    "sampling": {"steps": 2}, "schedule": {"timesteps": [1000., 500.]},
                    "record": {"concept": concept if labels else "", "seed": sample,
                               "prompt": f"a {concept}", "image": str(folder / "absent.png")}}
        (folder / "metadata.json").write_text(json.dumps(metadata))
        for step in (0, 1):
            directory = folder / f"step_{step:04d}"
            directory.mkdir()
            for site in sites:
                if missing and sample == 1 and step == 1 and site["id"] == "layer_001_attn":
                    continue
                value = 1 + 10 * sample + 2 * site["layer"] + step + 100 * (site["branch"] == "attn2")
                tar = np.full((2, 2), value, dtype=np.float32)
                margin = np.array([[-1., 0.], [1., 2.]], dtype=np.float32)
                mask = margin > 0
                save_file({"d_tar_0": tar, "d_anc_0": tar + margin, "margin_0": margin,
                           "mask_0": mask, "mask_union": mask,
                           "winner": np.where(mask, 0, -1).astype(np.int32)},
                          directory / (site["id"] + ".safetensors"))
    return run


class AnalysisTests(unittest.TestCase):
    def test_three_axes_branch_separation_and_prompt_cohorts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root)
            self.assertEqual(discover_runs(root / "results"), [run])
            output = root / "analysis"
            manifest = collect_statistics(run, output, reservoir_size=32)
            self.assertEqual(manifest["validation"]["files_read"], 12)
            self.assertEqual(manifest["validation"]["missing_files"], 0)
            self.assertEqual(manifest["validation"]["mask_vs_margin_mismatches"], 0)
            data = np.load(output / "concept_00_Pikachu/attn/statistics.npz")
            np.testing.assert_allclose(data["all__layer_maps"][:, 0, 0, 0], [6.5, 8.5])
            np.testing.assert_allclose(data["all__step_maps"][:, 0, 0, 0], [7., 8.])
            self.assertEqual(data["all__global_maps"][0, 0, 0], 7.5)
            self.assertEqual(data["target_prompt__global_maps"][0, 0, 0], 2.5)
            self.assertEqual(data["other_prompts__global_maps"][0, 0, 0], 12.5)
            self.assertEqual(data["all__global_maps"][3].mean(), .5)
            branch = np.load(output / "concept_00_Pikachu/attn2/statistics.npz")
            self.assertEqual(branch["all__global_maps"][0, 0, 0], 106.5)

    def test_missing_files_are_counted_without_zero_fill(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root, missing=True)
            output = root / "analysis"
            manifest = collect_statistics(run, output, reservoir_size=32)
            self.assertEqual(manifest["validation"]["missing_files"], 1)
            data = np.load(output / "concept_00_Pikachu/attn/statistics.npz")
            self.assertAlmostEqual(data["all__global_maps"][0, 0, 0], 46 / 7)
            np.testing.assert_array_equal(data["all__counts"], [[2, 2], [2, 1]])

    def test_complete_render_and_threshold_strictness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root, labels=False)
            output = root / "analysis"
            manifest = collect_statistics(run, output, reservoir_size=32)
            group = next(g for g in manifest["groups"] if g["branch"] == "attn")
            render_group(output, manifest, group)
            folder = output / group["directory"]
            self.assertTrue((folder / "all/global_01.png").is_file())
            self.assertTrue((folder / "all/by_layer_01.png").is_file())
            self.assertTrue((folder / "all/by_timestep_01.png").is_file())
            import csv
            with (folder / "threshold_sweep.csv").open() as stream:
                at_zero = next(row for row in csv.DictReader(stream) if float(row["tau"]) == 0)
            self.assertEqual(float(at_zero["all_coverage"]), .5)

    def test_reduction_uses_observation_counts(self):
        sums = np.zeros((1, 2, 1, 1, 1))
        sums[0, :, 0, 0, 0] = [2, 30]
        reduced = reduce_maps(sums, np.array([[1, 3]]))
        self.assertEqual(reduced["global_maps"].item(), 8)

    def test_reservoir_is_bounded_and_tracks_all_pixels(self):
        reservoir = Reservoir(4)
        values = np.arange(16, dtype=np.float32).reshape(4, 4)
        for _ in range(10):
            reservoir.add(values, values, values)
        self.assertEqual(reservoir.finish().shape, (4, 3))
        self.assertEqual(reservoir.pixels, 160)

    def test_batch_overview_matches_experiment_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root)
            output = root / "analysis"
            destination = output / "experiment"
            manifest = collect_statistics(run, destination, reservoir_size=32)
            write_overview(output, [{"output": str(destination), "samples": 2,
                                     "validation": manifest["validation"]}])
            import csv
            with (output / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            row = next(r for r in rows if r["branch"] == "attn" and r["prompt_group"] == "all")
            self.assertEqual(float(row["d_target"]), 7.5)
            self.assertEqual(json.loads((output / "validation_summary.json").read_text())["files_read"], 12)

    def test_examples_cover_all_seeds_and_pairs_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root)
            for sample in sorted(run.iterdir()):
                path = sample / "metadata.json"
                metadata = json.loads(path.read_text())
                metadata["record"]["prompt"] = "a <Pikachu> & friend"
                metadata["settings"]["pairs"].append({"target": "Snoopy", "anchor": "Cartoon Character"})
                path.write_text(json.dumps(metadata))
                from safetensors.numpy import load_file
                for tensor_path in sample.glob("step_*/*.safetensors"):
                    tensors = load_file(tensor_path)
                    tensors.update({f"{name}_1": tensors[f"{name}_0"]
                                    for name in ("d_tar", "d_anc", "margin", "mask")})
                    save_file(tensors, tensor_path)
            output = root / "analysis"
            with patch("DiT.attention_replacement.analyze.sample_figure", wraps=sample_figure) as render:
                summary = sample_figures(run, output)
                self.assertEqual(render.call_count, 2)
                self.assertTrue(all(len(call.args[2]["settings"]["pairs"]) == 2
                                    for call in render.call_args_list))
            self.assertEqual(summary["visualized_samples"], 2)
            self.assertEqual(summary["unique_prompts"], 1)
            self.assertEqual(summary["missing_traces"], 0)
            import csv
            folder = output / "sample_examples"
            with (folder / "samples.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["seed"] for row in rows], ["0", "1"])
            self.assertTrue(all((folder / row["figure"]).is_file() for row in rows))
            self.assertIn("a &lt;Pikachu&gt; &amp; friend", (folder / "index.html").read_text())
            with patch("DiT.attention_replacement.analyze.sample_figure", side_effect=AssertionError("must reuse")):
                resumed = sample_figures(run, output)
            self.assertEqual(resumed["reused"], 2)
            self.assertEqual(resumed["generated"], 0)

    def test_examples_only_filters_runs_and_preserves_aggregate_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = fixture(root / "short")
            long_run = fixture(root / "long")
            for path in long_run.glob("*/metadata.json"):
                metadata = json.loads(path.read_text())
                metadata["settings"]["step_end"] = 40
                path.write_text(json.dumps(metadata))
            self.assertEqual(discover_runs(root, step_end=2), [run])
            output = root / "analysis"
            destination = output / run.name
            collect_statistics(run, destination, reservoir_size=32)
            statistics = destination / "concept_00_Pikachu/attn/statistics.npz"
            before = statistics.read_bytes(), statistics.stat().st_mtime_ns
            (output / "analysis_runs.json").write_text("existing batch summary")
            with patch("DiT.attention_replacement.analyze.render_group", side_effect=AssertionError("must not replot")):
                reports = analyze_results(run, output, workers=1, examples_only=True, step_end=2)
            self.assertEqual(reports[0]["sample_examples"]["visualized_samples"], 2)
            self.assertEqual(before, (statistics.read_bytes(), statistics.stat().st_mtime_ns))
            self.assertEqual((output / "analysis_runs.json").read_text(), "existing batch summary")
            self.assertTrue((output / "sample_examples_runs.json").is_file())
            self.assertIn('run/sample_examples/index.html', (output / "sample_examples_index.html").read_text())


if __name__ == "__main__":
    unittest.main()
