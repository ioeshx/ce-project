"""Threshold decisions and prompt distributions, without loading model weights."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from DiT.detect_erase.encoding import TextEncoder
from DiT.detect_erase.similarity import publish_similarity


class SimilarityTests(unittest.TestCase):
    def encoder(self, features, **options):
        encoder = TextEncoder.__new__(TextEncoder)
        encoder.entries = {"Pikachu": {"tau": .8}, "Snoopy": {"tau": .8}}
        encoder.options = dict(similarity="cosine", tau=None, all_detect_prompts=True,
                               multi_concept="top1", erase_targets=["Pikachu"], **options)
        encoder.prototypes = (torch.eye(2),)
        encoder.features = lambda texts: (torch.tensor(features, dtype=torch.float32),)
        return encoder

    def manifest(self, all_detect=True):
        records = [dict(sample_id=f"sample-{index}", seed=seed, prompt=prompt, concept=concept)
                   for index, (seed, prompt, concept) in enumerate([
                       (0, "a photo of Pikachu", "Pikachu"), (1, "a photo of Pikachu", "Pikachu"),
                       (0, "a drawing of Pikachu", "Pikachu"), (0, "a photo of Snoopy", "Snoopy"),
                       (0, "a photo of Mickey", "Mickey")])]
        return dict(case_id="ip_pikachu", task="instance", targets=["Pikachu"], records=records,
                    config=dict(task="instance", evaluation={}, sampling={},
                                adapter=dict(similarity="cosine", tau=.5, all_detect_prompts=all_detect,
                                             detect_targets=["Pikachu", "Snoopy"], checkpoints=[])))

    def gates(self, manifest):
        decisions = []
        for record in manifest["records"]:
            pikachu, snoopy = {"Pikachu": (.75, .25), "Snoopy": (.25, .75), "Mickey": (.25, .125)}[record["concept"]]
            if record["prompt"] == "a drawing of Pikachu":
                pikachu = .5
            top = "Pikachu" if pikachu >= snoopy else "Snoopy"
            predicted = [top] if max(pikachu, snoopy) >= .5 else []
            decisions.append({**record, "scores": {"Pikachu": pikachu, "Snoopy": snoopy},
                              "thresholds": {"Pikachu": .5, "Snoopy": .5}, "threshold_applied": True,
                              "matched": [name for name, value in (("Pikachu", pikachu), ("Snoopy", snoopy)) if value >= .5],
                              "predicted": predicted, "selected": [name for name in predicted if name in manifest["targets"]]})
        return decisions

    def write_gates(self, path, decisions):
        path.write_text("".join(json.dumps(decision) + "\n" for decision in decisions), encoding="utf-8")

    def test_global_cosine_rejects_below_tau_and_preserves_case_intersection(self):
        encoder = self.encoder([[.7, .7], [1, 0], [0, 1], [.8, .6]])
        decisions = encoder.detect(["low", "Pikachu", "Snoopy", "boundary"])
        self.assertEqual(decisions[0]["matched"], [])
        self.assertEqual(decisions[0]["predicted"], [])
        self.assertEqual(decisions[0]["selected"], [])
        self.assertEqual(decisions[1]["selected"], ["Pikachu"])
        self.assertEqual(decisions[2]["predicted"], ["Snoopy"])
        self.assertEqual(decisions[2]["selected"], [])
        self.assertEqual(decisions[3]["selected"], ["Pikachu"])
        self.assertEqual(decisions[3]["scores"]["Pikachu"], decisions[3]["thresholds"]["Pikachu"])
        self.assertTrue(all(decision["threshold_applied"] for decision in decisions))

    def test_override_tau_and_best_eligible_candidate(self):
        encoder = self.encoder([[.8, .6]])
        encoder.entries["Pikachu"]["tau"] = .9
        encoder.entries["Snoopy"]["tau"] = .5
        self.assertEqual(encoder.detect(["q"])[0]["predicted"], ["Snoopy"])
        encoder.options["tau"] = .7
        self.assertEqual(encoder.detect(["q"])[0]["selected"], ["Pikachu"])
        encoder.options["tau"] = .9
        self.assertEqual(encoder.detect(["q"])[0]["predicted"], [])

    def test_local_mean_keeps_all_threshold_matches(self):
        encoder = self.encoder([[1, 1]])
        encoder.options.update(all_detect_prompts=False, tau=.5, multi_concept="mean",
                               erase_targets=["Pikachu", "Snoopy"])
        self.assertEqual(encoder.detect(["q"])[0]["selected"], ["Pikachu", "Snoopy"])

    def test_l2_distance_gating_is_unchanged(self):
        encoder = self.encoder([])
        encoder.options.update(similarity="l2", l2_threshold=.1)
        encoder.prototypes = [(torch.zeros(1, 1, 1), torch.zeros(1, 1)),
                              (torch.ones(1, 1, 1), torch.ones(1, 1))]
        condition = torch.full((1, 1, 1), .4), torch.full((1, 1), .4)
        decision = encoder.detect(["q"], condition)[0]
        self.assertEqual(decision["selected"], ["Pikachu"])
        self.assertFalse(decision["threshold_applied"])
        self.assertIsNone(decision["thresholds"]["Pikachu"])
        encoder.options["all_detect_prompts"] = False
        decision = encoder.detect(["q"], condition)[0]
        self.assertEqual(decision["selected"], [])
        self.assertTrue(decision["threshold_applied"])

    def test_distributions_use_true_concepts_unique_prompts_and_percentages(self):
        manifest = self.manifest()
        decisions = self.gates(manifest)
        # Repeated sampling replaces the old decision, rather than adding another observation.
        stale = copy.deepcopy(decisions[0])
        stale["scores"]["Pikachu"] = -.5
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "gates.jsonl"
            self.write_gates(log, [stale, *decisions])
            report = publish_similarity(manifest, log, set(), directory / "similarity_plot")
            self.assertEqual(report["recorded_sample_count"], 5)
            self.assertEqual(report["unique_prompt_count"], 4)
            groups = {(row["detector"], row["group"]): row for row in report["groups"]}
            self.assertEqual(groups["Pikachu", "target"]["count"], 2)
            self.assertEqual(groups["Pikachu", "target"]["threshold_count"], 2)
            self.assertEqual(groups["Pikachu", "target"]["threshold_percentage"], 100)
            self.assertEqual(groups["Snoopy", "target"]["count"], 1)
            self.assertEqual(groups["Snoopy", "target"]["top1_count"], 1)
            self.assertEqual(groups["Snoopy", "target"]["selected_count"], 0)
            plot_dir = directory / "similarity_plot"
            with (plot_dir / "histogram.csv").open(newline="") as handle:
                histogram = list(csv.DictReader(handle))
            for detector, group in groups:
                self.assertAlmostEqual(sum(float(row["percentage"]) for row in histogram
                                           if row["detector"] == detector and row["group"] == group), 100)
            with (plot_dir / "records.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertNotIn("-0.5", [row["score"] for row in rows])
            for detector in ("pikachu", "snoopy"):
                for suffix in ("png", "pdf"):
                    self.assertGreater((plot_dir / f"{detector}.{suffix}").stat().st_size, 100)

    def test_local_multi_target_has_one_plot_per_target(self):
        manifest = self.manifest(all_detect=False)
        manifest["targets"] = ["Pikachu", "Snoopy"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_gates(directory / "gates.jsonl", self.gates(manifest))
            report = publish_similarity(manifest, directory / "gates.jsonl", set(), directory)
            self.assertEqual(set(report["thresholds"]), {"Pikachu", "Snoopy"})
            manifest["targets"] = ["Pikachu"]
            report = publish_similarity(manifest, directory / "gates.jsonl", set(), directory / "single")
            self.assertEqual(set(report["thresholds"]), {"Pikachu"})
            self.assertFalse((directory / "single" / "snoopy.png").exists())

    def test_i2p_labels_mscoco_negatives_and_missing_records(self):
        manifest = self.manifest()
        manifest.update(case_id="i2p_nudity", task="i2p", targets=["nudity"])
        manifest["config"]["adapter"].update(detect_targets=["nudity"])
        decisions = [{**record, "scores": {"nudity": .5}, "thresholds": {"nudity": .5},
                      "threshold_applied": True, "predicted": ["nudity"], "selected": ["nudity"]}
                     for record in manifest["records"][:2]]
        keys = {(record["sample_id"], record["seed"]) for record in manifest["records"][:2]}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "gates.jsonl"
            self.write_gates(log, decisions)
            report = publish_similarity(manifest, log, keys, directory / "i2p")
            self.assertEqual(report["missing_sample_count"], 3)
            self.assertEqual(report["groups"][0]["count"], 1)
            self.assertEqual(report["groups"][1]["count"], 0)
            self.assertIsNone(report["groups"][1]["threshold_percentage"])
            manifest["task"] = "mscoco"
            report = publish_similarity(manifest, log, keys, directory / "mscoco")
            self.assertEqual(report["groups"][0]["count"], 0)
            self.assertEqual(report["groups"][1]["count"], 1)

    def test_legacy_logs_resolve_checkpoint_tau_without_rewriting_decisions(self):
        manifest = self.manifest()
        manifest["config"]["adapter"]["tau"] = None
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for target in ("Pikachu", "Snoopy"):
                checkpoint = directory / f"{target}.pt"
                torch.save({"tau": .6}, checkpoint)
                manifest["config"]["adapter"]["checkpoints"].append(dict(stage="detect", target=target, checkpoint=str(checkpoint)))
            decisions = self.gates(manifest)
            for decision in decisions:
                decision.pop("thresholds")
                decision.pop("threshold_applied")
            log = directory / "gates.jsonl"
            self.write_gates(log, decisions)
            original = log.read_bytes()
            report = publish_similarity(manifest, log, set(), directory / "plots")
            self.assertEqual(report["thresholds"], {"Pikachu": .6, "Snoopy": .6})
            self.assertFalse(report["threshold_applied"])
            self.assertEqual(log.read_bytes(), original)

    def test_sampling_logs_before_generation_without_tracking(self):
        from DiT.detect_erase.prompts import PromptModel

        adapter = PromptModel.__new__(PromptModel)
        adapter.flux = False
        condition = torch.zeros(1, 1, 2), torch.zeros(1, 2)
        adapter.encode = lambda texts: condition
        manifest = self.manifest()
        decisions = self.gates(manifest)[:1]
        adapter.negatives = lambda texts, encoded: (condition, decisions)
        adapter.latents = lambda seeds, sampling: torch.zeros(1, 1, 1, 1)
        adapter.pipe = Mock(side_effect=RuntimeError("generation stopped"))
        sampling = dict(height=8, width=8, steps=1, guidance_scale=4.5, max_sequence_length=2)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "gates.jsonl"
            adapter.options = {"gate_log": str(log)}
            with self.assertRaisesRegex(RuntimeError, "generation stopped"):
                adapter.generate(manifest["records"][:1], sampling)
            self.assertEqual(json.loads(log.read_text())["scores"], decisions[0]["scores"])

    def test_detect_only_writes_reports_without_swanlab(self):
        from DiT.detect_erase import sample

        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case_dir = directory / "exps" / manifest["case_id"]
            case_dir.mkdir(parents=True)
            manifest["config"]["adapter"]["gate_log"] = str(case_dir / "detect_gates.jsonl")
            encoder = self.encoder([[1, 0], [1, 0], [.8, .6], [0, 1], [.7, .7]])
            encoder.prepare_detector = Mock()
            encoder.close = Mock()
            args = SimpleNamespace(track=False, detect_only=True, device=None, batch_size=8, skip_vlm=False,
                                   real_nude_mapping=directory / "unused.csv")
            with patch.object(sample, "TextEncoder", return_value=encoder), patch.object(sample, "load_concept_prompts", return_value={}):
                sample.run_cases(args, directory, [manifest], {})
            self.assertTrue((case_dir / "similarity_plot" / "pikachu.png").exists())
            self.assertTrue(json.loads((case_dir / "detect_summary.json").read_text())["threshold_applied"])
            with (case_dir / "detect_records.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(json.loads(rows[0]["predicted_detect_prompts"]), ["Pikachu"])

    def test_image_sampling_publishes_similarity_without_tracking(self):
        from DiT.detect_erase import sample

        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            case_dir = directory / "exps" / manifest["case_id"]
            case_dir.mkdir(parents=True)
            log = case_dir / "gates.jsonl"
            manifest["config"]["adapter"]["gate_log"] = str(log)
            args = SimpleNamespace(track=False, detect_only=False, device=None, batch_size=8,
                                   real_nude_mapping=directory / "unused.csv", skip_vlm=False,
                                   phase="sample", device_map=None, cpu_offload=False, offline=True)
            with patch("benchmark.adapters.sample_manifest", side_effect=lambda *a, **kw: self.write_gates(log, self.gates(manifest))):
                sample.run_cases(args, directory, [manifest], {})
            self.assertTrue((case_dir / "similarity_plot" / "pikachu.png").exists())
            self.assertTrue((case_dir / "trigger_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
