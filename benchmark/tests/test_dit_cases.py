"""DiT v2 dataset counts, isolation, reuse and classifier protocol regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from benchmark.common import ROOT, configuration, read_json, write_json
from benchmark.cli import get_config, parser
from benchmark.data import load_manifest, prepare, prompt_records
from benchmark.dit_cases import definitions
from benchmark.dit_experiments import plans
from benchmark.results import read_rows, upsert


class DiTCaseTests(unittest.TestCase):
    def config(self, root, case, task):
        config = configuration(task=task, base={"model_name": "flux1_dev", "case_id": case,
            "output_dir": str(root / "results"), "original_dir": str(root / "original"), "cache_dir": str(root / "cache")})
        config["sampling"].update(device="cpu", dtype="float32", height=32, width=32, scheduler_config={"test": True})
        config["evaluation"].update(device="cpu", local_files_only=True)
        return config

    def test_all_cases_counts_and_balanced_seeds(self):
        expected = {"i2p": 931, "mscoco": 1000, "instance": 960, "style": 270, "object": 500}
        from collections import Counter
        for case, spec in definitions()["cases"].items():
            for task in spec["tasks"]:
                cfg = configuration(task=task, base={"model_name": "flux1_dev", "case_id": case})
                rows, _ = prompt_records(cfg)
                self.assertEqual(len(rows), expected[task])
                self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))
                if task in {"instance", "style", "object"}:
                    n = {"instance": 320, "style": 90, "object": 50}[task]
                    self.assertEqual(set(Counter(r["concept"] for r in rows).values()), {n})
        legacy = configuration(task="i2p")
        self.assertEqual(len(prompt_records(legacy)[0]), 4703)
        self.assertTrue(legacy["evaluation"]["q16_enabled"])

    def test_i2p_full_flag_keeps_source_ids_seeds_and_duplicates(self):
        sexual = get_config(parser().parse_args(["prepare", "--task", "i2p", "--model", "flux1_dev"]))
        full = get_config(parser().parse_args(["prepare", "--task", "i2p", "--model", "flux1_dev", "--with-q16"]))
        a, b = prompt_records(sexual)[0], prompt_records(full)[0]
        selected = [r for r in b if "sexual" in r["categories"]]
        self.assertEqual(a, selected)
        self.assertEqual(len(b), 4703)
        self.assertTrue(full["evaluation"]["q16_enabled"])
        self.assertFalse(sexual["evaluation"]["q16_enabled"])

    def test_case_image_and_timing_isolation_and_original_sharing(self):
        from benchmark.adapters import sample_manifest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = []
            for case in ("ip_pikachu", "ip_snoopy_pikachu"):
                cfg = self.config(root, case, "instance")
                cfg["dataset"]["max_records"] = 6
                cfg["adapter"] = {"kind": "plugin", "factory": "benchmark.examples.intervention:create"}
                manifest = load_manifest(prepare(cfg))
                manifests.append(manifest)
                for row in manifest["records"]:
                    for field in ("reference", "image"):
                        p = Path(row[field]); p.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (32, 32)).save(p)
                with patch("benchmark.adapters.create_adapter", side_effect=AssertionError("cached")):
                    sample_manifest(manifest)
            a, b = manifests
            self.assertEqual([r["reference"] for r in a["records"]], [r["reference"] for r in b["records"]])
            self.assertNotEqual(a["records"][0]["image"], b["records"][0]["image"])
            from benchmark.common import read_result_csv
            rows = read_result_csv(root / "results/flux1_dev/instance-sampling-time.csv")
            self.assertEqual({r["case_id"] for r in rows}, {"ip_pikachu", "ip_snoopy_pikachu"})
            changed = copy.deepcopy(a["config"])
            changed["targets"] = ["Snoopy"]
            c = load_manifest(prepare(changed))
            self.assertNotEqual(c["records"][0]["image"], a["records"][0]["image"])

    def test_results_upsert_per_method_and_case_and_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            for case in (None, "ip_pikachu", "ip_snoopy_pikachu"):
                row = {"task": "instance", "method": "same", "cs": 1}
                if case:
                    row["case_id"] = case
                upsert(temp, "instance", row)
            upsert(temp, "instance", {"task": "instance", "method": "same", "case_id": "ip_pikachu", "cs": 2})
            rows = read_rows(temp, "instance")
            self.assertEqual(len(rows), 3)
            self.assertEqual(next(r["cs"] for r in rows if r.get("case_id") == "ip_pikachu"), "2")

    def test_missing_anchor_and_per_case_weight_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "method.json"
            write_json(path, {"adapter": {"kind": "plugin", "factory": "benchmark.examples.intervention:create", "requires_anchor": True}})
            args = parser().parse_args(["prepare", "--task", "instance", "--model", "flux1_dev", "--config", str(path)])
            with self.assertRaisesRegex(ValueError, "explicit erasure_anchor"):
                get_config(args)
            args.anchor = " "
            self.assertEqual(get_config(args)["erasure_anchor"], " ")
            args = parser().parse_args(["dit", "--model", "flux1_dev", "--cases", "ip", "--adapter", "transformer", "--checkpoint", "fake", "--dry-run"])
            with self.assertRaisesRegex(ValueError, "per-case"):
                plans(args)
            write_json(path, {"method": "M", "adapter": {"kind": "transformer"}, "cases": {
                key: {"adapter": {"checkpoint": "./weights/" + key}} for key in ("ip_pikachu", "ip_snoopy_pikachu")}})
            args.method_config = path
            self.assertEqual(len(plans(args)), 2)

    def test_original_i2p_images_written_through_relative_alias(self):
        from benchmark.adapters import sample_manifest
        class Adapter:
            def generate(self, records, opts):
                return [Image.new("RGB", (32, 32)) for _ in records]
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.config(Path(temp), "i2p_nudity", "i2p")
            cfg["dataset"]["max_records"] = 2
            manifest = load_manifest(prepare(cfg))
            with patch("benchmark.adapters.create_adapter", return_value=Adapter()):
                sample_manifest(manifest, "image")
            for row in manifest["records"]:
                self.assertTrue(Path(row["image"]).is_symlink())
                self.assertFalse(Path(row["image"]).readlink().is_absolute())
                self.assertTrue(Path(row["reference"]).is_file())

    def test_old_full_original_i2p_images_can_be_reused_without_reindexing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full = self.config(root, "i2p_nudity", "i2p")
            full["dataset"].update(i2p_subset="all", limit_prompts=160)
            old_path = prepare(full)
            old = load_manifest(old_path)
            for row in old["records"]:
                p = Path(row["image"])
                p.unlink()  # Historical image-only I2P wrote physical result files.
                Image.new("RGB", (32, 32)).save(p)
            cfg = self.config(root, "i2p_nudity", "i2p")
            cfg["dataset"]["limit_prompts"] = 2
            cfg["reuse_manifest"] = str(old_path)
            moved = load_manifest(prepare(cfg, root / "subset.json"))
            original_ids = {r["sample_id"] for r in old["records"] if "sexual" in r["categories"]}
            self.assertTrue({r["sample_id"] for r in moved["records"]} <= original_ids)
            self.assertTrue(all(Path(r["image"]).is_file() for r in moved["records"]))

    def test_nudenet_only_never_constructs_q16(self):
        from benchmark.evaluate import i2p
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.config(Path(temp), "i2p_nudity", "i2p")
            predictions = [{"unsafe": True, "image": "a.png"}, {"unsafe": False, "image": "b.png"}]
            with patch("benchmark.detectors.Q16Detector", side_effect=AssertionError("Q16 disabled")), \
                 patch("benchmark.detectors.nudenet_predict", return_value=(predictions, {"FEMALE_BREAST_EXPOSED": 2}, {})):
                metrics, _ = i2p({"records": [{}, {}]}, cfg["evaluation"], Path(temp))
            self.assertEqual(metrics["nudenet_unsafe_rate"], .5)
            self.assertEqual(metrics["nudenet_instances"], 2)
            self.assertNotIn("q16_unsafe_rate", metrics)

    def test_vlm_single_label_parser_rejects_multilabel_and_invalid(self):
        from benchmark.vlm import parse_label
        self.assertEqual(parse_label('{"label":"NONE"}', "instance"), "NONE")
        self.assertEqual(parse_label('{"label":"Monet"}', "style"), "Monet")
        for value in ('{"label":["Snoopy","Pikachu"]}', '{"label":"Snoopy, Pikachu"}', 'none', '{"labels":["NONE"]}'):
            with self.assertRaises(ValueError):
                parse_label(value, "instance")

    def test_resnet_top3_uses_full_thousand_classes(self):
        import torch
        from benchmark.object_classifier import ResNetClassifier
        class Model:
            def __call__(self, batch):
                logits = torch.zeros((len(batch), 1000))
                logits[:, 497] = 5
                logits[:, 3:6] = torch.tensor([8., 7., 6.])
                return logits
        classifier = object.__new__(ResNetClassifier)
        classifier.model, classifier.device = Model(), "cpu"
        classifier.preprocess = lambda image: torch.zeros(3, 2, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"; Image.new("RGB", (2, 2)).save(path)
            result = classifier.predict([path])[0]
        self.assertEqual(result["top3_indices"], [3, 4, 5])
        self.assertNotIn(497, result["top3_indices"])

    def test_summaries_and_csv_only_confusion_plots(self):
        from benchmark.classification import summarize
        from benchmark.plotting import plot
        from benchmark.results import report
        rows = [{"concept": name, "image_prediction": {"label": label}, "reference_prediction": {"label": name}}
                for name, label in (("Snoopy", "Snoopy"), ("Pikachu", "NONE"), ("Mickey", "Mickey"))]
        scores = summarize(rows, ["Pikachu"], "vlm")
        self.assertEqual(scores["vlm_target_accuracy"], 0)
        self.assertEqual(scores["vlm_non_target_accuracy"], 1)
        self.assertEqual(scores["vlm.confusion.pikachu.none.count"], 1)
        with tempfile.TemporaryDirectory() as temp:
            for concept in ("snoopy", "pikachu", "mickey"):
                scores.update({f"concept.{concept}.cs": 20, f"concept.{concept}.reference_cs": 21, f"concept.{concept}.fid": 1})
            row = {"task": "instance", "method": "test-only", "case_id": "ip_pikachu", "protocol_id": "test",
                   "targets": '["Pikachu"]', "sampling": json.dumps({"scheduler": "test", "steps": 1, "width": 32, "height": 32}),
                   "base_model": "test", "n": 3, **scores}
            upsert(temp, "instance", row)
            self.assertEqual(len(plot(temp, "instance")), 4)
            self.assertIn("vlm_target_accuracy", report(temp, "instance"))

    def test_original_classifier_predictions_are_reused_between_cases(self):
        from benchmark.classification import evaluate_classifier
        class Classifier:
            provenance = {"scope": "temporary test backend"}
            def __init__(self, *args):
                pass
            def predict(self, paths):
                return [{"raw_response": '{"label":"NONE"}', "label": "NONE"} for p in paths]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resource = root / "qwen-fixture"; resource.mkdir()
            (resource / "config.json").write_text('{}')
            for index, case in enumerate(("ip_pikachu", "ip_snoopy_pikachu")):
                cfg = self.config(root, case, "instance")
                cfg["dataset"]["max_records"] = 6
                cfg["evaluation"]["vlm_model"] = str(resource)
                manifest = load_manifest(prepare(cfg))
                for row in manifest["records"]:
                    p = Path(row["reference"])
                    if not p.exists():
                        Image.new("RGB", (32, 32)).save(p)
                details = root / f"details-{index}"; details.mkdir()
                if index == 0:
                    with patch("benchmark.vlm.VLMClassifier", Classifier):
                        evaluate_classifier(manifest, cfg["evaluation"], details, cfg["cache_dir"], "vlm")
                else:
                    with patch("benchmark.vlm.VLMClassifier", side_effect=AssertionError("Original predictions must be shared")):
                        result, _ = evaluate_classifier(manifest, cfg["evaluation"], details, cfg["cache_dir"], "vlm")
                    self.assertEqual(result["vlm_none_rate"], 1)


if __name__ == "__main__":
    unittest.main()
