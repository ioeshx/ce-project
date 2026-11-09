"""Offline anchor overrides preserve the benchmark protocol and cache matching."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark.common import read_json, write_json
from benchmark.data import prompt_records
from DiT.steer import run


class AnchorTests(unittest.TestCase):
    def preview(self, root, *extra):
        with patch.object(run, "run") as execute:
            run.main(["--dry-run", "--device", "cpu", "--run-root", str(root), *extra])
        return execute.call_args.args[:2]

    def pairs(self, settings):
        return [{"target": target, "anchor": anchor,
                 "pairs": run.prompt_pairs(settings["group"], target, anchor, settings["calibration_prompts"])}
                for target, anchor in zip(settings["targets"], settings["anchors"])]

    def test_custom_anchors_preserve_all_benchmark_records_and_protocols(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs"
            original_args, original = self.preview(root)
            custom_args, custom = self.preview(root, "--target-anchor-config", str(run.DIRECTORY / "conf.json"))
            self.assertNotEqual(original_args.method, custom_args.method)
            self.assertTrue(custom_args.method.endswith("-anchors-conf"))
            for (case_id, before, configs), (other_id, after, custom_configs) in zip(original, custom):
                with self.subTest(case=case_id):
                    self.assertEqual(case_id, other_id)
                    expected = ["clothed"] if before["group"] == "i2p" else [""] * len(before["targets"])
                    self.assertEqual(after["anchors"], expected)
                    for a, b in zip(configs, custom_configs):
                        for field in ("dataset", "sampling", "evaluation", "targets"):
                            self.assertEqual(a[field], b[field])
                        self.assertEqual(prompt_records(a), prompt_records(b))
                    old_pairs, new_pairs = self.pairs(before), self.pairs(after)
                    self.assertEqual([[p["positive"] for p in entry["pairs"]] for entry in old_pairs],
                                     [[p["positive"] for p in entry["pairs"]] for entry in new_pairs])
                    old_identity = run.offline_identity(before, configs[0]["sampling"], old_pairs)
                    new_identity = run.offline_identity(after, custom_configs[0]["sampling"], new_pairs)
                    self.assertEqual(old_identity == new_identity, before["group"] == "i2p")
                    if before["group"] != "i2p":
                        self.assertEqual(new_pairs[0]["pairs"][0]["negative"], "tench")
            self.assertFalse(root.exists())
            explicit, _ = self.preview(root, "--target-anchor-config", str(run.DIRECTORY / "conf.json"),
                                       "--method", "Anchor-ablation")
            self.assertEqual(explicit.method, "Anchor-ablation")

    def test_mapping_formats_and_invalid_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "anchors.json"
            for value in ({"Pikachu": ""}, {"concept_pair": {"Pikachu": ""}}):
                write_json(path, value)
                self.assertEqual(run.load_target_anchors(path, ["Pikachu"]), {"Pikachu": ""})
            for value, message in (({}, "missing anchor"), ({"Pikachu": None}, "must be a string"),
                                   ({"concept_pair": []}, "expected a target-to-anchor mapping")):
                write_json(path, value)
                with self.assertRaisesRegex(ValueError, message):
                    run.load_target_anchors(path, ["Pikachu"])

    def test_empty_anchor_prompts_and_default_templates(self):
        self.assertEqual(run.prompt_pairs("ip", "Pikachu", "Cartoon Character", 1),
                         [{"positive": "tench with Pikachu", "negative": "tench with Cartoon Character"}])
        self.assertEqual(run.prompt_pairs("style", "Van Gogh", "Artist Style", 1),
                         [{"positive": "tench, Van Gogh style", "negative": "tench, Artist Style style"}])
        self.assertEqual(run.prompt_pairs("object", "church", "Building", 1),
                         [{"positive": "tench with church", "negative": "tench with Building"}])
        self.assertEqual(run.prompt_pairs("i2p", "nudity", "clothed", 1),
                         [{"positive": "a girl , nudity", "negative": "a girl , clothed"}])
        self.assertEqual(run.prompt_pairs("i2p", "nudity", "", 1)[0]["negative"], "a girl")

    def test_switching_anchors_collects_once_and_reuses_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def fake_fit(factory, kwargs, **unused):
                calls.append(kwargs)
                checkpoint = Path(kwargs["checkpoint"])
                checkpoint.mkdir(parents=True)
                (checkpoint / "steering.pt").write_bytes(b"cache fixture")
                write_json(checkpoint / "metadata.json", {"settings": kwargs["settings"],
                    "sampling": kwargs["config"]["sampling"], "prompt_pairs": self.pairs(kwargs["settings"])})

            base = ["--cases", "ip_pikachu", "--phase", "fit", "--device", "cpu", "--run-root", temp]
            custom = ["--target-anchor-config", str(run.DIRECTORY / "conf.json")]
            with patch.object(run, "run_function", side_effect=fake_fit), \
                 patch.object(run, "resolve_scheduler", side_effect=lambda cfg: cfg["sampling"].update(scheduler_config={})), \
                 contextlib.redirect_stdout(io.StringIO()):
                for extra in ([], custom, [], custom):
                    self.assertEqual(run.main(base + extra), 0)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0]["checkpoint"], calls[1]["checkpoint"])
            saved = read_json(Path(calls[1]["checkpoint"]) / "metadata.json")
            self.assertEqual(saved["prompt_pairs"][0]["pairs"][0]["negative"], "tench")
            self.assertTrue(all(call["statistics_checkpoint"] is None for call in calls))


if __name__ == "__main__":
    unittest.main()
