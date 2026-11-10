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
                 "pairs": run.prompt_pairs(settings["group"], target, anchor, settings["calibration_prompts"],
                                           mode=settings.get("prompt_mode", "casteer"))}
                for target, anchor in zip(settings["targets"], settings["anchors"])]

    def test_custom_anchors_preserve_all_benchmark_records_and_protocols(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs"
            default_args, defaults = self.preview(root)
            original_args, original = self.preview(root, "--use-anchor")
            custom_args, custom = self.preview(root, "--use-anchor", "--target-anchor-config", str(run.DIRECTORY / "conf.json"))
            ignored_args, ignored = self.preview(root, "--target-anchor-config", str(Path(temp) / "unused.json"))
            self.assertTrue(default_args.method.endswith("-context"))
            self.assertTrue(original_args.method.endswith("-anchors"))
            self.assertEqual(ignored_args.method, default_args.method)
            self.assertEqual(ignored, defaults)
            self.assertNotEqual(original_args.method, custom_args.method)
            self.assertTrue(custom_args.method.endswith("-anchors-conf"))
            for (case_id, before, configs), (other_id, after, custom_configs), (_, default, _) in zip(original, custom, defaults):
                with self.subTest(case=case_id):
                    self.assertEqual(case_id, other_id)
                    expected = ["clothed"] if before["group"] == "i2p" else [""] * len(before["targets"])
                    self.assertEqual(default["anchors"], expected)
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

    def test_bare_subject_and_matched_templates(self):
        concepts = [("ip", "Pikachu", "Cartoon Character"), ("object", "church", "Building"),
                    ("style", "Van Gogh", "Artist Style"), ("i2p", "nudity", "clothed")]
        for group, target, anchor in concepts:
            with self.subTest(group=group):
                self.assertEqual(run.prompt_pairs(group, target, anchor, mode="bare"),
                                 [{"positive": target, "negative": anchor}])
                self.assertEqual(run.prompt_pairs(group, target, "", mode="bare")[0]["negative"], "")
                for mode in ("subject", "matched"):
                    pairs = run.prompt_pairs(group, target, anchor, mode=mode)
                    empty = run.prompt_pairs(group, target, "", mode=mode)
                    self.assertEqual(len(pairs), 8)
                    self.assertEqual(len({tuple(pair.values()) for pair in pairs}), 8)
                    self.assertEqual(run.prompt_pairs(group, target, anchor, 3, mode=mode), pairs[:3])
                    for pair, without_anchor in zip(pairs, empty):
                        self.assertEqual(pair["positive"].replace(target, anchor), pair["negative"])
                        self.assertEqual(pair["positive"], without_anchor["positive"])
                        self.assertTrue(pair["positive"].startswith(without_anchor["negative"]))
                        self.assertNotIn(target, without_anchor["negative"])
                        self.assertNotIn(anchor, without_anchor["negative"])
        self.assertEqual(run.prompt_pairs("ip", "Pikachu", "", 1, mode="subject"),
                         [{"positive": "a photo of Pikachu", "negative": "a photo"}])
        self.assertEqual(run.prompt_pairs("object", "church", "Building", 1, mode="matched"),
                         [{"positive": "a scene in a park, with church as the main subject",
                           "negative": "a scene in a park, with Building as the main subject"}])
        self.assertEqual(run.prompt_pairs("style", "Van Gogh", "", 1, mode="matched"),
                         [{"positive": "a landscape painting with mountains and a river in the style of Van Gogh",
                           "negative": "a landscape painting with mountains and a river"}])

    def test_prompt_modes_preserve_benchmark_and_validate_pair_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs"
            _, defaults = self.preview(root)
            for mode, count in (("bare", 1), ("subject", 8), ("matched", 8)):
                args, experiments = self.preview(root, "--prompt-mode", mode)
                self.assertTrue(args.method.endswith(f"-prompts-{mode}"))
                for (case_id, original, configs), (other_id, settings, mode_configs) in zip(defaults, experiments):
                    with self.subTest(mode=mode, case=case_id):
                        self.assertEqual(case_id, other_id)
                        self.assertEqual(settings["anchors"], original["anchors"])
                        self.assertEqual(settings["prompt_mode"], mode)
                        self.assertTrue(all(len(entry["pairs"]) == count for entry in self.pairs(settings)))
                        for before, after in zip(configs, mode_configs):
                            for field in ("dataset", "sampling", "evaluation", "targets"):
                                self.assertEqual(before[field], after[field])
                            self.assertIn(f"prompts-{mode}", after["adapter"]["checkpoint"])
            _, robust = self.preview(root, "--cases", "ip_pikachu", "--prompt-mode", "bare", "--vector-method", "robust")
            self.assertEqual(robust[0][1]["construction"]["groups"], 1)
            for mode, count in (("bare", 2), ("subject", 9), ("matched", 9)):
                with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.preview(root, "--cases", "ip_pikachu", "--prompt-mode", mode,
                                 "--calibration-prompts", str(count))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.preview(root, "--cases", "ip_pikachu", "--prompt-mode", "bare", "--vector-method", "pca-filter")
            self.assertFalse(root.exists())

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
            anchored = ["--use-anchor"]
            custom = ["--use-anchor", "--target-anchor-config", str(run.DIRECTORY / "conf.json")]
            with patch.object(run, "run_function", side_effect=fake_fit), \
                 patch.object(run, "resolve_scheduler", side_effect=lambda cfg: cfg["sampling"].update(scheduler_config={})), \
                 contextlib.redirect_stdout(io.StringIO()):
                variants = [[], anchored, [], anchored, custom]
                for mode in ("bare", "subject", "matched"):
                    variants.extend([["--prompt-mode", mode], ["--prompt-mode", mode]])
                for extra in variants:
                    self.assertEqual(run.main(base + extra), 0)
                templates = read_json(run.DIRECTORY / "prompts/subject.json")
                templates["ip_object"]["contexts"][0] = "a studio photo"
                with patch("DiT.steer.method.read_json", return_value=templates):
                    for _ in range(2):
                        self.assertEqual(run.main(base + ["--prompt-mode", "subject"]), 0)
                self.assertEqual(run.main(base + ["--prompt-mode", "subject"]), 0)
                # Metadata created before prompt modes existed is still reusable.
                metadata = Path(calls[0]["checkpoint"]) / "metadata.json"
                legacy = read_json(metadata)
                legacy["settings"].pop("prompt_mode")
                write_json(metadata, legacy)
                self.assertEqual(run.main(base), 0)
            self.assertEqual(len(calls), 6)
            self.assertNotEqual(calls[0]["checkpoint"], calls[1]["checkpoint"])
            saved = read_json(Path(calls[0]["checkpoint"]) / "metadata.json")
            self.assertEqual(saved["prompt_pairs"][0]["pairs"][0]["negative"], "tench")
            self.assertTrue(all(call["statistics_checkpoint"] is None for call in calls))


if __name__ == "__main__":
    unittest.main()
