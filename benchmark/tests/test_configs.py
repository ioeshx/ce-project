"""Executable case configs agree with both suites and preserve sampling identity."""
from pathlib import Path
import tempfile
import unittest

from benchmark.cli import get_config, parser
from benchmark.common import ROOT, configuration, digest, write_json
from benchmark.data import load_manifest, prepare, prompt_records, sampling_protocol
from benchmark.erasure_protocol import CIFAR50_TARGETS, evaluation_tasks
from benchmark.experiment_cases import CASE_CONFIGS, CASE_TARGETS, DEFAULT_CASES


class ConfigTests(unittest.TestCase):
    def test_seven_json_configs_prepare_distinct_cases_without_loading_models(self):
        expected = [("Snoopy",), ("Snoopy", "Mickey"), ("Snoopy", "Mickey", "SpongeBob"),
                    ("Van Gogh",), ("Monet",), ("Picasso",), CIFAR50_TARGETS]
        methods = set()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for case, targets in zip(DEFAULT_CASES, expected):
                with self.subTest(case=case):
                    cfg = configuration(CASE_CONFIGS[case])
                    self.assertEqual(tuple(cfg["targets"]), targets)
                    self.assertEqual(CASE_TARGETS[case][1], targets)
                    self.assertEqual(evaluation_tasks(CASE_TARGETS[case][0]), (cfg["task"],))
                    self.assertEqual(cfg["erasure_anchor"], " ")
                    self.assertNotIn(cfg["method"], methods)
                    methods.add(cfg["method"])
                    cfg.update(output_dir=str(root / "results"), cache_dir=str(root / "cache"),
                               original_dir=str(root / "original"))
                    manifest = load_manifest(prepare(cfg))
                    self.assertEqual(tuple(manifest["targets"]), targets)
                    self.assertEqual(len(manifest["records"]), {"instance": 4000, "style": 900, "mscoco": 1000}[cfg["task"]])

    def test_cifar50_source_resolves_and_explicit_cli_targets_survive_plan_reload(self):
        args = parser().parse_args(["prepare", "--config", str(ROOT / "configs/mscoco.json")])
        self.assertEqual(get_config(args)["targets"], list(CIFAR50_TARGETS))
        args.targets = "Snoopy,Mickey"
        cfg = get_config(args)
        self.assertEqual(cfg["targets"], ["Snoopy", "Mickey"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            write_json(path, cfg)
            self.assertEqual(configuration(path)["targets"], ["Snoopy", "Mickey"])

    def test_cifar50_config_reuses_the_exact_coco_reference_protocol(self):
        cfg = configuration(ROOT / "configs/mscoco.json")
        records, _ = prompt_records(cfg)
        self.assertEqual([r["seed"] for r in records], list(range(1000)))
        self.assertEqual(digest({"records": records, "model": cfg["base_model"],
                                 "sampling": sampling_protocol(cfg)}),
                         "a971b6e6c75f9e45043ce7d80f141f862403476e289ae6bf23441849dad86351")


if __name__ == "__main__":
    unittest.main()
