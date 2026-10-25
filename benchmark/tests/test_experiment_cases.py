"""Six concept cases and one joint CIFAR-50 -> COCO case per method."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from benchmark.common import ROOT
from benchmark.experiment_cases import CASE_TARGETS, DEFAULT_CASES, INSTANCE_CASES, STYLE_CASES
from benchmark.erasure_protocol import CIFAR50_TARGETS, evaluation_tasks, object_retain_concepts


class ExperimentCaseTests(unittest.TestCase):
    @unittest.skipUnless((ROOT.parent / "retention_repair/cfrr").is_dir() and
                         (ROOT.parent / "output_response/response_erasure").is_dir(),
                         "Legacy CFRR/OutputResponse source packages are not in this workspace.")
    def test_two_suites_expand_six_concept_cases_and_one_coco_case(self):
        expected = ("instance_snoopy", "instance_snoopy_mickey", "instance_snoopy_mickey_spongebob",
                    "style_van_gogh", "style_monet", "style_picasso", "mscoco_cifar50")
        self.assertEqual(DEFAULT_CASES, expected)
        self.assertEqual(INSTANCE_CASES, expected[:3])
        self.assertEqual(STYLE_CASES, expected[3:6])
        from benchmark.dataset.prompt_templates import CIFAR50_classes
        self.assertEqual(CIFAR50_TARGETS, tuple(CIFAR50_classes))
        self.assertEqual(len(CIFAR50_TARGETS), 50)
        for project, package, runner in (("retention_repair", "cfrr", "run_process"),
                                          ("output_response", "response_erasure", "run")):
            sys.path.insert(0, str(ROOT.parent / project))
            cli = importlib.import_module(f"{package}.cli")
            experiments = importlib.import_module(f"{package}.experiments")
            args = cli.parser().parse_args(["suite", "--dry-run"])
            self.assertEqual(tuple(args.cases.split(",")), expected)
            with patch(f"{package}.experiments.{runner}") as run:
                experiments.suite(args)
            commands = [call.args[0] for call in run.call_args_list]
            stage = "erase" if package == "cfrr" else "edit"
            edits = [cli.parser().parse_args([str(v) for v in cmd[3:]]) for cmd in commands if cmd[3] == stage]
            self.assertEqual(len(edits), 7)
            for case, edit in zip(expected, edits):
                self.assertEqual(edit.task, CASE_TARGETS[case][0])
                self.assertEqual(tuple(edit.targets.split(",")), CASE_TARGETS[case][1])
                if package == "cfrr":
                    self.assertEqual(edit.anchors, " ")
            benchmarks = [cli.parser().parse_args([str(v) for v in cmd[3:]]) for cmd in commands if cmd[3] == "benchmark"]
            self.assertEqual(len(benchmarks), 14)
            self.assertEqual(len({b.method for b in benchmarks}), 14)
            self.assertEqual(sum(b.tasks == "mscoco" for b in benchmarks), 2)  # one per comparison variant
            for case in expected:
                rows = [b for b in benchmarks if b.method.endswith(f"-sd14-{case}")]
                self.assertEqual(len(rows), 2)
                for row in rows:
                    self.assertEqual(tuple(row.tasks.split(",")), evaluation_tasks(CASE_TARGETS[case][0]))
                    self.assertEqual(row.mscoco_count, 1000)

    @unittest.skipUnless((ROOT.parent / "retention_repair/cfrr").is_dir() and
                         (ROOT.parent / "output_response/response_erasure").is_dir(),
                         "Legacy CFRR/OutputResponse source packages are not in this workspace.")
    def test_cifar50_calibration_has_all_targets_and_no_coco1k_leakage(self):
        for project, package, module_name in (("retention_repair", "cfrr", "records"),
                                               ("output_response", "response_erasure", "data")):
            sys.path.insert(0, str(ROOT.parent / project))
            module = importlib.import_module(f"{package}.{module_name}")
            plan_fn = module.calibration_plan if package == "cfrr" else module.response_plan
            with self.subTest(method=package):
                plan = plan_fn(ROOT, "object", CIFAR50_TARGETS)
                for split in ("target_train", "target_validation"):
                    self.assertEqual(len(plan[split]), 50 * 4)
                    self.assertEqual({r["concept"] for r in plan[split]}, set(CIFAR50_TARGETS))
                    self.assertTrue(all("in the style of" not in r["prompt"] for r in plan[split]))
                texts = [r["prompt"].strip().casefold() for rows in plan.values() for r in rows]
                self.assertEqual(len(texts), len(set(texts)))
                coco_test = {r["text"].strip().casefold() for r in module.read_csv(ROOT / "dataset/mscoco.csv")[:1000]}
                self.assertFalse(set(texts) & coco_test)
                for rows in plan.values():
                    for r in rows:
                        if r["group"] in {"object", "mscoco"}:
                            self.assertFalse(module.contains_target(r["prompt"], CIFAR50_TARGETS))
                        if r["group"] == "mscoco":
                            self.assertGreaterEqual(r["source_index"], 1000)
                        if r["group"] == "object":
                            self.assertIn(r["concept"], object_retain_concepts(CIFAR50_TARGETS))
                if package == "response_erasure":
                    self.assertEqual(len(plan["basis"]), 100)


if __name__ == "__main__":
    unittest.main()
