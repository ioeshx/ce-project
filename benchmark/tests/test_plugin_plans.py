"""Both methods must prepare all concept/COCO plans without polluting results."""
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

from benchmark.common import ROOT, configuration, read_json
from benchmark.data import prepare, load_manifest, sampling_protocol
from benchmark.plugin_plans import plugin_plan


class PluginPlanTests(unittest.TestCase):
    def test_checkpoint_side_plans_do_not_redirect_benchmark_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "runs/case/checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "weights.safetensors").write_bytes(b"plan identity fixture only")
            cfg = configuration(task="mscoco", base={"output_dir": str(root / "results")})
            cfg["adapter"] = {"kind": "plugin", "checkpoint": str(checkpoint)}
            path, _ = plugin_plan(root / "runs/case", "plans", cfg, phase="prepare")
            self.assertTrue(path.is_relative_to(root / "runs/case/plans"))
            self.assertEqual(read_json(path)["output_dir"], str(root / "results/sd14"))

    @unittest.skipUnless((ROOT.parent / "retention_repair/sample_mscoco.py").is_file() and
                         (ROOT.parent / "output_response/sample_mscoco.py").is_file(),
                         "Legacy method entry points are not in this workspace.")
    def test_coco_dispatchers_use_native_method_and_preserve_legacy_formats(self):
        import importlib.util
        from types import SimpleNamespace
        from unittest.mock import patch
        from benchmark.common import write_json
        for project, package, native in (("retention_repair", "cfrr", "casu2-sd1-v1"),
                                          ("output_response", "response_erasure", "output-response-sd14-v1")):
            sys.path.insert(0, str(ROOT.parent / project))
            spec = importlib.util.spec_from_file_location(f"{package}_coco_dispatch", ROOT.parent / project / "sample_mscoco.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp)
                args = ["--checkpoint", str(checkpoint), "--method", "example"]
                write_json(checkpoint / "metadata.json", {"format": native})
                with patch(f"{package}.cli.main", return_value=0) as run:
                    self.assertEqual(module.main(args), 0)
                    self.assertEqual(run.call_args.args[0], ["benchmark", "--tasks", "mscoco", *args])
                write_json(checkpoint / "metadata.json", {"format": "casu2-sdxl-v1"})
                with patch.dict(sys.modules, {"sample_mscoco_legacy": SimpleNamespace(main=lambda argv: ("legacy", argv))}):
                    self.assertEqual(module.main(args), ("legacy", args))

    def test_plan_versions_are_immutable_and_weight_sensitive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            weights = checkpoint / "weights.safetensors"
            weights.write_bytes(b"fixture for plan checksums only")
            cfg = configuration(task="mscoco")
            cfg["adapter"] = {"kind": "plugin", "checkpoint": str(checkpoint)}
            cfg["dataset"]["limit_prompts"] = 2
            results = root / "results"
            a, manifest = plugin_plan(results, "plans", cfg, phase="prepare")
            old = a.read_bytes()
            # No sample is possible until prepare actually creates its manifest.
            with self.assertRaises(ValueError):
                plugin_plan(results, "plans", cfg, phase="sample")
            manifest.write_text("{}")
            self.assertEqual(a, plugin_plan(results, "plans", cfg, phase="evaluate")[0])
            changed = copy.deepcopy(cfg)
            changed["sampling"]["steps"] += 1
            b, _ = plugin_plan(results, "plans", changed, phase="prepare")
            self.assertNotEqual(a, b)
            self.assertEqual(a.read_bytes(), old)
            weights.write_bytes(b"changed weights at same checkpoint path")
            c, _ = plugin_plan(results, "plans", cfg, phase="prepare")
            self.assertNotEqual(a, c)
            dry_root = root / "dry"
            plugin_plan(dry_root, "plans", cfg, phase="prepare", dry_run=True)
            self.assertFalse(dry_root.exists())
            with self.assertRaises(ValueError):
                plugin_plan(dry_root, "plans", cfg, phase="sample")
            self.assertFalse(dry_root.exists())

    @unittest.skipUnless((ROOT.parent / "retention_repair/cfrr").is_dir() and
                         (ROOT.parent / "output_response/response_erasure").is_dir(),
                         "Legacy CFRR/OutputResponse source packages are not in this workspace.")
    def test_both_methods_instance_style_and_coco_use_shared_readable_protocol(self):
        import torch
        for project, package in (("retention_repair", "cfrr"), ("output_response", "response_erasure")):
            sys.path.insert(0, str(ROOT.parent / project))
            cli = importlib.import_module(f"{package}.cli")
            experiments = importlib.import_module(f"{package}.experiments")
            io = importlib.import_module(f"{package}.io" if package == "cfrr" else f"{package}.common")
            from benchmark.experiment_cases import CASE_CONFIGS, CASE_TARGETS, DEFAULT_CASES
            from benchmark.erasure_protocol import evaluation_tasks, erasure_protocol
            for case in DEFAULT_CASES:
                task, targets = CASE_TARGETS[case]
                count = {"instance": 4000, "style": 900, "object": 1000}[task]
                with self.subTest(method=package, case=case), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    bench = root / "benchmark"
                    bench.mkdir()
                    (bench / "__main__.py").touch()
                    (bench / "dataset").symlink_to(ROOT / "dataset", target_is_directory=True)
                    checkpoint = root / "checkpoint"
                    io.save_checkpoint(checkpoint, {"block.attn2.to_v.weight": torch.ones(2, 2)},
                                       {"base_model": io.DEFAULT_MODEL, "task": task, "targets": list(targets)})
                    args = cli.parser().parse_args(["benchmark", "--checkpoint", str(checkpoint), "--method", f"{package}-{case}",
                                                    "--benchmark-root", str(bench), "--phase", "prepare"])
                    _, plans = experiments.benchmark_configs(args)
                    paths = {}
                    for entry in plans:
                        config_path, manifest_path = entry[-2:]
                        cfg = configuration(config_path)
                        self.assertEqual(cfg["configuration_source"], str(CASE_CONFIGS[case]))
                        self.assertEqual(cfg["description"], read_json(CASE_CONFIGS[case])["description"])
                        self.assertEqual(cfg["adapter"]["kind"], "plugin")
                        manifest = load_manifest(prepare(cfg, manifest_path))
                        current = manifest["task"]
                        paths[current] = config_path
                        self.assertEqual(len(manifest["records"]), 1000 if current == "mscoco" else count)
                        self.assertEqual(manifest["targets"], list(targets))
                        self.assertEqual(cfg[package if package == "cfrr" else "output_response"]["erasure_protocol"],
                                         erasure_protocol(task, targets))
                        self.assertEqual(manifest["sampling"]["steps"], 20)
                        self.assertEqual(manifest["sampling"]["dtype"], "float16")
                        self.assertEqual(manifest["sampling"]["scheduler"], "DPMSolverMultistepScheduler")
                        self.assertTrue(all(Path(r["reference"]).is_relative_to(bench / "original") for r in manifest["records"]))
                        self.assertTrue(all(Path(r["image"]).is_relative_to(bench / "results") for r in manifest["records"]))
                        self.assertNotRegex(str(config_path.relative_to(bench)), r"[a-f0-9]{16,64}")
                    self.assertEqual(set(paths), set(evaluation_tasks(task)))
                    # An explicit single task must find the same immutable plan.
                    args.tasks = evaluation_tasks(task)[0]
                    args.phase = "evaluate"
                    _, later = experiments.benchmark_configs(args)
                    self.assertEqual(later[0][-2], paths[args.tasks])


if __name__ == "__main__":
    unittest.main()
