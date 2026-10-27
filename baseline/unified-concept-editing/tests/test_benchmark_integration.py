"""CPU integration fixtures, never reported as empirical baseline scores."""
import copy
import csv
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))
from uce_benchmark import WORKSPACE
from uce_benchmark.adapter import apply_weights
from uce_benchmark.editing import edit, load_upstream
from uce_benchmark.protocol import cases, edit_settings
from uce_benchmark.runner import benchmark_config, parser
from benchmark.common import configuration, file_digest, read_json, write_json
from benchmark.data import load_manifest, prepare, prompt_records, sampling_protocol
from benchmark.erasure_protocol import CIFAR50_TARGETS
from benchmark.plugin_plans import plugin_plan
from benchmark.sampling_profile import PROFILE


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = torch.nn.Module()
        self.attn2.to_k = torch.nn.Linear(4, 4, bias=False)
        self.attn2.to_v = torch.nn.Linear(4, 4, bias=False)
        self.attn2.to_q = torch.nn.Linear(4, 4, bias=False)


class TinyPipe:
    def __init__(self):
        self.unet = TinyUNet()
        self.scheduler = SimpleNamespace(config={})
        self.tokenizer = lambda *a, **kw: {"attention_mask": torch.ones(1, 4, dtype=torch.long)}
        self.tokenizer.model_max_length = 77

    @staticmethod
    def vector(text):
        return torch.tensor([[1.0, len(text) / 20, sum(map(ord, text)) % 11 / 11, 0.3]])

    def encode_prompt(self, prompt, **kwargs):
        return (self.vector(prompt)[:, None, :].repeat(1, 4, 1),)

    def enable_attention_slicing(self):
        pass

    def to(self, device):
        return self


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, *extra):
        return parser().parse_args(["run", *extra])

    def fixture_checkpoint(self, root, case, args):
        checkpoint = root / case / "checkpoint"
        checkpoint.mkdir(parents=True)
        settings = edit_settings(case, args)
        save_file({"attn2.to_k.weight": torch.eye(4), "attn2.to_v.weight": torch.eye(4)}, str(checkpoint / "uce.safetensors"))
        write_json(checkpoint / "metadata.json", {"format": "uce-sd14-partial-kv-v1",
            "base_model": settings["base_model"], "task": settings["task"], "targets": settings["targets"],
            "settings": settings, "weights_sha256": file_digest(checkpoint / "uce.safetensors")})
        return checkpoint

    def test_native_uce_joint_fifty_target_solve_and_partial_weight_loading(self):
        native = load_upstream()
        native.device, native.torch_dtype = "cpu", torch.float32
        pipe = TinyPipe()
        originals = copy.deepcopy(pipe.unet.state_dict())
        targets = list(CIFAR50_TARGETS)
        retains = ["tractor", "train"]
        with tempfile.TemporaryDirectory() as temp, torch.no_grad():
            native.UCE(pipe, targets, [" "] * 50, retains, 1, 1, .5, temp, "uce")
            weights = load_file(str(Path(temp) / "uce.safetensors"))
            self.assertEqual(set(weights), {"attn2.to_k.weight", "attn2.to_v.weight"})
            # Independent evaluation of equation 7, including two retain terms.
            x = torch.cat([pipe.vector(t) for t in targets]).double()
            y = torch.cat([pipe.vector(" ")] * 50).double()
            r = torch.cat([pipe.vector(t) for t in retains]).double()
            denom = .5 * torch.eye(4).double() + x.T @ x + r.T @ r
            for key, actual in weights.items():
                w = originals[key].double()
                numer = .5 * w + (w @ y.T) @ x + (w @ r.T) @ r
                expected = torch.linalg.solve(denom.T, numer.T).T
                # Native UCE explicitly inverts a float32 Gram matrix; compare
                # against the double solve at a tolerance consistent with its
                # conditioning rather than imposing float64 accuracy.
                tolerance = 10 * torch.finfo(torch.float32).eps * torch.linalg.cond(denom)
                relative_error = torch.linalg.norm(actual.double() - expected) / torch.linalg.norm(expected)
                self.assertLess(float(relative_error), float(tolerance))
            apply_weights(pipe.unet, weights)
            torch.testing.assert_close(pipe.unet.attn2.to_q.weight, originals["attn2.to_q.weight"])
            with self.assertRaises(ValueError):
                apply_weights(pipe.unet, {"attn2.to_k.weight": weights["attn2.to_k.weight"]})

    def test_edit_checkpoint_resume_and_offline_provenance(self):
        args = self.args("--device", "cpu", "--sampling-dtype", "float32")
        settings = edit_settings("mscoco_cifar50", args)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint"
            with patch("diffusers.DiffusionPipeline.from_pretrained", return_value=TinyPipe()) as load:
                edit(checkpoint, settings, offline=True)
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            info = read_json(checkpoint / "metadata.json")
            self.assertEqual(info["targets"], list(CIFAR50_TARGETS))
            self.assertEqual(info["settings"]["guide_concepts"], [" "] * 50)
            with patch("diffusers.DiffusionPipeline.from_pretrained", side_effect=AssertionError("must resume")):
                edit(checkpoint, settings, offline=True)
            changed = copy.deepcopy(settings)
            changed["lamb"] = 1.0
            with self.assertRaises(ValueError):
                edit(checkpoint, changed)

    def test_all_nine_prepared_cases_read_actual_benchmark_protocols(self):
        selected = cases("instance,style,i2p_q16,i2p_nudenet,mscoco")
        self.assertEqual(len(selected), 9)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args("--run-root", str(root), "--output-dir", str(root / "results"),
                             "--cache-dir", str(root / "cache"), "--original-dir", str(root / "original"))
            for case in selected:
                with self.subTest(case=case):
                    checkpoint = self.fixture_checkpoint(root, case, args)
                    cfg = benchmark_config(case, args, checkpoint)
                    config, manifest_path = plugin_plan(root / case, "plans", cfg, phase="prepare")
                    manifest = load_manifest(prepare(configuration(config), manifest_path))
                    n = {"instance": 4000, "style": 900, "mscoco": 1000, "i2p": 4703}[cfg["task"]]
                    self.assertEqual(len(manifest["records"]), n)
                    self.assertEqual(manifest["sampling"], PROFILE)
                    self.assertTrue(all(Path(r["image"]).is_relative_to(root / "results") for r in manifest["records"]))
                    self.assertTrue(Path(config).is_relative_to(root / case))
                    if cfg["task"] == "mscoco":
                        self.assertEqual(cfg["targets"], list(CIFAR50_TARGETS))
                        self.assertEqual([r["seed"] for r in manifest["records"]], list(range(1000)))
                        base = configuration(task="mscoco")
                        self.assertEqual(prompt_records(cfg)[0], prompt_records(base)[0])
                        self.assertEqual(sampling_protocol(cfg), sampling_protocol(base))
                    if cfg["task"] == "i2p":
                        self.assertEqual(sum("sexual" in r["categories"] for r in manifest["records"]), 931)
                    args.phase = "evaluate"
                    self.assertEqual(plugin_plan(root / case, "plans", cfg, phase="evaluate")[0], config)

    def test_six_shell_entry_points_expand_full_cases_and_propagate_arguments(self):
        for suffix, total in [("instance", 3), ("style", 3), ("i2p_q16", 1),
                              ("i2p_nudenet", 1), ("mscoco", 1), ("all", 9)]:
            script = METHOD_ROOT / "scripts" / f"benchmark_{suffix}.sh"
            subprocess.run(["bash", "-n", str(script)], check=True)
            result = subprocess.run(["bash", str(script), "--device", "cuda:1", "--offline", "--dry-run"],
                                    check=True, cwd="/tmp", capture_output=True, text=True)
            commands = [shlex.split(line[len("[command] "):]) for line in result.stdout.splitlines()
                        if line.startswith("[command] ")]
            parsed = [parser().parse_args(cmd[3:]) for cmd in commands]
            self.assertEqual(sum(a.command == "edit" for a in parsed), total)
            self.assertEqual(sum(a.command == "benchmark" for a in parsed), total)
            for args in parsed:
                self.assertTrue(args.offline)
                self.assertEqual(args.device, "cuda:1")
                if args.command == "benchmark":
                    self.assertEqual(args.phase, "run")

    def test_prepare_is_immutable_and_native_checkpoint_loads_through_adapter(self):
        from benchmark.adapters import DiffusersAdapter
        from uce_benchmark.adapter import UCEAdapter
        from uce_benchmark.runner import benchmark
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = parser().parse_args(["benchmark", "--case", "mscoco_cifar50", "--phase", "prepare",
                "--run-root", str(root), "--output-dir", str(root / "results"),
                "--original-dir", str(root / "original"), "--cache-dir", str(root / "cache"),
                "--max-records", "2"])
            checkpoint = self.fixture_checkpoint(root, args.case, args)
            benchmark(args)
            manifest = next((root / args.case / "plans").rglob("mscoco-manifest.json"))
            previous = manifest.read_bytes()
            benchmark(args)
            self.assertEqual(manifest.read_bytes(), previous)
            cfg = load_manifest(manifest)["config"]
            def initialize(adapter, config):
                adapter.pipe = TinyPipe()
            with patch.object(DiffusersAdapter, "__init__", initialize):
                adapter = UCEAdapter(cfg)
                torch.testing.assert_close(adapter.pipe.unet.attn2.to_k.weight, torch.eye(4))
                wrong = copy.deepcopy(cfg)
                wrong["targets"] = ["Snoopy"]
                with self.assertRaises(ValueError):
                    UCEAdapter(wrong)
            args.steps = 21
            benchmark(args)
            self.assertEqual(manifest.read_bytes(), previous)
            self.assertEqual(len(list((root / args.case / "plans").rglob("mscoco-manifest.json"))), 2)
            args.steps = 22
            args.phase = "sample"
            with self.assertRaises(ValueError):
                benchmark(args)

    def image_manifest(self, root):
        from benchmark.data import SCHEMA
        cfg = configuration(task="i2p", base={"output_dir": str(root / "results")})
        cfg["sampling"].update(width=16, height=16, dtype="float32", device="cpu")
        cfg["evaluation"]["device"] = "cpu"
        records = []
        for i in range(3):
            path = root / f"image-{i}.png"
            Image.new("RGB", (16, 16), "white").save(path)
            records.append({"sample_id": f"sample-{i}", "image": str(path), "prompt": f"fixture {i}",
                            "seed": i, "concept": "", "categories": ["sexual"] if i < 2 else ["violence"]})
        manifest = {"schema": SCHEMA, "task": "i2p", "method": "test-fixture", "targets": ["nudity"],
                    "base_model": cfg["base_model"], "sampling": cfg["sampling"], "config": cfg,
                    "records": records, "adapter_identity": {"kind": "existing"}}
        path = root / "manifest.json"
        write_json(path, manifest)
        return path

    def test_independent_detectors_publish_na_and_failure_keeps_previous_csv(self):
        from uce_benchmark.i2p import evaluate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.image_manifest(root)
            # Fixtures only exercise routing/accounting/publication, not accuracy.
            class Detector:
                model = None
                provenance = {"test_fixture": True}
                def __init__(self, opts):
                    pass
                def predict(self, records, batch_size):
                    return [{"sample_id": r["sample_id"], "unsafe": i == 0} for i, r in enumerate(records)]
            with patch("uce_benchmark.i2p.Q16Detector", Detector), \
                 patch("uce_benchmark.i2p.nudenet_predict", side_effect=AssertionError("unrequested detector")):
                row = evaluate(manifest, "q16")
            self.assertEqual(row["q16_n"], 3)
            self.assertEqual(row["q16_unsafe_rate"], 1 / 3)
            self.assertIsNone(row["nudenet_unsafe_rate"])
            csv_path = root / "results/sd14/i2p-metric.csv"
            old = csv_path.read_bytes()
            with patch("uce_benchmark.i2p.Q16Detector", side_effect=FileNotFoundError("missing weights")):
                with self.assertRaises(FileNotFoundError):
                    evaluate(manifest, "q16")
            self.assertEqual(csv_path.read_bytes(), old)
            def nude(records, opts):
                return ([{"sample_id": r["sample_id"], "unsafe": i == 0, "detections": [],
                          "accepted_detections": []} for i, r in enumerate(records[:2])],
                        {c: int(i == 0) for i, c in enumerate(opts["nudenet_classes"])}, {"test_fixture": True})
            with patch("uce_benchmark.i2p.Q16Detector", side_effect=AssertionError("unrequested detector")), \
                 patch("uce_benchmark.i2p.nudenet_predict", nude):
                row = evaluate(manifest, "nudenet")
            self.assertEqual(row["nudenet_n"], 2)
            self.assertEqual(row["nudenet_unsafe_rate"], .5)
            self.assertEqual(row["nudenet_instances"], 1)
            self.assertIsNone(row["q16_unsafe_rate"])
            with csv_path.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertEqual(len(list((root / "results/sd14/plots/i2p").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
