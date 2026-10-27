"""CPU contracts and real tiny Diffusers training; no empirical paper scores."""
import copy
import csv
import json
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
from esd_benchmark.adapter import ESDAdapter
from esd_benchmark.protocol import cases, training_settings
from esd_benchmark.runner import benchmark, benchmark_config, parser
from esd_benchmark.training import TrainingPipeline, train
from benchmark.adapters import DiffusersAdapter
from benchmark.common import ROOT, configuration, file_digest, read_json, write_json
from benchmark.data import load_manifest, prepare, prompt_records, sampling_protocol
from benchmark.erasure_protocol import CIFAR50_TARGETS
from benchmark.plugin_plans import plugin_plan
from benchmark.sampling_profile import PROFILE
from utils.esd_trainer import StableDiffusionESDAdapter


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = torch.nn.Module()
        self.attn2.to_k = torch.nn.Linear(4, 4, bias=False)
        self.attn2.to_v = torch.nn.Linear(4, 4, bias=False)
        self.conv = torch.nn.Conv2d(1, 1, 1)


def tiny_pipeline(root):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab[c] = len(vocab)
        vocab[c + "</w>"] = len(vocab)
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tok = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"), model_max_length=77)
    encoder = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocab), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=77,
        bos_token_id=0, eos_token_id=1, pad_token_id=1))
    unet = UNet2DConditionModel(sample_size=4, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32,), down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), cross_attention_dim=8, attention_head_dim=4, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(16, 32),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), norm_num_groups=8)
    pipe = TrainingPipeline(vae=vae, text_encoder=encoder, tokenizer=tok, unet=unet,
        scheduler=DDIMScheduler(num_train_timesteps=32), safety_checker=None,
        feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, *extra):
        return parser().parse_args(["run", *extra])

    def fixture_checkpoint(self, root, case, args):
        checkpoint = root / case / "checkpoint"
        checkpoint.mkdir(parents=True)
        settings = training_settings(case, args)
        unet = TinyUNet()
        names = StableDiffusionESDAdapter().select_parameter_names(unet, settings["train_method"])
        values = {name: torch.ones_like(dict(unet.named_parameters())[name]) for name in names}
        save_file(values, str(checkpoint / "esd.safetensors"))
        write_json(checkpoint / "metadata.json", {"format": "esd-sd14-benchmark-v1",
            "base_model": settings["base_model"], "task": settings["task"], "targets": settings["targets"],
            "settings": settings, "trainable_parameters": names,
            "weights_sha256": file_digest(checkpoint / "esd.safetensors")})
        return checkpoint

    def test_all_nine_cases_and_training_budgets_read_benchmark_jsons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.args("--run-root", str(root), "--output-dir", str(root / "results"),
                "--original-dir", str(root / "original"), "--cache-dir", str(root / "cache"))
            names = cases("instance,style,i2p_q16,i2p_nudenet,mscoco")
            self.assertEqual(len(names), 9)
            for case in names:
                with self.subTest(case=case):
                    checkpoint = self.fixture_checkpoint(root, case, args)
                    cfg = benchmark_config(case, args, checkpoint)
                    config, manifest = plugin_plan(root / case, "plans", cfg, phase="prepare")
                    loaded = load_manifest(prepare(configuration(config), manifest))
                    self.assertEqual(len(loaded["records"]), {"instance": 4000, "style": 900, "i2p": 4703, "mscoco": 1000}[cfg["task"]])
                    self.assertEqual(loaded["sampling"], PROFILE)
                    settings = cfg["esd"]["training_settings"]
                    self.assertEqual(settings["iterations"], 200 * len(cfg["targets"]))
                    self.assertEqual(settings["train_method"], "esd-u" if cfg["task"] == "i2p" else "esd-x")
                    self.assertEqual(settings["training_protocol"]["dtype"], "float32")
                    self.assertTrue(settings["training_exception"])
                    if cfg["task"] == "mscoco":
                        self.assertEqual(cfg["targets"], list(CIFAR50_TARGETS))
                        self.assertEqual([r["seed"] for r in loaded["records"]], list(range(1000)))
                        base = configuration(task="mscoco")
                        self.assertEqual(prompt_records(cfg)[0], prompt_records(base)[0])
                        self.assertEqual(sampling_protocol(cfg), sampling_protocol(base))
                    if cfg["task"] == "i2p":
                        self.assertEqual(sum("sexual" in r["categories"] for r in loaded["records"]), 931)

    def test_real_native_optimizer_trains_joint_fifty_and_unconditional_variants(self):
        for case, iterations in [("mscoco_cifar50", 50), ("i2p_nudenet", 2)]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                self.check_real_training(Path(temp), case, iterations)

    def check_real_training(self, root, case, iterations):
        args = self.args("--device", "cpu", "--sampling-dtype", "float32", "--iterations", str(iterations),
                         "--training-steps", "2", "--training-resolution", "8")
        settings = training_settings(case, args)
        pipe = tiny_pipeline(root)
        original = copy.deepcopy(pipe.unet.state_dict())
        with patch.object(TrainingPipeline, "from_pretrained", return_value=pipe) as load:
            checkpoint = train(root / "checkpoint", settings, offline=True, cpu_threads=2)
        self.assertTrue(load.call_args.kwargs["local_files_only"])
        info = read_json(checkpoint / "metadata.json")
        self.assertEqual(info["target_counts"], {t: iterations // len(settings["targets"]) for t in settings["targets"]})
        self.assertEqual(info["targets"], settings["targets"])
        weights = load_file(str(checkpoint / "esd.safetensors"))
        self.assertTrue(all(torch.isfinite(w).all() for w in weights.values()))
        self.assertTrue(any(not torch.equal(value, original[key]) for key, value in weights.items()))
        with patch.object(TrainingPipeline, "from_pretrained", side_effect=AssertionError("must resume")):
            self.assertEqual(train(checkpoint, settings), checkpoint)
        changed = copy.deepcopy(settings)
        changed["negative_guidance"] = 3
        with self.assertRaises(ValueError):
            train(checkpoint, changed)

    def test_rollout_noise_is_float32_cpu_then_cast_and_recordwise(self):
        pipe = SimpleNamespace(vae_scale_factor=8, scheduler=SimpleNamespace(init_noise_sigma=1.0))
        generators = [torch.Generator().manual_seed(s) for s in [0, 19]]
        batch = TrainingPipeline.prepare_latents(pipe, 2, 4, 64, 64, torch.float16, "cpu", generators)
        expected = torch.stack([torch.randn((4, 8, 8), generator=torch.Generator().manual_seed(s)).half() for s in [0, 19]])
        torch.testing.assert_close(batch, expected, rtol=0, atol=0)
        for i, seed in enumerate([0, 19]):
            single = TrainingPipeline.prepare_latents(pipe, 1, 4, 64, 64, torch.float16, "cpu",
                torch.Generator().manual_seed(seed))
            torch.testing.assert_close(batch[i], single[0], rtol=0, atol=0)

    def test_adapter_and_immutable_plans(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = parser().parse_args(["benchmark", "--case", "mscoco_cifar50", "--phase", "prepare",
                "--run-root", str(root), "--output-dir", str(root / "results"),
                "--original-dir", str(root / "original"), "--cache-dir", str(root / "cache"), "--max-records", "2"])
            self.fixture_checkpoint(root, args.case, args)
            benchmark(args)
            manifest = next((root / args.case / "plans").rglob("mscoco-manifest.json"))
            old = manifest.read_bytes()
            benchmark(args)
            self.assertEqual(old, manifest.read_bytes())
            cfg = load_manifest(manifest)["config"]
            def initialize(adapter, config):
                adapter.pipe = SimpleNamespace(unet=TinyUNet())
            with patch.object(DiffusersAdapter, "__init__", initialize):
                model = ESDAdapter(cfg)
                torch.testing.assert_close(model.pipe.unet.attn2.to_k.weight, torch.ones(4, 4))
                cfg["targets"] = ["Snoopy"]
                with self.assertRaises(ValueError):
                    ESDAdapter(cfg)
            args.steps = 21
            benchmark(args)
            self.assertEqual(old, manifest.read_bytes())
            args.steps = 22
            args.phase = "sample"
            with self.assertRaises(ValueError):
                benchmark(args)

    def test_six_shells_expand_all_targets_and_phase_arguments(self):
        for suffix, total in [("instance", 3), ("style", 3), ("i2p_q16", 1),
                              ("i2p_nudenet", 1), ("mscoco", 1), ("all", 9)]:
            script = METHOD_ROOT / "scripts" / f"benchmark_{suffix}.sh"
            subprocess.run(["bash", "-n", str(script)], check=True)
            result = subprocess.run(["bash", str(script), "--device", "cuda:1", "--offline", "--dry-run"],
                check=True, cwd="/tmp", capture_output=True, text=True)
            parsed = [parser().parse_args(shlex.split(line[10:])[3:]) for line in result.stdout.splitlines()
                      if line.startswith("[command] ")]
            self.assertEqual(sum(a.command == "edit" for a in parsed), total)
            self.assertEqual(sum(a.command == "benchmark" for a in parsed), total)
            self.assertTrue(all(a.device == "cuda:1" and a.offline for a in parsed))
            for args in parsed:
                if args.command == "benchmark":
                    self.assertEqual(args.phase, "run")
        args = self.args("--iterations", "49")
        with self.assertRaises(ValueError):
            training_settings("mscoco_cifar50", args)

    def image_manifest(self, root):
        from benchmark.common import SCHEMA
        cfg = configuration(task="i2p", base={"output_dir": str(root / "results")})
        cfg["sampling"].update(width=16, height=16, dtype="float32", device="cpu")
        cfg["evaluation"]["device"] = "cpu"
        records = []
        for i in range(3):
            path = root / f"image-{i}.png"
            Image.new("RGB", (16, 16), "white").save(path)
            records.append({"sample_id": f"sample-{i}", "image": str(path), "prompt": f"fixture {i}",
                            "seed": i, "concept": "", "categories": ["sexual"] if i < 2 else ["violence"]})
        path = root / "manifest.json"
        write_json(path, {"schema": SCHEMA, "task": "i2p", "method": "test-fixture", "targets": ["nudity"],
            "base_model": cfg["base_model"], "sampling": cfg["sampling"], "config": cfg,
            "records": records, "adapter_identity": {"kind": "existing"}})
        return path

    def test_independent_i2p_routing_counts_na_and_failure_preserves_old_row(self):
        from esd_benchmark.i2p import evaluate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.image_manifest(root)
            class Detector:
                model = None
                provenance = {"test_fixture": True}
                def __init__(self, opts):
                    pass
                def predict(self, records, batch_size):
                    return [{"sample_id": r["sample_id"], "unsafe": i == 0} for i, r in enumerate(records)]
            with patch("esd_benchmark.i2p.Q16Detector", Detector), \
                 patch("esd_benchmark.i2p.nudenet_predict", side_effect=AssertionError("unrequested detector")):
                row = evaluate(manifest, "q16")
            self.assertEqual(row["q16_n"], 3)
            self.assertEqual(row["q16_unsafe_rate"], 1 / 3)
            self.assertIsNone(row["nudenet_unsafe_rate"])
            result = root / "results/sd14/i2p-metric.csv"
            old = result.read_bytes()
            with patch("esd_benchmark.i2p.Q16Detector", side_effect=FileNotFoundError("missing weights")):
                with self.assertRaises(FileNotFoundError):
                    evaluate(manifest, "q16")
            self.assertEqual(old, result.read_bytes())
            def nude(records, opts):
                return ([{"sample_id": r["sample_id"], "unsafe": i == 0, "detections": [],
                          "accepted_detections": []} for i, r in enumerate(records[:2])],
                        {c: int(i == 0) for i, c in enumerate(opts["nudenet_classes"])}, {"test_fixture": True})
            with patch("esd_benchmark.i2p.Q16Detector", side_effect=AssertionError("unrequested detector")), \
                 patch("esd_benchmark.i2p.nudenet_predict", nude):
                row = evaluate(manifest, "nudenet")
            self.assertEqual(row["nudenet_n"], 2)
            self.assertEqual(row["nudenet_unsafe_rate"], .5)
            self.assertIsNone(row["q16_unsafe_rate"])
            with result.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            self.assertEqual(len(list((root / "results/sd14/plots/i2p").glob("*.pdf"))), 1)


if __name__ == "__main__":
    unittest.main()
