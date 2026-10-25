"""DiT protocol, native latent layouts, paired sampling, and result isolation.

Small random tensors/images below are temporary test fixtures, never scores.
"""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from benchmark.cli import get_config, parser
from benchmark.common import configuration, read_json, write_csv, write_json
from benchmark.data import load_manifest, prepare, prompt_records, sampling_protocol
from benchmark.dit_adapter import DiTAdapter
from benchmark.dit_experiments import plans
from benchmark.metrics import feature_fingerprint
from benchmark.migrate_model_results import ModelResultsMigration
from benchmark.model_profiles import DIT_MODELS, MODEL_IDS, profile


def scheduler_config():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return dict(FlowMatchEulerDiscreteScheduler().config)


class Pipe:
    def __init__(self, flux):
        from diffusers import FluxPipeline
        self.transformer = torch.nn.Linear(2, 2, bias=False)
        self.transformer.config = SimpleNamespace(in_channels=64 if flux else 16)
        self.vae_scale_factor = 8
        self.vae = SimpleNamespace(enable_slicing=lambda: None)
        self._pack_latents = FluxPipeline._pack_latents
        self.calls = []
        self.loaded_lora = None
        self.fused = None

    def to(self, device):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    def load_lora_weights(self, path, **kwargs):
        self.loaded_lora = path

    def fuse_lora(self, **kwargs):
        self.fused = kwargs

    def maybe_free_model_hooks(self):
        pass

    def register_modules(self, **modules):
        for name, module in modules.items():
            setattr(self, name, module)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))
                                       for _ in kwargs["prompt"]])


class DiTTests(unittest.TestCase):
    def config(self, model, root):
        cfg = configuration(task="mscoco", base={"model_name": model, "output_dir": str(root / "results"),
                            "original_dir": str(root / "original")})
        cfg["sampling"].update(device="cpu", dtype="bfloat16", scheduler_config=scheduler_config())
        return cfg

    def test_four_model_profiles_and_datasets(self):
        expected = {"flux1_dev": (28, 3.5), "flux1_schnell": (4, 0),
                    "sd30_medium": (28, 7), "sd35_medium": (40, 4.5)}
        self.assertEqual(set(DIT_MODELS), set(expected))
        alias = configuration(task="mscoco", base={"base_model": "stabilityai/stable-diffusion-3-medium"})
        self.assertEqual(alias["model_name"], "sd30_medium")
        self.assertEqual(alias["base_model"], MODEL_IDS["sd30_medium"])
        external = configuration(task="mscoco", base={"model_name": "sd30_medium",
            "adapter": {"kind": "existing"}, "sampling": {"architecture": "sd3"}})
        self.assertEqual(external["sampling"]["architecture"], "sd3")
        with self.assertRaisesRegex(ValueError, "guidance_scale=0"):
            configuration(task="i2p", base={"model_name": "flux1_schnell", "sampling": {"guidance_scale": 7.5}})
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            configuration(task="i2p", base={"model_name": "sd30_medium", "sampling": {"width": 520}})
        for name, (steps, guidance) in expected.items():
            for task in ("i2p", "mscoco"):
                with self.subTest(model=name, task=task):
                    cfg = configuration(task=task, base={"model_name": name})
                    self.assertEqual(cfg["base_model"], MODEL_IDS[name])
                    self.assertEqual(cfg["sampling"]["steps"], steps)
                    self.assertEqual(cfg["sampling"]["guidance_scale"], guidance)
                    self.assertEqual(cfg["sampling"]["dtype"], "bfloat16")
                    self.assertNotIn("attention_processor", cfg["sampling"])
                    self.assertEqual(Path(cfg["output_dir"]).name, name)
                    rows, _ = prompt_records(cfg)
                    self.assertEqual(len(rows), 4703 if task == "i2p" else 1000)
                    if task == "i2p":
                        self.assertEqual(sum("sexual" in r["categories"] for r in rows), 931)
                    else:
                        self.assertEqual([r["seed"] for r in rows], list(range(1000)))

    def test_native_pipeline_kwargs_and_noise_are_batch_independent(self):
        records = [{"prompt": "a test fixture", "seed": seed} for seed in (3, 9)]
        with tempfile.TemporaryDirectory() as temp:
            for model in DIT_MODELS:
                with self.subTest(model=model):
                    cfg = self.config(model, Path(temp))
                    flux = model.startswith("flux")
                    pipe = Pipe(flux)
                    cls = "FluxPipeline" if flux else "StableDiffusion3Pipeline"
                    with patch(f"diffusers.{cls}.from_pretrained", return_value=pipe):
                        adapter = DiTAdapter(cfg)
                    adapter.generate(records, cfg["sampling"])
                    together = pipe.calls[-1]["latents"]
                    adapter.generate(records[1:], cfg["sampling"])
                    torch.testing.assert_close(together[1:], pipe.calls[-1]["latents"], rtol=0, atol=0)
                    expected = torch.stack([torch.randn((16, 64, 64), generator=torch.Generator().manual_seed(r["seed"]))
                                            for r in records]).bfloat16()
                    if flux:
                        expected = pipe._pack_latents(expected, 2, 16, 64, 64)
                    torch.testing.assert_close(together, expected, rtol=0, atol=0)
                    self.assertEqual(together.shape, (2, 1024, 64) if flux else (2, 16, 64, 64))
                    self.assertEqual(pipe.calls[0]["max_sequence_length"], cfg["sampling"]["max_sequence_length"])
                    self.assertEqual("negative_prompt" in pipe.calls[0], not flux)
                    adapter.close()

    def test_transformer_lora_and_full_pipeline_loading(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "weights.safetensors"
            save_file({"weight": torch.ones(2, 2)}, str(weights))
            for kind in ("transformer", "lora", "diffusers"):
                cfg = self.config("sd35_medium", root)
                cfg["adapter"] = {"kind": kind, "checkpoint": str(weights), "model_path": "edited-directory", "scale": .5}
                pipe = Pipe(False)
                with patch("diffusers.StableDiffusion3Pipeline.from_pretrained", return_value=pipe) as factory:
                    adapter = DiTAdapter(cfg)
                if kind == "transformer":
                    torch.testing.assert_close(pipe.transformer.weight, torch.ones(2, 2))
                elif kind == "lora":
                    self.assertEqual(pipe.loaded_lora, str(weights))
                    self.assertEqual(pipe.fused, {"lora_scale": .5})
                else:
                    self.assertEqual(factory.call_args.args[0], "edited-directory")
                adapter.close()

    def test_balanced_placement_keeps_map_and_uses_native_execution_device(self):
        with tempfile.TemporaryDirectory() as temp:
            for model in ("sd30_medium", "sd35_medium"):
                with self.subTest(model=model):
                    cfg = self.config(model, Path(temp))
                    cfg["sampling"]["device"] = "cuda:0"
                    original_protocol = sampling_protocol(cfg)
                    cfg["sampling"].update(device_map="balanced", max_memory={"0": "22GiB", "1": "22GiB"})
                    cfg = configuration(task="mscoco", base=cfg)
                    self.assertEqual(sampling_protocol(cfg), original_protocol)
                    pipe = Pipe(model.startswith("flux"))
                    # CPU fixture stands in for the hook-derived execution
                    # device. Moving the whole pipeline would invalidate the map.
                    pipe._execution_device = torch.device("cpu")
                    pipe.hf_device_map = {"transformer": 1, "text_encoder": 0, "vae": 0}
                    cls = "FluxPipeline" if model.startswith("flux") else "StableDiffusion3Pipeline"
                    with patch("torch.cuda.is_available", return_value=True), \
                         patch(f"diffusers.{cls}.from_pretrained", return_value=pipe) as factory, \
                         patch.object(pipe, "to", side_effect=AssertionError("must preserve device map")):
                        adapter = DiTAdapter(cfg)
                        self.assertEqual(factory.call_args.kwargs["device_map"], "balanced")
                        self.assertEqual(factory.call_args.kwargs["max_memory"], {0: "22GiB", 1: "22GiB"})
                        adapter.generate([{"prompt": "placement fixture", "seed": 2}], cfg["sampling"])
                        self.assertEqual(pipe.calls[-1]["latents"].device.type, "cpu")
                    adapter.close()

    def test_flux_transformer_layers_split_after_resident_components(self):
        from safetensors.torch import save_file
        GiB = 1024 ** 3
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "transformer.safetensors"
            save_file({"weight": torch.ones(2, 2)}, str(checkpoint))
            for model in ("flux1_dev", "flux1_schnell"):
                for kind in ("original", "diffusers", "transformer", "lora"):
                    with self.subTest(model=model, kind=kind):
                        cfg = self.config(model, root)
                        cfg["sampling"].update(device="cuda:0", device_map="auto",
                                               max_memory={"0": "16GiB", "1": "16GiB"})
                        cfg["adapter"].update(kind=kind, checkpoint=str(checkpoint), model_path="edited-flux")
                        events = []
                        class Component(torch.nn.Linear):
                            def __init__(self, name):
                                super().__init__(2, 2)
                                self.name = name
                            def to(self, device):
                                self.requested_device = str(device)
                                events.append(self.name)
                                return self
                            def enable_slicing(self):
                                pass
                        pipe = Pipe(True)
                        pipe.text_encoder, pipe.text_encoder_2, pipe.vae = (
                            Component(name) for name in ("CLIP", "T5", "VAE"))
                        pipe._execution_device = torch.device("cpu")
                        transformer = pipe.transformer
                        transformer.hf_device_map = {"transformer_blocks.0": 0, "single_transformer_blocks.0": 1}
                        def memory(device):
                            events.append(f"memory-{device}")
                            return (14 * GiB if device == 0 else 24 * GiB, 24 * GiB)
                        def load(*args, **kwargs):
                            events.append("transformer-load")
                            self.assertEqual(kwargs["max_memory"], {0: 12 * GiB, 1: 16 * GiB})
                            self.assertEqual(kwargs["device_map"], "auto")
                            self.assertEqual(args[0], "edited-flux" if kind == "diffusers" else MODEL_IDS[model])
                            return transformer
                        with patch("torch.cuda.is_available", return_value=True), \
                             patch("torch.cuda.device_count", return_value=2), \
                             patch("torch.cuda.mem_get_info", side_effect=memory), \
                             patch("diffusers.FluxPipeline.from_pretrained", return_value=pipe) as factory, \
                             patch("diffusers.FluxTransformer2DModel.from_pretrained", side_effect=load), \
                             patch.object(pipe, "to", side_effect=AssertionError("must preserve layer hooks")):
                            adapter = DiTAdapter(cfg)
                            self.assertIsNone(factory.call_args.kwargs["transformer"])
                            self.assertNotIn("device_map", factory.call_args.kwargs)
                            self.assertEqual(events, ["CLIP", "T5", "VAE", "memory-0", "memory-1", "transformer-load"])
                            self.assertEqual(pipe.text_encoder.requested_device, "cuda:0")
                            self.assertEqual(adapter.runtime_info["scope"], "transformer_blocks")
                            if kind == "transformer":
                                torch.testing.assert_close(pipe.transformer.weight, torch.ones(2, 2))
                            elif kind == "lora":
                                self.assertEqual(pipe.loaded_lora, str(checkpoint))
                            adapter.generate([{"prompt": "layer placement fixture", "seed": 2}], cfg["sampling"])
                            adapter.close()

    def test_flux_memory_budget_limits_and_headroom(self):
        from benchmark.flux_placement import transformer_memory_budget
        GiB = 1024 ** 3
        with patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.mem_get_info", return_value=(10 * GiB, 24 * GiB)):
            self.assertEqual(transformer_memory_budget({}), {0: 8 * GiB, 1: 8 * GiB})
            self.assertEqual(transformer_memory_budget({"max_memory": {"1": "6GiB"}}), {1: 6 * GiB})
            with self.assertRaisesRegex(ValueError, "not visible"):
                transformer_memory_budget({"max_memory": {"2": "6GiB"}})
            with self.assertRaisesRegex(ValueError, "insufficient free memory"):
                transformer_memory_budget({"gpu_memory_reserve_gib": 11})

    def test_placement_cli_and_conflicts(self):
        args = parser().parse_args(["dit", "--model", "flux1_dev", "--device-map", "balanced",
                                    "--max-memory", '{"0":"22GiB","1":"22GiB"}'])
        args.task = "i2p"  # The paired runner selects each task before get_config.
        cfg = get_config(args)
        self.assertEqual(cfg["sampling"]["device_map"], "balanced")
        self.assertEqual(cfg["sampling"]["max_memory"]["1"], "22GiB")
        evaluate_args = parser().parse_args(["evaluate", "--task", "i2p", "--device", "cpu"])
        evaluated = get_config(evaluate_args, {"task": "i2p", "config": cfg})
        self.assertEqual(evaluated["sampling"]["device"], "cuda")
        self.assertEqual(evaluated["evaluation"]["device"], "cpu")
        args.cpu_offload = True
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            get_config(args)
        for opts in ({"device_map": "balanced", "device": "cpu"},
                     {"max_memory": {"0": "22GiB"}},
                     {"device_map": "balanced", "max_memory": {"gpu0": "22GiB"}},
                     {"device_map": "balanced", "max_memory": {"cpu": "22GiB"}},
                     {"gpu_memory_reserve_gib": float("nan")}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                configuration(task="i2p", base={"model_name": "flux1_dev", "sampling": opts})

    def test_real_tiny_transformers_flow_sampling_and_vae_decode_on_cpu(self):
        # Exercise real Diffusers denoising/decoding, with tiny random weights
        # and fixed embedding fixtures instead of downloading four large models.
        from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline,
                               FluxTransformer2DModel, StableDiffusion3Pipeline, SD3Transformer2DModel)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            for model in DIT_MODELS:
                with self.subTest(model=model):
                    torch.manual_seed(12)
                    flux = model.startswith("flux")
                    vae = AutoencoderKL(block_out_channels=(8, 8), layers_per_block=1,
                                         down_block_types=("DownEncoderBlock2D",) * 2,
                                         up_block_types=("UpDecoderBlock2D",) * 2,
                                         latent_channels=4, norm_num_groups=4, sample_size=32, shift_factor=0.0)
                    if flux:
                        transformer = FluxTransformer2DModel(in_channels=16, num_layers=1, num_single_layers=1,
                            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
                            pooled_projection_dim=16, axes_dims_rope=(2, 2, 4), guidance_embeds=model == "flux1_dev")
                        pipe = FluxPipeline(FlowMatchEulerDiscreteScheduler(), vae, None, None, None, None, transformer)
                    else:
                        transformer = SD3Transformer2DModel(sample_size=16, in_channels=4, out_channels=4,
                            num_layers=1, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
                            caption_projection_dim=16, pooled_projection_dim=16, pos_embed_max_size=16,
                            dual_attention_layers=(0,) if model == "sd35_medium" else ())
                        pipe = StableDiffusion3Pipeline(transformer, FlowMatchEulerDiscreteScheduler(), vae,
                                                       None, None, None, None, None, None)
                    def encode(**kwargs):
                        batch = len(kwargs["prompt"]) if isinstance(kwargs["prompt"], list) else 1
                        positive, pooled = torch.ones(batch, 4, 16), torch.ones(batch, 16)
                        if flux:
                            return positive, pooled, torch.zeros(4, 3)
                        return positive, torch.zeros_like(positive), pooled, torch.zeros_like(pooled)
                    cfg = self.config(model, Path(temp))
                    cfg["sampling"].update(height=32, width=32, steps=2, dtype="float32")
                    cls = "FluxPipeline" if flux else "StableDiffusion3Pipeline"
                    with patch(f"diffusers.{cls}.from_pretrained", return_value=pipe), \
                         patch.object(pipe, "encode_prompt", side_effect=encode):
                        adapter = DiTAdapter(cfg)
                        images = adapter.generate([{"prompt": "CPU fixture", "seed": 2}], cfg["sampling"])
                        self.assertEqual(images[0].size, (32, 32))
                        self.assertTrue(np.isfinite(np.asarray(images[0])).all())
                        adapter.close()

    def test_paired_plans_are_immutable_and_can_resume_one_task(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "method.json", {"targets": ["nudity", "violence"]})
            args = parser().parse_args(["dit", "--model", "flux1_dev", "--phase", "prepare",
                                       "--config", str(root / "method.json"),
                                       "--output-dir", str(root / "results"), "--original-dir", str(root / "original")])
            with patch("diffusers.FlowMatchEulerDiscreteScheduler.load_config", return_value=scheduler_config()):
                initial = plans(args)
                for task, config_path, manifest in initial:
                    cfg = configuration(config_path)
                    m = load_manifest(prepare(cfg, manifest))
                    self.assertEqual(len(m["records"]), 4703 if task == "i2p" else 1000)
                    self.assertEqual(m["targets"], ["nudity", "violence"])
                    self.assertTrue(all(Path(r["image"]).is_relative_to(root / "results/flux1_dev") for r in m["records"]))
                args.tasks, args.phase = "mscoco", "evaluate"
                self.assertEqual(plans(args)[0][1], initial[1][1])
                args.steps = 9
                with self.assertRaisesRegex(ValueError, "No matching"):
                    plans(args)
            dry = parser().parse_args(["dit", "--model", "flux1_schnell", "--dry-run", "--output-dir", str(root / "dry")])
            plans(dry)
            self.assertFalse((root / "dry").exists())

    def test_original_and_method_sampling_use_the_same_dit_architecture(self):
        from benchmark.adapters import sample_manifest
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.config("flux1_dev", Path(temp))
            cfg["dataset"]["limit_prompts"] = 2
            cfg["adapter"] = {"kind": "plugin", "factory": "benchmark.examples.intervention:create"}
            manifest = load_manifest(prepare(cfg))
            selected = []
            class Adapter:
                runtime_info = {"scope": "temporary-test-fixture"}
                def generate(self, records, opts):
                    return [Image.new("RGB", (512, 512)) for _ in records]
                def close(self):
                    pass
            def create(config):
                selected.append((config["model_name"], config["adapter"]["kind"]))
                return Adapter()
            with patch("benchmark.adapters.create_adapter", side_effect=create):
                sample_manifest(manifest)
            self.assertEqual(selected, [("flux1_dev", "original"), ("flux1_dev", "plugin")])
            for field in ("reference", "image"):
                runtime_path = Path(manifest["records"][0][field]).parent / "sampling-runtime.json"
                self.assertEqual(read_json(runtime_path), Adapter.runtime_info)

    def test_sd14_migration_preserves_images_scores_and_feature_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "benchmark"
            historical = root.parent / "casu-2/runs/original/config.json"
            historical.parent.mkdir(parents=True)
            historical.write_text('{"legacy_threshold": -Infinity}\n')
            images = root / "results/mscoco/method/images"
            images.mkdir(parents=True)
            paths = [images / f"image-{i}.png" for i in range(2)]
            for p in paths:
                Image.new("RGB", (16, 16)).save(p)
            contents = [p.read_bytes() for p in paths]
            config = {"task": "mscoco", "base_model": MODEL_IDS["sd14"], "sampling": {}, "output_dir": str(root / "results")}
            manifest = images.parent / "manifest.json"
            write_json(manifest, {"schema": "concept-benchmark-v1", "task": "mscoco", "base_model": MODEL_IDS["sd14"],
                                 "sampling": {}, "config": config,
                                 "records": [{"image": str(p), "concept": "", "prompt": "a test\nfixture"} for p in paths]})
            write_csv(root / "results/mscoco-metric.csv", [{"task": "mscoco", "method": "fixture", "base_model": MODEL_IDS["sd14"],
                      "cs": 1.25, "manifest": str(manifest), "sampling": "{}"}])
            write_csv(images.parent / "per-image.csv", [{"image": str(paths[0]), "prompt": "a test\nfixture"}])
            old = feature_fingerprint(paths)
            cache = root / "cache/fid/fixture/cache.npz"
            cache.parent.mkdir(parents=True)
            features = np.ones((2, 2048), dtype=np.float32)
            np.savez_compressed(cache, fingerprint=old, features=features)
            write_json(cache.parent / "run.json", {"identity": {"feature_fingerprint": old}})
            migration = ModelResultsMigration(root)
            migration.apply()
            self.assertEqual(historical.read_text(), '{"legacy_threshold": -Infinity}\n')
            moved = [Path(migration.translate(str(p))) for p in paths]
            self.assertEqual([p.read_bytes() for p in moved], contents)
            self.assertTrue((root / "results/sd14/mscoco-metric.csv").is_file())
            self.assertEqual(read_json(Path(migration.translate(str(manifest))))["config"]["output_dir"], str(root / "results/sd14"))
            with np.load(cache) as data:
                self.assertEqual(str(data["fingerprint"].item()), feature_fingerprint(moved))
                np.testing.assert_array_equal(data["features"], features)
            import csv
            with (root / "results/sd14/mscoco/method/per-image.csv").open() as handle:
                self.assertEqual(next(csv.DictReader(handle))["prompt"], "a test\nfixture")
            (root / "results/mscoco").symlink_to(root / "results/sd14/mscoco", target_is_directory=True)
            self.assertEqual(ModelResultsMigration(root).moves, {})


if __name__ == "__main__":
    unittest.main()
