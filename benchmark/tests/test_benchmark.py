"""Meaningful protocol, numerical, persistence, and adapter-contract regressions.

Mocks below test plumbing only. No mock scores are saved as benchmark results.
Run: python -m unittest discover -s benchmark/tests -v
"""
from __future__ import annotations

import copy
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.common import configuration, read_json, write_json
from benchmark.data import import_casu, load_manifest, prepare, prompt_records, validate_manifest
from benchmark.metrics import fid_from_features
from benchmark.results import groups, read_rows, report, upsert


def concurrent_write(root, index):
    upsert(root, "mscoco", {"task": "mscoco", "method": f"m{index}", "cs": index, "protocol_id": "p"})


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, task="instance"):
        cfg = configuration(task=task)
        cfg.update(output_dir=str(self.root / "results"), cache_dir=str(self.root / "cache"), original_dir=str(self.root / "original"))
        cfg["evaluation"].update(device="cpu", batch_size=2, local_files_only=True)
        return cfg

    def small_manifest(self):
        cfg = self.config()
        cfg["dataset"].update(concepts=["Snoopy", "Mickey"], limit_prompts=1, seeds=[0, 1, 2])
        m = load_manifest(prepare(cfg))
        for r in m["records"]:
            for field in ("image", "reference"):
                p = Path(r[field])
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (512, 512), color="white").save(p)
        return m

    def test_default_data_and_seeds(self):
        expected = {"instance": 4000, "style": 900, "i2p": 4703, "mscoco": 1000}
        for task, count in expected.items():
            rows, _ = prompt_records(self.config(task))
            self.assertEqual(len(rows), count)
            self.assertEqual(len({r["sample_id"] for r in rows}), count)
            self.assertEqual(rows, prompt_records(self.config(task))[0])
            if task == "i2p":
                self.assertEqual(sum("sexual" in r["categories"] for r in rows), 931)
                self.assertEqual(rows[0]["seed"], int(rows[0]["source_sampling"]["sd_seed"]))
            elif task in {"instance", "style"}:
                self.assertEqual([r["seed"] for r in rows[:10]], list(range(10)))
            else:
                self.assertEqual(rows[-1]["seed"], 999)

    def test_custom_concept_templates(self):
        cfg = self.config()
        cfg["dataset"]["concepts"] = ["Custom character"]
        rows, _ = prompt_records(cfg)
        self.assertEqual(len(rows), 800)
        self.assertTrue(all("Custom character" in r["prompt"] for r in rows))

    def test_shared_originals_and_protocol_changes(self):
        cfg = self.config()
        a = load_manifest(prepare(cfg))
        cfg["method"] = "different"
        cfg["targets"] = ["Mickey"]
        b = load_manifest(prepare(cfg))
        self.assertEqual(a["records"][0]["reference"], b["records"][0]["reference"])
        cfg["sampling"]["steps"] += 1
        c = load_manifest(prepare(cfg))
        self.assertNotEqual(a["records"][0]["reference"], c["records"][0]["reference"])

    def test_manifest_alignment_missing_corrupt_and_dimensions(self):
        m = self.small_manifest()
        validate_manifest(m)
        broken = copy.deepcopy(m)
        broken["records"][1]["sample_id"] = broken["records"][0]["sample_id"]
        with self.assertRaises(ValueError):
            validate_manifest(broken)
        p = Path(m["records"][0]["image"])
        Image.new("RGB", (64, 64)).save(p)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_manifest(m)
        p.write_bytes(b"bad image")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            validate_manifest(m)

    def test_fid_self_zero_and_lowrank_equivalence(self):
        rng = np.random.default_rng(42)
        a, b = rng.normal(size=(35, 7)), rng.normal(size=(29, 7)) + 0.2
        self.assertAlmostEqual(fid_from_features(a, a, "lowrank")[0], 0, places=9)
        self.assertAlmostEqual(fid_from_features(a, b, "lowrank")[0],
                               fid_from_features(a, b, "covariance")[0], places=9)
        with self.assertRaises(ValueError):
            fid_from_features(a[:1], b)
        with self.assertRaises(ValueError):
            fid_from_features(a * np.nan, b)

    def test_uneven_clip_batch_and_all_targets_na(self):
        from benchmark.evaluate import clip_fid
        m = self.small_manifest()
        prompts = ["first prompt", "second prompt", "third prompt"]
        for row, prompt in zip(m["records"], prompts):
            row["prompt"] = prompt
        class Scorer:
            revision = "test-only"
            def __init__(self, *args):
                pass
            def score(self, paths, texts):
                # Prompt identity follows the image record even in the uneven final batch.
                for path, text in zip(paths, texts):
                    self_outer.assertEqual(Path(path).name, m["records"][prompts.index(text)]["sample_id"] + ".png")
                return np.array([1.0 if t != "third prompt" else 10.0 for t in texts])
        class Extractor:
            def __init__(self, *args):
                pass
            def extract(self, paths, *args):
                return np.arange(len(paths) * 2048, dtype=np.float32).reshape(len(paths), 2048) / 2048
        self_outer = self
        out = self.root / "details"
        out.mkdir()
        with patch("benchmark.metrics.CLIPScorer", Scorer), patch("benchmark.metrics.FIDExtractor", Extractor):
            result, _ = clip_fid(m, m["config"]["evaluation"], out, m["config"]["cache_dir"])
        self.assertEqual(result["cs"], 4.0)  # per-image mean, not mean([1,10])=5.5
        self.assertIsNone(result["fid_non_target_macro"])
        self.assertIsNone(result["cs_non_target"])
        # Existing CASU caches may fingerprint the original symlink instead of its target.
        from benchmark.metrics import feature_fingerprint
        aliases = self.root / "alias"
        aliases.symlink_to(Path(m["records"][0]["reference"]).parent, target_is_directory=True)
        for row in m["records"]:
            for field in ("reference", "image"):
                row[f"{field}_alias"] = str(aliases / Path(row[field]).name)
        features = np.arange(3 * 2048, dtype=np.float32).reshape(3, 2048)
        cache = self.root / "legacy-features.npz"
        np.savez_compressed(cache, features=features, fingerprint=feature_fingerprint([Path(r["reference_alias"]) for r in m["records"]]))
        opts = dict(m["config"]["evaluation"], fid_cache_files={"snoopy": {"reference": str(cache), "image": str(cache)}})
        with patch("benchmark.metrics.CLIPScorer", Scorer), \
             patch("benchmark.metrics.FIDExtractor", side_effect=AssertionError("Must reuse verified legacy features")):
            clip_fid(m, opts, out, self.root / "empty-cache")

    def test_upsert_concurrency_and_failure_preserves_csv(self):
        root = str(self.root)
        ctx = multiprocessing.get_context("fork")
        processes = [ctx.Process(target=concurrent_write, args=(root, i)) for i in range(8)]
        for p in processes:
            p.start()
        for p in processes:
            p.join(30)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(len(read_rows(root, "mscoco")), 8)
        upsert(root, "mscoco", {"task": "mscoco", "method": "m0", "cs": 99, "protocol_id": "new"})
        rows = read_rows(root, "mscoco")
        self.assertEqual(len(rows), 8)
        self.assertEqual(next(r["cs"] for r in rows if r["method"] == "m0"), "99")
        before = (self.root / "mscoco-metric.csv").read_bytes()
        with self.assertRaises(ValueError):
            upsert(root, "mscoco", {"task": "mscoco", "method": "m0", "cs": float("nan")})
        self.assertEqual(before, (self.root / "mscoco-metric.csv").read_bytes())
        self.assertEqual(len(groups(rows)), 2)

    def test_failed_evaluation_does_not_publish(self):
        from benchmark.evaluate import evaluate
        m = self.small_manifest()
        path = self.root / "manifest.json"
        write_json(path, m)
        out = Path(m["config"]["output_dir"])
        upsert(out, "instance", {"task": "instance", "method": "original", "cs": 88})
        before = (out / "instance-metric.csv").read_bytes()
        with patch("benchmark.evaluate.clip_fid", side_effect=RuntimeError("detector/model failed")):
            with self.assertRaises(RuntimeError):
                evaluate(path)
        self.assertEqual(before, (out / "instance-metric.csv").read_bytes())

    def test_csv_only_plots_and_reports_group_protocols(self):
        from benchmark.plotting import plot
        import json
        for i in range(2):
            upsert(self.root, "mscoco", {"task": "mscoco", "method": f"m{i}", "cs": 20+i, "fid": i,
                   "protocol_id": str(i) * 64, "targets": "[]", "n": 10, "base_model": "test",
                   "sampling": json.dumps({"scheduler": "ddim", "steps": 50, "guidance_scale": 7.5, "width": 512, "height": 512})})
        self.assertEqual(len(plot(self.root, "mscoco")), 4)
        text = report(self.root, "mscoco")
        self.assertIn("group 1", text)
        self.assertIn("group 2", text)
        self.assertTrue((self.root / "mscoco-metric.txt").is_file())

    def test_sampling_resumes_and_replaces_corruption(self):
        from benchmark.adapters import sample_manifest
        m = self.small_manifest()
        # Make edited paths distinct from the originals.
        for i, row in enumerate(m["records"]):
            row["image"] = str(self.root / f"edited-{i}.png")
        Path(m["records"][0]["image"]).write_bytes(b"damaged")
        class Adapter:
            def generate(self, rows, opts):
                return [Image.new("RGB", (opts["width"], opts["height"])) for _ in rows]
            def close(self):
                pass
        with patch("benchmark.adapters.create_adapter", return_value=Adapter()) as create:
            sample_manifest(m, "image")
            self.assertEqual(create.call_count, 1)
            validate_manifest(m)
            sample_manifest(m, "image")
            self.assertEqual(create.call_count, 1)

    def test_casu_import_preserves_twenty_steps_without_checkpoint(self):
        import hashlib
        import json
        source = self.root / "casu"
        source.mkdir()
        rows = [{"concept": "Snoopy", "prompt": "a Snoopy", "seed": 0,
                 "original": "original.png", "edit": "edited.png"}]
        original = [{k: v for k, v in rows[0].items() if k not in {"original", "edit", "compare"}}]
        write_json(source / "manifest.json", rows)
        write_json(source / "dataset.json", {"record_count": 1, "records_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                   "task": "instance", "targets": ["Snoopy"], "sampling": {"sd_ckpt": "test", "architecture": "sd1", "steps": 20,
                   "guidance_scale": 7.5, "negative_prompt": "", "height": 512, "width": 512, "dtype": "auto"}})
        m = load_manifest(import_casu(source, self.config()))
        self.assertEqual(m["sampling"]["steps"], 20)
        self.assertEqual(m["sampling"]["scheduler"], "dpm_solver_multistep")
        self.assertTrue(Path(m["records"][0]["reference"]).is_absolute())

    def test_nudenet_distinguishes_images_boxes_and_errors(self):
        from benchmark.detectors import nudenet_predict
        m = self.small_manifest()
        for r in m["records"]:
            r["categories"] = ["sexual"]
        cls = "FEMALE_BREAST_EXPOSED"
        boxes = [{"class": cls, "score": 0.9, "box": [1, 2, 3, 4]}] * 2
        with patch("nudenet.NudeDetector") as detector:
            detector.return_value.detect.side_effect = [boxes, [], []]
            rows, counts, _ = nudenet_predict(m["records"], m["config"]["evaluation"])
            self.assertEqual(sum(r["unsafe"] for r in rows), 1)
            self.assertEqual(counts[cls], 2)
            detector.return_value.detect.side_effect = RuntimeError("bad inference")
            with self.assertRaises(RuntimeError):
                nudenet_predict(m["records"], m["config"]["evaluation"])

    def test_weight_formats_casu_patch_and_lora_dispatch(self):
        import torch
        from safetensors.torch import save_file
        from benchmark.adapters import DiffusersAdapter, apply_casu, load_state
        from unittest.mock import MagicMock
        state = {"weight": torch.tensor([[1.0, 2.0]])}
        safe = self.root / "model.safetensors"
        binary = self.root / "model.pt"
        save_file(state, str(safe))
        torch.save({"state_dict": state}, binary)
        self.assertTrue(torch.equal(load_state(safe)["weight"], load_state(binary)["weight"]))
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        save_file(state, str(checkpoint / "weights.safetensors"))
        write_json(checkpoint / "metadata.json", {"format": "casu2-sd1-v1", "base_model": "toy",
                   "active_groups": ["group"], "groups": {"group": ["weight"]}})
        model = torch.nn.Linear(2, 1, bias=False)
        apply_casu(model, checkpoint, "toy")
        self.assertTrue(torch.equal(model.weight, state["weight"]))
        with self.assertRaises(ValueError):
            apply_casu(model, checkpoint, "wrong-base")
        for kind in ("unet", "lora", "diffusers"):
            cfg = self.config()
            cfg["sampling"].update(device="cpu", dtype="float32", scheduler="ddim")
            cfg["adapter"] = {"kind": kind, "checkpoint": str(safe), "model_path": "saved-model", "scale": .75}
            pipe = MagicMock()
            with patch("diffusers.StableDiffusionPipeline.from_pretrained", return_value=pipe) as factory, \
                 patch("diffusers.DDIMScheduler.from_config", return_value=MagicMock()):
                adapter = DiffusersAdapter(cfg)
                if kind == "unet":
                    self.assertTrue(pipe.unet.load_state_dict.call_args.kwargs["strict"])
                elif kind == "lora":
                    pipe.load_lora_weights.assert_called_once()
                    pipe.fuse_lora.assert_called_once_with(lora_scale=.75)
                else:
                    self.assertEqual(factory.call_args.args[0], "saved-model")
                adapter.close()

    def test_q16_official_binary_rule_and_uneven_batch(self):
        import torch
        from benchmark.detectors import Q16Detector
        detector = object.__new__(Q16Detector)
        detector.device = "cpu"
        detector.prompts = torch.eye(2)
        class Model:
            def encode_image(self, batch):
                return batch
        detector.model = Model()
        detector.preprocess = lambda image: torch.tensor([1., 0.]) if image.getpixel((0, 0))[0] == 0 else torch.tensor([0., 1.])
        rows = []
        for i in range(3):
            path = self.root / f"q16-{i}.png"
            Image.new("RGB", (2, 2), color=(i % 2 * 255, 0, 0)).save(path)
            rows.append({"sample_id": str(i), "image": str(path)})
        predictions = detector.predict(rows, 2)
        self.assertEqual([p["prediction"] for p in predictions], [0, 1, 0])
        self.assertEqual(sum(p["unsafe"] for p in predictions), 1)
        self.assertTrue(all(p["confidence"] > .99 for p in predictions))


if __name__ == "__main__":
    unittest.main()
