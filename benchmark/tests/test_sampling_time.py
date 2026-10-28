"""Sampling clocks, GPU synchronization, cached resumes and latest-result updates."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from PIL import Image
import torch

from benchmark.adapters import atomic_image, sample_manifest
from benchmark.common import configuration, read_json
from benchmark.data import load_manifest, prepare
from benchmark.sampling_time import cuda_devices, synchronize


class SamplingTimeTests(unittest.TestCase):
    def manifest(self, root, method="fixture"):
        config = configuration(task="mscoco", base={"method": method, "output_dir": str(root / "results"),
            "original_dir": str(root / "original"), "dataset": {"limit_prompts": 3},
            "sampling": {"device": "cpu", "dtype": "float32", "height": 32, "width": 32, "batch_size": 2}})
        path = prepare(config)
        return path, load_manifest(path)

    def test_generation_load_save_and_gpu_wait_are_measured_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest = self.manifest(root)
            clock, events = [0.0], []
            class Adapter:
                def generate(self, records, options):
                    clock[0] += 2 * len(records)
                    events.append("generate")
                    return [Image.new("RGB", (32, 32)) for _ in records]
                def close(self):
                    events.append("close")
                    clock[0] += 1
            def load(config):
                clock[0] += 5
                events.append("load")
                return Adapter()
            def sync(devices):
                self.assertEqual(devices, [0, 1])
                if events[-1] == "generate":
                    clock[0] += 1  # Pending GPU work must count as generation.
                events.append("sync")
            def save(image, destination):
                clock[0] += .5
                events.append("save")
                atomic_image(image, destination)
            with patch("benchmark.adapters.time.perf_counter", side_effect=lambda: clock[0]), \
                 patch("benchmark.adapters.create_adapter", side_effect=load), \
                 patch("benchmark.adapters.atomic_image", side_effect=save), \
                 patch("benchmark.adapters.release_memory"), \
                 patch("benchmark.sampling_time.cuda_devices", return_value=[0, 1]), \
                 patch("benchmark.sampling_time.synchronize", side_effect=sync):
                result = sample_manifest(manifest, "image", manifest_path=path)
            row = result["row"]
            self.assertEqual(row["generated_images"], 3)
            self.assertEqual(row["reused_images"], 0)
            self.assertEqual(row["model_load_seconds"], 5)
            self.assertEqual(row["generation_seconds"], 8)
            self.assertEqual(row["image_save_seconds"], 1.5)
            self.assertEqual(row["cleanup_seconds"], 1)
            self.assertEqual(row["total_seconds"], 15.5)
            self.assertAlmostEqual(row["seconds_per_image"], 8 / 3)
            self.assertEqual(result["phases"][0]["cuda_devices"], [0, 1])
            self.assertEqual(events, ["load", "sync", "sync", "generate", "sync", "save", "save",
                                      "sync", "generate", "sync", "save", "close"])
            self.assertEqual(read_json(Path(row["details_file"]))["row"]["manifest"], str(path))
            with (root / "results/sd14/time/mscoco-sampling-time.csv").open() as handle:
                saved = next(csv.DictReader(handle))
            self.assertEqual(float(saved["generation_seconds"]), 8)
            self.assertTrue((root / "results/sd14/time/mscoco-sampling-time.txt").is_file())

    def test_resume_cached_other_methods_and_failure_do_not_erase_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, manifest = self.manifest(root)
            class Adapter:
                def generate(self, records, options):
                    return [Image.new("RGB", (32, 32)) for _ in records]
                def close(self):
                    pass
            with patch("benchmark.adapters.create_adapter", return_value=Adapter()) as create, \
                 patch("benchmark.adapters.release_memory"):
                first = sample_manifest(manifest, "image")
                first_path = Path(first["row"]["details_file"])
                before = first_path.read_bytes()
                cached = sample_manifest(manifest, "image")
                self.assertEqual(create.call_count, 1)
                self.assertEqual(cached["row"]["status"], "cached")
                self.assertIsNone(cached["row"]["seconds_per_image"])
                self.assertEqual(cached["row"]["model_load_seconds"], 0)
                self.assertEqual(first_path.read_bytes(), before)
                Path(manifest["records"][1]["image"]).write_bytes(b"corrupt fixture")
                resumed = sample_manifest(manifest, "image")
                self.assertEqual(resumed["row"]["generated_images"], 1)
                self.assertEqual(resumed["row"]["reused_images"], 2)
                other_path, other = self.manifest(root, "other-fixture")
                # Original aliases from the other method may reuse references;
                # either way the summary must retain this distinct method.
                sample_manifest(other, "image")
            csv_path = root / "results/sd14/time/mscoco-sampling-time.csv"
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({r["method"] for r in rows}, {"fixture", "other-fixture"})
            self.assertEqual(len(rows), 2)
            previous = csv_path.read_bytes()
            Path(manifest["records"][0]["image"]).unlink()
            failed = Adapter()
            with patch.object(failed, "generate", side_effect=RuntimeError("fixture inference failure")), \
                 patch.object(failed, "close") as close, \
                 patch("benchmark.adapters.create_adapter", return_value=failed), \
                 patch("benchmark.adapters.release_memory"):
                with self.assertRaisesRegex(RuntimeError, "fixture inference failure"):
                    sample_manifest(manifest, "image")
                close.assert_called_once()
            self.assertEqual(csv_path.read_bytes(), previous)

    def test_both_mode_records_reference_and_image_times_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            path, manifest = self.manifest(Path(temp))
            class Adapter:
                def generate(self, records, options):
                    return [Image.new("RGB", (32, 32)) for _ in records]
                def close(self):
                    pass
            with patch("benchmark.adapters.create_adapter", return_value=Adapter()) as create, \
                 patch("benchmark.adapters.release_memory"):
                document = sample_manifest(manifest, "both")
            # The original-model image aliases refer to the freshly filled
            # reference cache, so sampling must not generate the same set twice.
            self.assertEqual(create.call_count, 1)
            self.assertEqual(document["row"]["reference_generated_images"], 3)
            self.assertEqual(document["row"]["image_generated_images"], 0)
            self.assertEqual(document["row"]["image_reused_images"], 3)
            self.assertEqual([p["field"] for p in document["phases"]], ["reference", "image"])

    def test_cuda_sync_covers_sharded_models_and_optional_plugin_devices(self):
        module = torch.nn.Linear(2, 2)
        module.hf_device_map = {"block.0": 0, "block.1": 1, "offloaded": "cpu"}
        adapter = SimpleNamespace(pipe=SimpleNamespace(components={"transformer": module}),
                                  sampling_devices=["cuda:2"])
        with patch("torch.cuda.is_initialized", return_value=True), \
             patch("torch.cuda.current_device", return_value=0), patch("torch.cuda.synchronize") as sync:
            devices = cuda_devices(adapter, {"device": "cuda"})
            self.assertEqual(devices, [0, 1, 2])
            synchronize(devices)
            self.assertEqual(sync.call_args_list, [call(0), call(1), call(2)])
        with patch("torch.cuda.is_initialized", return_value=False), patch("torch.cuda.current_device") as current:
            self.assertEqual(cuda_devices(adapter, {"device": "cpu"}), [])
            current.assert_not_called()


if __name__ == "__main__":
    unittest.main()
