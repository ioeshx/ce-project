"""Move ce-new to ce-project with the original checkout still present."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark.common import ROOT, configuration, read_json, read_result_csv, write_csv, write_json
from benchmark.data import load_manifest, prepare
from benchmark.metrics import cached_features, feature_fingerprint
from benchmark.migrate_paths import migrate


class PortablePathTests(unittest.TestCase):
    def prepare_fixture(self, project):
        bench = project / "benchmark"
        (bench / "dataset").mkdir(parents=True)
        shutil.copy2(ROOT / "dataset/mscoco.csv", bench / "dataset/mscoco.csv")
        with patch("benchmark.common.ROOT", bench):
            config = configuration(task="mscoco", base={"dataset": {"path": "dataset/mscoco.csv", "limit_prompts": 2},
                "sampling": {"device": "cpu", "dtype": "float32", "height": 32, "width": 32}})
            manifest_path = prepare(config)
            manifest = load_manifest(manifest_path)
        for record in manifest["records"]:
            Image.new("RGB", (32, 32)).save(record["reference"])
        return bench, manifest_path, manifest

    def test_relative_manifest_links_registry_and_fid_survive_move(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "ce-new", Path(temp) / "ce-project"
            bench, path, manifest = self.prepare_fixture(a)
            raw = json.loads(path.read_text())
            self.assertEqual(raw["path_base"], "benchmark")
            self.assertEqual(raw["config"]["evaluation"]["q16_prompts"], "weights/Q16/ViT-L-14/prompts.p")
            self.assertNotIn(str(a), path.read_text())
            self.assertTrue(all(not Path(r["image"]).is_absolute() for r in raw["records"]))
            images = [Path(r["reference"]) for r in manifest["records"]]
            features = np.ones((2, 2048), dtype=np.float32)
            cache = bench / "cache/fid/fixture/features.npz"
            cache.parent.mkdir(parents=True)
            np.savez_compressed(cache, features=features, fingerprint=feature_fingerprint(images))
            shutil.copytree(a, b, symlinks=True, copy_function=shutil.copy2)
            moved_bench, moved_path = b / "benchmark", b / path.relative_to(a)
            with patch("benchmark.common.ROOT", moved_bench):
                moved = load_manifest(moved_path)
                self.assertTrue(all(str(Path(r["image"])).startswith(str(b)) for r in moved["records"]))
                self.assertTrue(all(Path(r["image"]).is_file() for r in moved["records"]))
                self.assertTrue(all(not Path(r["image"]).readlink().is_absolute() for r in moved["records"]))
                prepared = load_manifest(prepare(moved["config"]))
                self.assertEqual(prepared["records"], moved["records"])
                with patch("benchmark.adapters.create_adapter", side_effect=AssertionError("must not sample again")):
                    from benchmark.adapters import sample_manifest
                    self.assertEqual(sample_manifest(moved)["row"]["status"], "cached")
                def extract(paths):
                    self.fail("Moved FID features should be reusable")
                loaded = cached_features([Path(r["reference"]) for r in moved["records"]],
                                         moved_bench / cache.relative_to(bench), extract)
                np.testing.assert_array_equal(loaded, features)

    def test_legacy_absolute_paths_rebase_without_using_original_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "ce-new", Path(temp) / "ce-project"
            bench, path, manifest = self.prepare_fixture(a)
            write_json(path, manifest)  # Recreate a legacy absolute-path manifest.
            for record in manifest["records"]:
                image = Path(record["image"])
                image.unlink()
                image.symlink_to(record["reference"])
            images = [Path(r["reference"]) for r in manifest["records"]]
            cache = bench / "cache/fid/fixture/features.npz"
            cache.parent.mkdir(parents=True)
            values = np.arange(4096, dtype=np.float32).reshape(2, 2048)
            old_fingerprint = feature_fingerprint(images, portable=False)
            np.savez_compressed(cache, features=values, fingerprint=old_fingerprint)
            write_json(cache.parent / "run.json", {"schema": "benchmark-readable-run-v1", "identity": {"feature_fingerprint": old_fingerprint}})
            write_csv(bench / "results/sd14/mscoco-metric.csv", [{"method": "original", "task": "mscoco", "cs": 23.5,
                      "fid": 0.1, "manifest": str(path)}])
            pixels = [image.read_bytes() for image in images]
            shutil.copytree(a, b, symlinks=True, copy_function=shutil.copy2)
            moved_bench, moved_path = b / "benchmark", b / path.relative_to(a)
            with patch("benchmark.common.ROOT", moved_bench):
                moved = load_manifest(moved_path)
                self.assertTrue(all(str(Path(r["reference"])).startswith(str(b)) for r in moved["records"]))
                self.assertTrue(all(not Path(r["image"]).readlink().is_absolute() for r in moved["records"]))
                log = migrate(moved_bench, apply=True)
                self.assertEqual(log["feature_caches_rebased"], 1)
                self.assertNotIn(str(a), moved_path.read_text())
                with np.load(moved_bench / cache.relative_to(bench)) as saved:
                    np.testing.assert_array_equal(saved["features"], values)
                    self.assertEqual(str(saved["fingerprint"].item()), feature_fingerprint([Path(r["reference"]) for r in moved["records"]]))
                rows = read_result_csv(moved_bench / "results/sd14/mscoco-metric.csv")
                self.assertEqual(rows[0]["manifest"], str(moved_path))
                self.assertEqual((float(rows[0]["cs"]), float(rows[0]["fid"])), (23.5, .1))
                self.assertEqual([Path(r["reference"]).read_bytes() for r in moved["records"]], pixels)
                self.assertEqual(migrate(moved_bench, apply=True)["files"], [])

    def test_local_models_are_relative_and_prompt_text_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temp:
            bench = Path(temp) / "ce-new/benchmark"
            (bench / "models/flux").mkdir(parents=True)
            path = bench / "configs/method.json"
            document = {"base_model": str(bench / "models/flux"), "prompt": str(bench / "must stay literal"),
                        "adapter": {"checkpoint": str(bench / "weights/lora")},
                        "evaluation": {"clip_model": "openai/clip-vit-base-patch16"}}
            write_json(path, document, portable=True)
            raw = json.loads(path.read_text())
            self.assertEqual(raw["base_model"], "./models/flux")
            self.assertEqual(raw["prompt"], document["prompt"])
            self.assertEqual(raw["evaluation"]["clip_model"], "openai/clip-vit-base-patch16")
            self.assertEqual(read_json(path), document)

    def test_wrapper_uses_its_checkout_and_preserves_caller_relative_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "ce-new", Path(temp) / "ce-project"
            for project in (a, b):
                bench = project / "benchmark"
                bench.mkdir(parents=True)
                for source in ROOT.glob("*.py"):
                    shutil.copy2(source, bench / source.name)
                shutil.copytree(ROOT / "configs", bench / "configs")
                shutil.copytree(ROOT / "script", bench / "script")
            profiles = b / "benchmark/configs/dit_sampling.json"
            values = json.loads(profiles.read_text())
            values["profiles"]["flux1_dev"]["sampling"]["steps"] = 999
            profiles.write_text(json.dumps(values))
            env = dict(os.environ, PYTHON=sys.executable)
            command = ["bash", str(a / "benchmark/script/dit.sh"), "--model", "flux1_dev", "--dry-run"]
            result = subprocess.run(command, cwd=b, env=env, check=True, capture_output=True, text=True)
            self.assertIn("flow-euler28", result.stdout)
            self.assertNotIn("flow-euler999", result.stdout)
            (b / "caller.json").write_text(json.dumps({"sampling": {"steps": 17}}))
            result = subprocess.run([*command, "--config", "caller.json"], cwd=b, env=env, check=True, capture_output=True, text=True)
            self.assertIn("flow-euler17", result.stdout)
            self.assertFalse((a / "benchmark/results").exists())
            self.assertFalse((b / "benchmark/results").exists())


if __name__ == "__main__":
    unittest.main()
