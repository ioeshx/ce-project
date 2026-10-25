"""Readable names must preserve cache isolation and migration compatibility."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from benchmark.common import configuration, digest, read_json, slug, write_json
from benchmark.data import load_manifest, prepare
from benchmark.layout import method_directory, sampling_identity


class LayoutTests(unittest.TestCase):
    def test_cifar50_results_can_render_without_overlong_filenames_or_titles(self):
        import json
        from benchmark.erasure_protocol import CIFAR50_TARGETS
        from benchmark.plotting import plot
        from benchmark.results import protocol_label, upsert
        from benchmark.sampling_profile import defaults
        with tempfile.TemporaryDirectory() as temp:
            row = {"task": "mscoco", "method": "test-only-cifar50", "targets": json.dumps(CIFAR50_TARGETS),
                   "sampling": json.dumps(defaults()), "base_model": "test fixture",
                   "protocol_id": "cifar50-test-only", "n": 1000, "cs": 1, "fid": 2}
            upsert(temp, "mscoco", row)
            paths = plot(temp, "mscoco")
            self.assertEqual(len(paths), 2)
            self.assertIn("CIFAR-50", protocol_label(row))
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertIn("mscoco-cifar-50-", path.name)
                self.assertLess(len(path.name.encode()), 255)

    def test_readable_name_collisions_do_not_merge_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            a = method_directory(temp, "instance", "a/b")
            b = method_directory(temp, "instance", "a?b")
            self.assertEqual(a.name, "a-b")
            self.assertEqual(b.name, "a-b-run-2")
            self.assertNotEqual(a, b)
            self.assertEqual(a, method_directory(temp, "instance", "a/b"))
            self.assertEqual(slug("../Van Gogh"), "Van-Gogh")

    def test_time_named_cache_reuses_exact_identity_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = configuration(task="mscoco")
            cfg.update(output_dir=str(root / "results"), cache_dir=str(root / "cache"), original_dir=str(root / "original"))
            cfg["dataset"]["limit_prompts"] = 2
            first = load_manifest(prepare(cfg))
            cfg["method"] = "comparison"
            second = load_manifest(prepare(cfg))
            self.assertEqual(first["records"][0]["reference"], second["records"][0]["reference"])
            self.assertIn("mscoco-2-captions-dpm20", first["records"][0]["reference"])
            cfg["sampling"]["steps"] += 1
            third = load_manifest(prepare(cfg))
            self.assertNotEqual(first["records"][0]["reference"], third["records"][0]["reference"])
            self.assertEqual(read_json(Path(first["records"][0]["reference"]).parent / "run.json")["identity"]["sampling"]["steps"], 20)

    def test_migration_preserves_images_manifests_and_resumption(self):
        from benchmark.migrate_layout import Migration
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "benchmark"
            cfg = configuration(task="mscoco")
            cfg.update(output_dir=str(root / "results"), cache_dir=str(root / "cache"), original_dir=str(root / "original"))
            cfg["dataset"]["limit_prompts"] = 2
            from benchmark.data import prompt_records, sampling_protocol
            records, source = prompt_records(cfg)
            protocol = sampling_protocol(cfg)
            old = root / "cache/originals" / digest(sampling_identity(records, cfg["base_model"], protocol))
            old.mkdir(parents=True)
            for r in records:
                p = old / (r["sample_id"] + ".png")
                Image.new("RGB", (512, 512)).save(p)
                r.update(image=str(p), reference=str(p))
            method = root / "results/mscoco/original-abcdef12"
            manifest = method / "manifest.json"
            write_json(manifest, {"schema": "concept-benchmark-v1", "task": "mscoco", "targets": [], "method": "original",
                       "base_model": cfg["base_model"], "sampling": protocol, "dataset": source,
                       "adapter_identity": {"kind": "original"}, "config": cfg, "records": records})
            migration = Migration(root).plan()
            migration.apply(remove_tests=True)
            new_path = Path(migration.translated(manifest))
            loaded = load_manifest(new_path)
            self.assertFalse(old.exists())
            self.assertEqual(loaded["sampling"], protocol)
            self.assertTrue(all(Path(r["image"]).is_file() for r in loaded["records"]))
            after = load_manifest(prepare(cfg))
            self.assertEqual(after["records"][0]["reference"], loaded["records"][0]["reference"])
            self.assertEqual(Migration(root).plan().moves, {})


if __name__ == "__main__":
    unittest.main()
