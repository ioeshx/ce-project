import glob
import os
import torch
import hashlib
from pathlib import Path
from cleanfid import fid
from gem.evaluation.base_evaluator import BaseEvaluator


class FIDEvaluator(BaseEvaluator):

    def _get_stats_name(self, ref_dir: str) -> str:
        """Generates a unique, reproducible name for the reference folder."""
        abs_path = os.path.abspath(ref_dir)
        # Create a short hash of the path to ensure uniqueness and filesystem safety
        path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:8]
        folder_name = Path(abs_path).name.lower().replace("-", "_")
        return f"{folder_name}_{path_hash}"

    def _prepare_reference_stats(self, ref_dir: str):
        """Checks if stats exist for this folder; if not, precomputes them."""
        stats_name = self._get_stats_name(ref_dir)

        if not fid.test_stats_exists(stats_name, mode="clean"):
            print(f"--- Precomputing FID statistics for: {ref_dir} ---")
            print(f"--- Saving as: {stats_name} ---")
            fid.make_custom_stats(stats_name, ref_dir, mode="clean")

        return stats_name

    def process_images(self, image_file_paths, dataset=None, **kwargs) -> dict:
        # 1. Determine the reference directory
        reference_folder = kwargs.get("reference_folder")
        sample_folder = os.path.dirname(image_file_paths[0])

        print("Computing FID between:")
        print("-> Reference folder:", reference_folder, f"-> {len(os.listdir(reference_folder))} images")
        print("-> Sample folder:", sample_folder, f"-> {len(os.listdir(sample_folder))} images -> ", glob.escape(sample_folder))

        if not reference_folder:
            return {'total': 0, 'fid_score': float('inf'), 'error': 'No reference_folder provided'}

        stats_name = self._prepare_reference_stats(reference_folder)
        assert fid.test_stats_exists(stats_name, mode="clean")
        print("Prepared stats with name:", stats_name)

        # Compute FID Score using the cached stats_name
        score = fid.compute_fid(glob.escape(sample_folder), glob.escape(reference_folder), mode="clean")

        return {
            'total': len(image_file_paths),
            'fid_score': float(score),
            'mode': 'clean',
            'stats_used': stats_name
        }