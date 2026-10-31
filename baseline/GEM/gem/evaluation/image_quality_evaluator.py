import os
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import torch
import lpips

from gem.evaluation.base_evaluator import BaseEvaluator
from gem.utils.memory import memory_cleanup


def preprocess_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return np.array(img)


def to_tensor(img):
    return torch.tensor(img / 255.0).permute(2, 0, 1).unsqueeze(0).float()


def compute_metrics(img1_path, img2_path, lpips_model):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path, size=img1.shape[:2][::-1])

    gray1 = Image.fromarray(img1).convert("L")
    gray2 = Image.fromarray(img2).convert("L")
    ssim_score = ssim(np.array(gray1), np.array(gray2), data_range=255)

    psnr_score = psnr(img1, img2, data_range=255)

    diff = img1.astype(np.float32) - img2.astype(np.float32)
    mse_score = np.mean(diff ** 2)
    mae_score = np.mean(np.abs(diff))

    img1_t = to_tensor(img1).cuda()
    img2_t = to_tensor(img2).cuda()

    with torch.no_grad():
        lpips_score = lpips_model(img1_t, img2_t).cpu().item()

    return {
        "ssim": ssim_score,
        "psnr": psnr_score,
        "mse": mse_score,
        "mae": mae_score,
        "lpips": lpips_score
    }


class ImageQualityEvaluator(BaseEvaluator):

    def process_images(self, image_file_paths, reference_folder=None, **kwargs) -> dict:
        assert reference_folder, "Reference folder must be provided for pairwise image quality evaluation"

        image_folder = os.path.dirname(image_file_paths[0])
        print(f"Comparing {len(image_file_paths)} image pairs from '{image_folder}' and '{reference_folder}'...\n")

        lpips_model = lpips.LPIPS(net='alex').cuda()

        all_metrics = defaultdict(list)
        for image_a_path in tqdm(image_file_paths, desc="Computing metrics"):
            image_b_path = os.path.join(reference_folder, os.path.basename(image_a_path))

            if not os.path.exists(image_b_path):
                print(f"Warning: Missing corresponding image for '{image_a_path}' in '{reference_folder}'")
                continue

            metrics = compute_metrics(image_a_path, image_b_path, lpips_model)

            for key, value in metrics.items():
                all_metrics[key].append(value)

        summary = {"total": len(image_file_paths)}
        summary.update({key: np.mean(values).item() for key, values in all_metrics.items()})

        del lpips_model
        memory_cleanup()
        return summary







