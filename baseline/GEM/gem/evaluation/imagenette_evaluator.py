import os
from collections import defaultdict, Counter

import numpy as np

from gem.datasets.prompts.simple_imagenette_dataset import IMAGENETTE_CLASSES
from gem.evaluation.base_evaluator import BaseEvaluator
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import torch


IMAGENET_TO_IMAGENETTE = {
    0: 0, 217: 1, 482: 2, 491: 3, 497: 4,
    566: 5, 569: 6, 571: 7, 574: 8, 701: 9
}


class ImagenetteClassifier:

    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval().to(self.device)

        self.transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()

        # Imagenette class names
        self.class_names = IMAGENETTE_CLASSES
        self.imagenet_to_imagenette = IMAGENET_TO_IMAGENETTE

    def __call__(self, image_path: str):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                predicted_idx = outputs.argmax(1).item()
                imagenette_idx = self.imagenet_to_imagenette.get(predicted_idx, None)

                if imagenette_idx is None:
                    label = "out_of_imagenette"
                else:
                    label = self.class_names[imagenette_idx]

            return {
                "label": label,
                "error": None
            }

        except Exception as e:
            return {
                "label": "error",
                "error": str(e)
            }


class ImagenetteEvaluator(BaseEvaluator):

    def process_images(self, image_file_paths, target_class: str = None, **kwargs) -> dict:
        classifier = ImagenetteClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')

        summary = Counter()
        per_class_results = defaultdict(lambda: {"correct": 0, "total": 0})

        print(f"Evaluating {len(image_file_paths)} Imagenette images...\n")

        for image_path in tqdm(image_file_paths, desc="Classifying"):
            filename = os.path.basename(image_path).lower()

            # Infer ground truth class from filename
            gt_label = next(
                (cls for cls in classifier.class_names if cls.replace(" ", "-") in filename),
                None
            ) or target_class

            if gt_label is None:
                raise ValueError(f"Could not infer ground truth label from filename ({filename}) or provided target class ({target_class}) ...")

            result = classifier(image_path)

            if result["error"] or gt_label is None:
                summary["errors"] += 1
                continue

            predicted_label = result["label"]
            per_class_results[gt_label]["total"] += 1

            if predicted_label == gt_label:
                summary["correct"] += 1
                per_class_results[gt_label]["correct"] += 1
            else:
                summary["incorrect"] += 1

        total_evaluated = summary["correct"] + summary["incorrect"] + summary["errors"]
        accuracy = summary["correct"] / total_evaluated if total_evaluated > 0 else 0.0

        if target_class:
            # Format per-class accuracy
            class_accuracies = {
                cls: round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0
                for cls, v in per_class_results.items()
            }

            target_acc = class_accuracies.get(target_class)
            other_acc = np.mean([class_accuracies.get(c) for c in class_accuracies if c != target_class]).item()
            per_class_results = {
                "target_acc": target_acc,
                "other_acc": other_acc
            }

        return {
            "total": total_evaluated,
            "correct": summary["correct"],
            "incorrect": summary["incorrect"],
            "errors": summary["errors"],
            "accuracy": round(accuracy, 4),
            **per_class_results
        }
