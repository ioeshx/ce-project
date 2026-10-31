import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from gem.evaluation.base_evaluator import BaseEvaluator


class CLIPScorer:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image_path: str, prompt: str) -> dict:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(
                text=[prompt],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            outputs = self.model(**inputs)

            # Cosine similarity between image and text embeddings, scaled to [0, 1]
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine_sim = (image_embeds * text_embeds).sum(dim=-1).item()
            clip_score = (cosine_sim + 1.0) / 2.0

            return {"clip_score": clip_score, "error": None}

        except Exception as e:
            return {"clip_score": None, "error": str(e)}


class CLIPToTextEvaluator(BaseEvaluator):
    """
    Evaluates a set of images against a static text prompt using CLIP cosine similarity.
    Returns per-image scores plus aggregate statistics (mean, std, min, max).
    """

    def __init__(self, prompt: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.prompt = prompt
        self.device = device
        self.scorer = CLIPScorer(model_name="openai/clip-vit-base-patch32", device=device)

    def process_images(self, image_file_paths, **kwargs) -> dict:

        scores = []
        per_image_results = {}
        errors = []

        print(f"Evaluating {len(image_file_paths)} images against prompt: '{self.prompt}'\n")

        for image_path in tqdm(image_file_paths, desc="Scoring with CLIP"):
            result = self.scorer(image_path, self.prompt)

            if result["error"] is not None:
                errors.append({"path": image_path, "error": result["error"]})
                continue

            clip_score = result["clip_score"]
            scores.append(clip_score)
            per_image_results[os.path.basename(image_path)] = round(clip_score, 4)

        scores_array = np.array(scores)

        return {
            "prompt": self.prompt,
            "total": len(image_file_paths),
            "evaluated": len(scores),
            "errors": len(errors),
            "mean_clip_score": round(float(scores_array.mean()), 4) if len(scores) > 0 else None,
            "std_clip_score": round(float(scores_array.std()), 4) if len(scores) > 0 else None,
            "min_clip_score": round(float(scores_array.min()), 4) if len(scores) > 0 else None,
            "max_clip_score": round(float(scores_array.max()), 4) if len(scores) > 0 else None
        }