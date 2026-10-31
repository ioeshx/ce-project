import os
from collections import Counter

import torch
from gem.evaluation.base_evaluator import BaseEvaluator
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm

import pandas as pd


class CLIPScorer:
    """
    Computes the CLIP cosine similarity between an image and a corresponding text prompt.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Load the Hugging Face CLIP model + processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, image_path: str, text_prompt: str):
        """
        Returns a dict with:
          - score: cosine similarity between image and text prompt (float)
        """
        try:
            # Load and preprocess the image + text
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True)
            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Run the model (no gradient needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds  # shape: (1, D)
                text_embeds = outputs.text_embeds  # shape: (1, D)

            # Normalize the embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity: (1, D) @ (D, 1) → scalar
            similarity = (image_embeds @ text_embeds.T).item()

            return {"score": similarity}

        except Exception as e:
            return {"score": 0.0, "error": str(e)}


class CLIPEvaluator(BaseEvaluator):

    def process_images(self, image_file_paths, dataset=None, **kwargs) -> dict:
        #assert dataset, "CLIP evaluation requires a reference dataset!"

        # Create the CLIP scorer
        scorer = CLIPScorer()

        # Initialize counters
        summary = Counter()
        scores = []

        if dataset is not None:
            # Collect and sort image files
            image_file_paths = sorted(image_file_paths)

            if len(image_file_paths) != len(dataset):
                raise ValueError(
                    f"Number of images ({len(image_file_paths)}) does not match dataset length ({len(dataset)})."
                )

            prompts = [item.prompt for item in dataset]

        else:
            # read them form index.csv in the folder
            index_file = os.path.join(os.path.dirname(image_file_paths[0]), "index.csv")

            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Index file not found at {index_file}")

            table = pd.read_csv(index_file)
            prompts = table['prompt'].to_list()

        print(f"Scanning {len(image_file_paths)} images ...\n")
        for image_path, prompt in tqdm(zip(image_file_paths, prompts), desc="Processing images"):
            result = scorer(image_path, prompt)

            score = result.get("score", 0.0)
            scores.append(score)

            if "error" in result:
                summary["errors"] += 1

        # Final summary
        total = len(image_file_paths)
        avg_score = sum(scores) / total if total > 0 else 0.0

        return {
            'total': total,
            'errors': summary['errors'],
            'clip_score': avg_score
        }






