from gem.datasets.prompt_dataset import PromptDataset, PromptItem

import pandas as pd


class COCOC30KPromptDataset(PromptDataset):

    def __init__(self, coco_30k_csv_path="gem/datasets/prompt_datasets/data/coco/coco_30k.csv", max_samples: int = 100, name: str = 'coco_30k', num_images_per_prompt: int = 1):
        self.coco_30k_csv_path = coco_30k_csv_path
        self.max_samples = max_samples
        super().__init__(name=name + f'_n{max_samples}', num_images_per_prompt=num_images_per_prompt)

    def _load_prompt_items(self):
        entries = pd.read_csv(self.coco_30k_csv_path, nrows=self.max_samples)

        return [
            PromptItem(prompt=row.prompt, seed=row.evaluation_seed)
            for _, row in entries.iterrows()
        ]