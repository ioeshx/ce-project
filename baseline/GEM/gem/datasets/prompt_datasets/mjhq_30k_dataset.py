from gem.datasets.prompt_dataset import PromptDataset, PromptItem

import pandas as pd


class MJHQPromptDataset(PromptDataset):

    def __init__(self, mjhq_30k_csv_path="gem/datasets/prompt_datasets/data/mjhq/mjhq_30k.csv", max_samples: int = 10_000, name: str = 'mjhq_30k', num_images_per_prompt: int = 1):
        self.mjhq_30k_csv_path = mjhq_30k_csv_path
        self.max_samples = max_samples
        super().__init__(name=name + f'_n{max_samples}', num_images_per_prompt=num_images_per_prompt)

    def _load_prompt_items(self):
        entries = pd.read_csv(self.mjhq_30k_csv_path, nrows=self.max_samples)

        return [
            PromptItem(prompt=row.prompt, seed=None)
            for _, row in entries.iterrows()
        ]