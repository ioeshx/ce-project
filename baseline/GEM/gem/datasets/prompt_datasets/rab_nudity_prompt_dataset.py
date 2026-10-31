from gem.datasets.prompt_dataset import PromptDataset, PromptItem
from datasets import load_dataset


class RABNudityPromptDataset(PromptDataset):

    def __init__(self, max_samples: int = None):

        self.dataset = load_dataset("Chia15/RingABell-Nudity", split='train')
        self.max_samples = max_samples

        name_parts = ["rab_nudity"]

        if max_samples is not None:
            name_parts.append(f"n{max_samples}")

        name = "_".join(name_parts)

        super(RABNudityPromptDataset, self).__init__(name=name)

    def _load_prompt_items(self):
        prompts = []
        for idx, row in enumerate(self.dataset):

            prompts.append(PromptItem(prompt=row["prompt"], seed=row["evaluation_seed"]))

            if self.max_samples and len(prompts) >= self.max_samples:
                break

        return prompts
