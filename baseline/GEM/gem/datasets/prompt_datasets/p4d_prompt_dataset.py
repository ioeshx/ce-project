from gem.datasets.prompt_dataset import PromptDataset, PromptItem
from datasets import load_dataset


class P4DNudityPromptDataset(PromptDataset):

    def __init__(self, split: str = "train", max_samples: int = None):

        self.dataset = load_dataset("joycenerd/p4d", split=split)
        self.max_samples = max_samples

        name = f"p4d_nudity"

        if max_samples is not None:
            name += f"_n{max_samples}"

        super(P4DNudityPromptDataset, self).__init__(name=name)

    def _load_prompt_items(self):
        prompts = []
        for idx, row in enumerate(self.dataset):

            prompts.append(PromptItem(prompt=row["prompt"]))

            if self.max_samples and len(prompts) >= self.max_samples:
                break

        return prompts
