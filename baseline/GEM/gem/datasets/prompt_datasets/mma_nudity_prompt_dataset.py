from gem.datasets.prompt_dataset import PromptDataset, PromptItem
from datasets import load_dataset


class MMANudityPromptDataset(PromptDataset):

    def __init__(self, max_samples: int = None):

        self.dataset = load_dataset("YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark", split='train')
        self.max_samples = max_samples

        name_parts = ["mma_nudity"]

        if max_samples is not None:
            name_parts.append(f"n{max_samples}")

        name = "_".join(name_parts)

        super(MMANudityPromptDataset, self).__init__(name=name)

    def _load_prompt_items(self):
        prompts = []
        for idx, row in enumerate(self.dataset):

            prompts.append(PromptItem(prompt=row["adv_prompt"], seed=0))

            if self.max_samples and len(prompts) >= self.max_samples:
                break

        return prompts
