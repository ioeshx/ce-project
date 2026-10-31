from typing import Optional, List


class PromptItem:
    def __init__(self, prompt: Optional[str], seed: int = None, prompt_embeds: tuple = None):
        self.prompt = prompt
        self.seed = seed
        self.prompt_embeds = prompt_embeds


class PromptDataset:

    def __init__(self, name: str = None, num_images_per_prompt: int = 1):
        self.name = name
        self.num_images_per_prompt = num_images_per_prompt
        self.prompt_items: List[PromptItem] = self._load_prompt_items()

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.prompt_items)

    def __len__(self):
        return len(self.prompt_items)

    def _load_prompt_items(self):
        raise NotImplementedError

    def apply_augmentation(self, prompt_augmentation):
        for item in self.prompt_items:
            item.prompt = prompt_augmentation.apply(item.prompt)[0]
