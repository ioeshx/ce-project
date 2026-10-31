import itertools
import random
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from datasets import load_dataset


class PromptAugmentation:

    def apply(self, *prompts, n_templates=None, randomize=True):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


@dataclass
class PromptAugmentationConfig:
    type: str
    params: Dict[str, Any]

    def create(self) -> PromptAugmentation:
        if self.type == "hf_dataset":
            return HFDatasetPromptAugmentation(**self.params)
        elif self.type == "template":
            return TemplatePromptAugmentation(**self.params)
        else:
            raise ValueError(f"Unknown augmentation type: {self.type}")


class HFDatasetPromptAugmentation(PromptAugmentation):

    def __init__(self, dataset_id, split, field, seed: int = 0, max_samples: int = None):
        self.dataset_id = dataset_id
        self.split = split
        self.field = field
        self.default_n_templates = 1
        self._rng = np.random.default_rng(seed)

        self.dataset = load_dataset(dataset_id, split=split, streaming=True)
        self.dataset = list(row[field] for row in itertools.islice(self.dataset, max_samples))

    def apply(self, *prompts, n_templates=None, randomize=True):

        if n_templates is None:
            n_templates = self.default_n_templates  # Default number of templates if not specified

        augmented_prompts = []
        for i in range(n_templates):

            if randomize:
                caption = self._rng.choice(self.dataset)
            else:
                caption = self.dataset[i]

            caption_words = caption.split()
            random_index = random.randint(0, len(caption_words))  # Random position in the caption

            for prompt in prompts:
                caption_words = caption_words[:random_index] + [prompt] + caption_words[random_index:]
                augmented = " ".join(caption_words)
                augmented_prompts.append(augmented)

        return list(augmented_prompts)

    def __len__(self):
        return self.default_n_templates


class TemplatePromptAugmentation(PromptAugmentation):
    def __init__(self, templates: List[str], seed: int = 0):
        self.templates = templates
        self._rng = np.random.default_rng(seed)

    def apply(self, *prompts, n_templates=None, randomize=True):
        augmented_prompts = []

        if n_templates is None:
            n_templates = 1

        if n_templates > len(self.templates):
            raise ValueError("Number of templates requested exceeds available templates.")

        if randomize:
            templates = self._rng.choice(self.templates, n_templates, replace=False)
        else:
            templates = self.templates[:n_templates]

        for t in templates:
            for prompt in prompts:
                augmented = t.format(prompt)
                augmented_prompts.append(augmented)

        return list(augmented_prompts)

    def __len__(self):
        return len(self.templates)


class PrefixPromptAugmentation(PromptAugmentation):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def apply(self, *prompts, n_templates=None, randomize=False):
        augmented_prompts = []

        for prompt in prompts:
            augmented = self.prefix + " " + prompt
            augmented_prompts.append(augmented)

        return list(augmented_prompts)

    def __len__(self):
        return 1
