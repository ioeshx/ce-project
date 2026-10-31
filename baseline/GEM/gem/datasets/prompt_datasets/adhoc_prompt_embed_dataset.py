from gem.datasets.prompt_dataset import PromptDataset, PromptItem


class AdhocPromptEmbedDataset(PromptDataset):
    def __init__(self, name, prompt_embeds, num_images_per_prompt: int = 1, seed: int = None):
        self.prompt_embeds = prompt_embeds
        self.seed = seed
        super(AdhocPromptEmbedDataset, self).__init__(name=name, num_images_per_prompt=num_images_per_prompt)

        assert isinstance(prompt_embeds, list), "prompts must be a list of embeds"
        self.items = [PromptItem(None, seed, embeds) for embeds in prompt_embeds]

    def _load_prompt_items(self):
        return [
            PromptItem(None, (self.seed + idx) if self.seed is not None else None, embeds)
            for idx, embeds in enumerate(self.prompt_embeds)
        ]