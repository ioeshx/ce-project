from gem.datasets.prompt_dataset import PromptDataset, PromptItem


class AdhocPromptDataset(PromptDataset):
    def __init__(
            self,
            prompts,
            name: str = "adhoc",
            num_images_per_prompt: int = 1,
            seed: int = None
    ):
        self.prompts = prompts
        self.seed = seed
        super(AdhocPromptDataset, self).__init__(name=name, num_images_per_prompt=num_images_per_prompt)

        assert isinstance(prompts, list), f"Provided prompts must be a list of strings, but got: {prompts}"

    def _load_prompt_items(self):
        return [
            PromptItem(prompt, self.seed if self.seed is not None else None) for idx, prompt in enumerate(self.prompts)
        ]