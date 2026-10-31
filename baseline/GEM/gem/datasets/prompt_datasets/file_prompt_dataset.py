from gem.datasets.prompt_dataset import PromptDataset, PromptItem


class FilePromptDataset(PromptDataset):
    def __init__(self, filepath, name: str = None, seed: int = None):
        self.seed = seed
        self.filepath = filepath
        super(FilePromptDataset, self).__init__(name=name)

    def _load_prompt_items(self):
        with open(self.filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return [PromptItem(line, (self.seed + idx) if self.seed is not None else None) for idx, line in enumerate(lines)]
