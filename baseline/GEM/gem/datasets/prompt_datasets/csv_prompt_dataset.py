import csv
from typing import List

from gem.datasets import PromptDataset, PromptItem


class CSVPromptDataset(PromptDataset):
    """
    A PromptDataset implementation that loads prompt items from a CSV file.

    The CSV is expected to have a header row and a column named 'prompt'.
    """

    def __init__(self, filepath: str, name: str = None, seed: int = None, max_samples: int = None):
        """
        Initializes the CSV prompt dataset.

        Args:
            filepath: The path to the CSV file.
            name: Optional name for the dataset.
            seed: Optional seed to associate with the prompts.
        """
        self.seed = seed
        self.filepath = filepath
        self.max_samples = max_samples

        # Calls _load_prompt_items() during initialization
        super(CSVPromptDataset, self).__init__(name=name)

    def _load_prompt_items(self) -> List[PromptItem]:
        """
        Loads the prompt items from the specified CSV file.
        Expects a column named 'prompt'.
        """
        prompt_items = []
        try:
            with open(self.filepath, 'r', newline='', encoding='utf-8') as csvfile:
                # Use DictReader to read rows as dictionaries,
                # using the header row (e.g., 'idx', 'prompt') as keys.
                reader = csv.DictReader(csvfile)

                if 'prompt' not in reader.fieldnames:
                    raise ValueError(f"CSV file must contain a 'prompt' column. Found: {reader.fieldnames}")

                for idx, row in enumerate(reader):
                    prompt_text = row['prompt'].strip()
                    # Only create a PromptItem if the prompt is not empty
                    if prompt_text:
                        prompt_items.append(PromptItem(prompt_text, (self.seed + idx) if self.seed is not None else None))

                    if self.max_samples and len(prompt_items) >= self.max_samples:
                        break

        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            # Depending on desired error handling, you might re-raise or return empty
            return []

        except Exception as e:
            print(f"An error occurred while loading the CSV: {e}")
            return []

        return prompt_items