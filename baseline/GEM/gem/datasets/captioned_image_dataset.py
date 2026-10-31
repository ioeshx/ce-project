import csv

import torch

import os
from dotenv import load_dotenv

from gem.utils.image_utils import load_image_from_source

load_dotenv()


class LabeledImageDataset:

    def __init__(self, name, transform, max_samples):
        self.name = name
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        return self.max_samples


class LabeledImageFolderDataset(LabeledImageDataset, torch.utils.data.Dataset):

    def __init__(self, name, transform, max_samples, folder_path: str = None, text_transform=None):
        super(LabeledImageFolderDataset, self).__init__(name=name, transform=transform, max_samples=max_samples)
        self.folder_path = folder_path
        self.text_transform = text_transform

        index_csv_path = os.path.join(folder_path, 'index.csv')
        with open(index_csv_path, newline='') as csvfile:
            self.index = list(csv.DictReader(csvfile))

    def __len__(self):
        return self.max_samples or len(self.index)

    def __getitem__(self, idx):
        row = self.index[idx]
        filename = row['filename']
        prompt = row['prompt']

        if self.text_transform:
            prompt = self.text_transform(prompt)

        image_path = os.path.join(self.folder_path, filename)
        image = load_image_from_source(image_path)

        if self.transform:
            image = self.transform(image).unsqueeze(0)

        return {'caption': prompt, 'image': image}
