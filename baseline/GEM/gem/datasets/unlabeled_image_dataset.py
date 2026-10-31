import os
import torch

from gem.utils.image_utils import load_image_from_source


class UnlabeledImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, n_images=None, transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_filenames = self.image_filenames[:n_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = load_image_from_source(image_path)

        if self.transform:
            image = self.transform(image)
        return image
