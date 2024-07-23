import torch

import numpy as np
from os.path import basename, dirname
from torch.utils.data import Dataset

from PIL import Image
from kraken import binarization


class ImageDataset(Dataset):

    def __init__(
        self,
        image_paths: list[str],
        char2idx: dict,
        width: int,
        height: int,
        transform=None,
        has_text=True,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.char2idx = char2idx
        self.width = width
        self.height = height
        self.has_text = has_text

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if isinstance(image_path, str):
            image = Image.open(image_path).convert("L")
        else:
            image = image_path

        image = image.resize((self.width, self.height), Image.Resampling.BICUBIC)

        image = binarization.nlbin(image)
        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
        }
