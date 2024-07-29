import numpy as np
from torch.utils.data import Dataset

from PIL import Image
from .image_utils import nlbin


class ImageDataset(Dataset):
    
    """
    A custom Dataset class for loading and transforming images for machine learning tasks.

    Attributes:
        image_paths (list[str]): A list of file paths to the images.
        char2idx (dict): A dictionary mapping characters to indices.
        width (int): The target width of the images.
        height (int): The target height of the images.
        transform (callable, optional): A function/transform to apply to the images.
        has_text (bool): A flag indicating whether the images contain text.

    Methods:
        __init__(self, image_paths: list[str], char2idx: dict, width: int, height: int, transform=None, has_text=True):
            Initializes the ImageDataset with the given parameters.
    """
    def __init__(
        self,
        image_paths: list[str],
        char2idx: dict,
        width: int,
        height: int,
        transform=None,
        has_text=True,
    ):
        """
        Initializes the ImageDataset with the given parameters.

        Args:
            image_paths (list[str]): A list of file paths to the images.
            char2idx (dict): A dictionary mapping characters to indices.
            width (int): The target width of the images.
            height (int): The target height of the images.
            transform (callable, optional): A function/transform to apply to the images. Default is None.
            has_text (bool): A flag indicating whether the images contain text. Default is True.
        """
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

        image = nlbin(image)
        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
        }
