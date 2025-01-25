import torch
import torch.nn as nn

import numpy as np
from os.path import basename, dirname
from torch.utils.data import Dataset

torch.manual_seed(1231)
import cv2
from PIL import Image
from kraken import binarization


torch.__version__

width = 760
height = 80

MAX_TEXT_LENGTH = 69


def get_text_path(image_file: str):
    txt_dir = dirname(image_file)
    filename = basename(image_file).replace(".bin.png", "")
    text_path = txt_dir + "/" + filename + ".gt.txt"
    return text_path


def read_text(text_path: str):

    with open(text_path, "r", encoding="utf-8-sig") as f:
        text = f.readline().strip()
    return text


def text_to_tensor(text, char2idx, MAX_TEXT_LENGTH=MAX_TEXT_LENGTH):
    t = torch.IntTensor([char2idx[c] for c in text])
    rest = MAX_TEXT_LENGTH - t.shape[0]
    text_length = t.shape[0]
    ret = torch.cat([t, torch.zeros(rest, dtype=torch.int32)])
    return ret, text_length


class ImageDataset(Dataset):

    def __init__(
        self,
        image_paths,
        char2idx,
        transform=None,
        has_text=True,
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.char2idx = char2idx

        self.has_text = has_text

        if self.has_text:
            self.textpaths = [get_text_path(image_path) for image_path in image_paths]
            self.texts = [read_text(text_path) for text_path in self.textpaths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path

        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        pil_image = Image.fromarray(image)
        image = binarization.nlbin(pil_image)
        image = np.array(image)

        if self.has_text:
            text = self.texts[idx]
            target, target_len = text_to_tensor(
                text, char2idx=self.char2idx, MAX_TEXT_LENGTH=MAX_TEXT_LENGTH
            )

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": text if self.has_text else "",
            "target": target if self.has_text else "",
            "target_len": target_len if self.has_text else "",
        }
