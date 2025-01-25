import numpy as np
from kraken import pageseg, binarization
from PIL import Image


def segment(image):
    """
    image: np.array
    """
    # binarize

    if isinstance(image, (np.ndarray, np.generic)):
        image = Image.fromarray(image)

    image = binarization.nlbin(image)
    # segment
    regions = pageseg.segment(image)
    return regions
