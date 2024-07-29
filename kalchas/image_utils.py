from PIL import Image
import torch
from typing import Union
import numpy as np
from scipy.ndimage import (
    affine_transform,
    percentile_filter,
    gaussian_filter,
    binary_dilation,
)
from scipy.ndimage import zoom as _zoom
import logging
import warnings


__all__ = ["nlbin"]

logger = logging.getLogger(__name__)


def nlbin(
    im: Image.Image,
    threshold: float = 0.5,
    zoom: float = 0.5,
    escale: float = 1.0,
    border: float = 0.1,
    perc: int = 80,
    range: int = 20,
    low: int = 5,
    high: int = 90,
) -> Image.Image:
    """
    Performs binarization using non-linear processing.

    Args:
        im: Input image
        threshold:
        zoom: Zoom for background page estimation
        escale: Scale for estimating a mask over the text region
        border: Ignore this much of the border
        perc: Percentage for filters
        range: Range for filters
        low: Percentile for black estimation
        high: Percentile for white estimation

    Returns:
        PIL.Image.Image containing the binarized image

    Raises:
        ValueError: When trying to binarize an empty image.
    """
    im_str = get_im_str(im)
    logger.info(f"Binarizing {im_str}")
    if is_bitonal(im):
        logger.info(f"Skipping binarization because {im_str} is bitonal.")
        return im
    # convert to grayscale first
    logger.debug(f"Converting {im_str} to grayscale")
    im = im.convert("L")
    raw = pil2array(im)
    logger.debug("Scaling and normalizing")
    # rescale image to between -1 or 0 and 1
    raw = raw / float(np.iinfo(raw.dtype).max)
    # perform image normalization
    if np.amax(raw) == np.amin(raw):
        logger.warning(f"Trying to binarize empty image {im_str}")
        raise ValueError("Image is empty")
    image = raw - np.amin(raw)
    image /= np.amax(image)

    logger.debug("Interpolation and percentile filtering")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = _zoom(image, zoom)
        m = percentile_filter(m, perc, size=(range, 2))
        m = percentile_filter(m, perc, size=(2, range))
        mh, mw = m.shape
        oh, ow = image.shape
        scale = np.diag([mh * 1.0 / oh, mw * 1.0 / ow])
        m = affine_transform(m, scale, output_shape=image.shape)
    w, h = np.minimum(np.array(image.shape), np.array(m.shape))
    flat = np.clip(image[:w, :h] - m[:w, :h] + 1, 0, 1)

    # estimate low and high thresholds
    d0, d1 = flat.shape
    o0, o1 = int(border * d0), int(border * d1)
    est = flat[o0 : d0 - o0, o1 : d1 - o1]
    logger.debug("Threshold estimates {}".format(est))
    # by default, we use only regions that contain
    # significant variance; this makes the percentile
    # based low and high estimates more reliable
    logger.debug("Refine estimates")
    v = est - gaussian_filter(est, escale * 20.0)
    v = gaussian_filter(v**2, escale * 20.0) ** 0.5
    v = v > 0.3 * np.amax(v)
    v = binary_dilation(v, structure=np.ones((int(escale * 50), 1)))
    v = binary_dilation(v, structure=np.ones((1, int(escale * 50))))
    est = est[v]
    lo = np.percentile(est.ravel(), low)
    hi = np.percentile(est.ravel(), high)
    flat -= lo
    flat /= hi - lo
    flat = np.clip(flat, 0, 1)
    logger.debug(f"Thresholding at {threshold}")
    bin = np.array(255 * (flat > threshold), "B")
    return array2pil(bin)


def array2pil(a: np.ndarray) -> Image.Image:
    if a.dtype == np.dtype("B"):
        if a.ndim == 2:
            return Image.frombytes("L", (a.shape[1], a.shape[0]), a.tobytes())
        elif a.ndim == 3:
            return Image.frombytes("RGB", (a.shape[1], a.shape[0]), a.tobytes())
        else:
            raise Exception("bad image rank")
    elif a.dtype == np.dtype("float32"):
        return Image.frombytes("F", (a.shape[1], a.shape[0]), a.tobytes())
    else:
        raise Exception("unknown image type")


def is_bitonal(im: Union[Image.Image, torch.Tensor]) -> bool:
    """
    Tests a PIL image or torch tensor for bitonality.

    Args:
        im: Image to test

    Returns:
        True if the image contains only two different color values. False
        otherwise.
    """
    if isinstance(im, Image.Image):
        return im.getcolors(2) is not None and len(im.getcolors(2)) == 2
    elif isinstance(im, torch.Tensor):
        return len(im.int().unique()) == 2


def get_im_str(im: Image.Image) -> str:
    return im.filename if hasattr(im, "filename") else str(im)


def pil2array(im: Image.Image, alpha: int = 0) -> np.ndarray:
    if im.mode == "1":
        return np.array(im.convert("L"))
    return np.array(im)
