"""RGB manuscript line segmentation adapted from datasets-master/page_segmentation.

Detection uses the green channel, projection peaks, and curved whitespace seams.
Coordinates are half-open boxes in the unchanged source image, as used by Logios.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


MAX_PIXELS = 24_000_000


@dataclass
class LineCrop:
    bbox: list[int]
    image: Image.Image


def ink_mask(rgb):
    green = rgb[:, :, 1]
    block = max(3, min(81, (min(green.shape) // 2) * 2 - 1))
    binary = cv2.adaptiveThreshold(
        green, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 16
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    return keep[labels]


def detect_centers(mask):
    h, w = mask.shape
    inset = int(w * 0.12)
    profile = mask[:, inset:w - inset].mean(axis=1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    usable = heights[(areas >= 12) & (heights >= 4) & (heights < h * 0.25)]
    glyph_h = float(np.median(usable)) if len(usable) else max(8, h / 20)
    distance = max(8, round(glyph_h * 1.35))
    smooth = gaussian_filter1d(profile, max(1, glyph_h * 0.12))
    if smooth.max() < 0.008:
        return []
    peaks, _ = find_peaks(
        np.pad(smooth, 1),
        distance=distance,
        prominence=max(0.008, smooth.max() * 0.12),
        height=max(0.012, smooth.max() * 0.14),
    )
    return (peaks - 1).tolist()


def whitespace_seam(mask, upper, lower):
    """Minimum-cost left-to-right path avoiding ink and excessive bending."""
    h, w = mask.shape
    gap = lower - upper
    lo = max(0, int(upper + gap * 0.08))
    hi = min(h - 1, int(lower - gap * 0.08))
    if hi <= lo:
        return np.full(w, (upper + lower) // 2, dtype=int)
    dilated = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    ys = np.arange(lo, hi + 1)
    mid = (upper + lower) / 2
    costs = (
        12 * dilated[lo:hi + 1]
        + 0.08 * ((ys[:, None] - mid) / max(gap, 1)) ** 2
    )
    previous = costs[:, 0].copy()
    back = np.zeros(costs.shape, np.int8)
    for x in range(1, w):
        candidates = np.stack((
            np.r_[np.inf, previous[:-1]] + 0.08,
            previous,
            np.r_[previous[1:], np.inf] + 0.08,
        ))
        choice = candidates.argmin(axis=0)
        back[:, x] = choice - 1
        previous = costs[:, x] + candidates[choice, np.arange(len(ys))]
    y = int(previous.argmin())
    seam = np.empty(w, dtype=int)
    for x in range(w - 1, -1, -1):
        seam[x] = y + lo
        y += int(back[y, x])
    return seam


def segment_lines(image: Image.Image) -> list[LineCrop]:
    """Treat the selected page/crop as one text region, in top-to-bottom order."""
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 24 megapixels; resize it before OCR.")
    rgb = np.asarray(image.convert("RGB"))
    ink = ink_mask(rgb)
    centers = detect_centers(ink)
    if not centers:
        return []
    h, w = ink.shape
    seams = [np.zeros(w, dtype=int)]
    seams += [whitespace_seam(ink, a, b) for a, b in zip(centers, centers[1:])]
    seams += [np.full(w, h, dtype=int)]
    background = (
        np.median(rgb[ink == 0], axis=0)
        if np.any(ink == 0) else np.array([255] * 3)
    ).astype(np.uint8)
    padding = 5
    lines = []
    for top, bottom in zip(seams, seams[1:]):
        yy = np.arange(h)[:, None]
        inside = (yy >= top) & (yy < bottom)
        ys, xs = np.where((ink > 0) & inside)
        if not len(xs):
            continue
        x0, x1 = max(0, int(xs.min()) - padding), min(w, int(xs.max()) + padding + 1)
        y0, y1 = max(0, int(ys.min()) - padding), min(h, int(ys.max()) + padding + 1)
        crop = rgb[y0:y1, x0:x1].copy()
        crop[~inside[y0:y1, x0:x1]] = background
        # The extra border is synthetic; it must not shift the source bbox.
        padded = ImageOps.expand(
            Image.fromarray(crop), border=padding,
            fill=tuple(int(value) for value in background),
        )
        lines.append(LineCrop(bbox=[x0, y0, x1, y1], image=padded))
    return lines
