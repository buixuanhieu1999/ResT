"""Shadow-robust signature extraction for phone photos.

The model's own preprocessing (``train_cedar.SignatureTransform``) applies one
global Otsu threshold, which is fine for flat GPDS scans but turns shadows and
lighting gradients in phone photos into "ink". This module flattens the
illumination first, so only real strokes survive, and returns a clean
black-on-white image that is then handed to SignatureTransform unchanged.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

MAX_LONG_SIDE = 1600
INK_COLOURS = {
    "black": (0, 0, 0),
    "blue": (16, 42, 150),
}


class CleanResult(NamedTuple):
    rgb: np.ndarray  # HxWx3 uint8, ink on white (this is what the model sees)
    rgba: np.ndarray  # HxWx4 uint8, ink on transparent (for download)
    found_ink: bool


def _odd(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Accept gray, RGB or RGBA (any dtype Gradio might send) and return RGB uint8."""

    image = np.asarray(image)
    if image.dtype != np.uint8:
        scale = 255.0 if image.dtype.kind == "f" and image.max() <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        # Composite transparent pixels onto white so they don't read as ink.
        alpha = image[..., 3:4].astype(np.float32) / 255.0
        rgb = image[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return rgb.astype(np.uint8)
    return image[..., :3]


def _downscale(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    long_side = max(height, width)
    if long_side <= MAX_LONG_SIDE:
        return rgb
    scale = MAX_LONG_SIDE / long_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def darkness_map(gray: np.ndarray) -> np.ndarray:
    """Per-pixel ink darkness relative to the locally estimated paper brightness."""

    long_side = max(gray.shape)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(long_side / 90),) * 2
    )
    # Dilation erases thin dark strokes; the median smooths what is left into a
    # paper/shadow illumination estimate.
    background = cv2.dilate(gray, kernel)
    background = cv2.medianBlur(background, min(_odd(long_side / 25), 255))
    background = np.maximum(background.astype(np.float32), 1.0)
    return np.clip(1.0 - gray.astype(np.float32) / background, 0.0, 1.0)


def _ink_threshold(darkness: np.ndarray, sensitivity: float) -> float:
    candidates = darkness[darkness > 0.04]
    if candidates.size < 50:
        threshold = 0.25
    else:
        values = np.round(candidates * 255.0).astype(np.uint8).reshape(-1, 1)
        otsu, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(otsu) / 255.0
    return max(threshold, 0.10) / max(sensitivity, 1e-3)


def _keep_strokes(
    mask: np.ndarray, min_speck: float, remove_border_blobs: bool
) -> np.ndarray:
    height, width = mask.shape
    long_side = max(height, width)
    min_area = min_speck * (long_side / 1000.0) ** 2 * 40.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        if remove_border_blobs:
            touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
            oversized = w > 0.5 * width or h > 0.5 * height or area > 0.02 * width * height
            if touches_border and oversized:
                continue
        keep[index] = True
    return keep[labels]


def clean_signature(
    image: np.ndarray,
    sensitivity: float = 1.0,
    min_speck: float = 1.0,
    remove_border_blobs: bool = True,
    ink_colour: str = "black",
) -> CleanResult:
    """Extract the signature strokes from a photo and auto-crop them.

    ``sensitivity`` > 1 keeps fainter strokes; < 1 is stricter. ``min_speck``
    scales the smallest blob (in pixels at 1000 px long side, x40) kept.
    """

    rgb = _downscale(_to_rgb(image))
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    darkness = darkness_map(gray)
    threshold = _ink_threshold(darkness, sensitivity)
    strokes = _keep_strokes(darkness > threshold, min_speck, remove_border_blobs)

    height, width = strokes.shape
    if not strokes.any():
        white = np.full((height, width, 3), 255, dtype=np.uint8)
        clear = np.zeros((height, width, 4), dtype=np.uint8)
        return CleanResult(white, clear, False)

    low, high = 0.6 * threshold, 1.6 * threshold
    alpha = np.clip((darkness - low) / (high - low), 0.0, 1.0)
    support = cv2.dilate(strokes.astype(np.uint8), np.ones((3, 3), np.uint8))
    alpha *= support

    ys, xs = np.nonzero(strokes)
    pad = int(round(0.04 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    top, bottom = max(0, ys.min() - pad), min(height, ys.max() + pad + 1)
    left, right = max(0, xs.min() - pad), min(width, xs.max() + pad + 1)
    alpha = alpha[top:bottom, left:right]

    colour = np.array(INK_COLOURS.get(ink_colour, INK_COLOURS["black"]), np.float32)
    a = alpha[..., None]
    white_rgb = (255.0 * (1.0 - a) + colour * a).round().astype(np.uint8)
    rgba = np.empty((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., :3] = colour.astype(np.uint8)
    rgba[..., 3] = (alpha * 255.0).round().astype(np.uint8)
    return CleanResult(white_rgb, rgba, True)
