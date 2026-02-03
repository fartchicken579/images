"""Builds multiresolution pyramids for images."""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


def _resize_linear(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize a linear-space RGBA image using PIL for convenience."""

    height, width = image.shape[:2]
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    resized = pil_image.resize(new_size, resample=Image.BILINEAR)
    data = np.asarray(resized).astype(np.float32) / 255.0
    return data


def build_pyramid(image: np.ndarray, min_size: int, max_levels: int) -> List[np.ndarray]:
    """Build a coarse-to-fine pyramid from a linear RGBA image."""

    pyramid = [image]
    current = image
    for _ in range(max_levels - 1):
        height, width = current.shape[:2]
        if min(height, width) <= min_size:
            break
        current = _resize_linear(current, 0.5)
        pyramid.append(current)
    return list(reversed(pyramid))
