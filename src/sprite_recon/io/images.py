"""Image loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from sprite_recon.config.schema import Config
from sprite_recon.data import SpriteInput

_SRGB_THRESHOLD = 0.0031308


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    """Convert sRGB channel values (0..1) to linear space."""

    return np.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(channel: np.ndarray) -> np.ndarray:
    """Convert linear channel values (0..1) to sRGB."""

    return np.where(
        channel <= _SRGB_THRESHOLD,
        channel * 12.92,
        1.055 * np.power(np.clip(channel, 0.0, 1.0), 1.0 / 2.4) - 0.055,
    )


def _image_to_linear_rgba(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to linear RGBA float32 array."""

    rgba = image.convert("RGBA")
    data = np.asarray(rgba).astype(np.float32) / 255.0
    rgb = _srgb_to_linear(data[..., :3])
    alpha = data[..., 3:4]
    return np.concatenate([rgb, alpha], axis=-1)


def load_target_image(path: Path) -> np.ndarray:
    """Load target image and convert to linear RGBA."""

    image = Image.open(path)
    return _image_to_linear_rgba(image)


def save_image(path: Path, linear_rgba: np.ndarray) -> None:
    """Save a linear RGBA image as sRGB PNG."""

    rgb = _linear_to_srgb(linear_rgba[..., :3])
    alpha = np.clip(linear_rgba[..., 3:4], 0.0, 1.0)
    srgb = np.concatenate([rgb, alpha], axis=-1)
    srgb_uint8 = (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(srgb_uint8, mode="RGBA").save(path)


def _iter_sprite_paths(config: Config) -> Iterable[Path]:
    """Yield sprite paths from config in a deterministic order."""

    if config.sprite_paths:
        for sprite_path in config.sprite_paths:
            yield sprite_path
    if config.sprite_dir:
        for path in sorted(config.sprite_dir.iterdir()):
            if path.suffix.lower() in config.sprite_extensions:
                yield path


def load_sprite_images(config: Config) -> List[SpriteInput]:
    """Load sprite images from configured sources."""

    sprites: List[SpriteInput] = []
    for path in _iter_sprite_paths(config):
        image = Image.open(path)
        linear = _image_to_linear_rgba(image)
        sprites.append(SpriteInput(name=path.stem, rgba=linear, source_path=path))
    return sprites
