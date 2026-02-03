"""Sprite preprocessing and rotated atlas generation."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from sprite_recon.config.schema import LevelConfig
from sprite_recon.data import (
    SpriteBase,
    SpriteInput,
    SpriteLevel,
    SpritePyramid,
    SpriteRotation,
)

_ALPHA_THRESHOLD = 1e-3


def _validate_alpha(alpha: np.ndarray) -> np.ndarray:
    """Clamp alpha to [0, 1] and ensure it has finite values."""

    if not np.isfinite(alpha).all():
        raise ValueError("Alpha contains non-finite values.")
    return np.clip(alpha, 0.0, 1.0)


def _premultiply(rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RGBA to premultiplied-alpha representation."""

    alpha = _validate_alpha(rgba[..., 3])
    premultiplied = rgba.copy()
    premultiplied[..., :3] *= alpha[..., None]
    premultiplied[..., 3] = alpha
    return premultiplied, alpha


def _compute_bbox(alpha: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute bounding box of non-zero alpha region."""

    mask = alpha > _ALPHA_THRESHOLD
    if not np.any(mask):
        height, width = alpha.shape
        return (0, 0, width, height)
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _compute_norm(alpha: np.ndarray) -> float:
    """Compute L2 norm of alpha for later scoring normalization."""

    return float(np.sqrt(np.sum(alpha.astype(np.float64) ** 2)))


def _to_straight_alpha(premultiplied: np.ndarray) -> np.ndarray:
    """Convert premultiplied RGBA to straight-alpha RGBA."""

    alpha = np.clip(premultiplied[..., 3:4], 0.0, 1.0)
    safe_alpha = np.where(alpha <= 1e-6, 1.0, alpha)
    rgb = premultiplied[..., :3] / safe_alpha
    return np.concatenate([rgb, alpha], axis=-1)


def _to_premultiplied(straight: np.ndarray) -> np.ndarray:
    """Convert straight-alpha RGBA to premultiplied RGBA."""

    premultiplied = straight.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    return premultiplied


def _resize_linear(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize a linear-space premultiplied RGBA image using PIL."""

    height, width = image.shape[:2]
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    straight = _to_straight_alpha(image)
    pil_image = Image.fromarray((np.clip(straight, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    resized = pil_image.resize(new_size, resample=Image.BILINEAR)
    data = np.asarray(resized).astype(np.float32) / 255.0
    return _to_premultiplied(data)


def _rotate_image(premultiplied: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate premultiplied RGBA using PIL with expansion."""

    straight = _to_straight_alpha(premultiplied)
    pil_image = Image.fromarray((np.clip(straight, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    rotated = pil_image.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
    data = np.asarray(rotated).astype(np.float32) / 255.0
    return _to_premultiplied(data)


def _validate_premultiplied(premultiplied: np.ndarray, alpha: np.ndarray) -> None:
    """Ensure premultiplied RGB does not exceed alpha."""

    rgb = premultiplied[..., :3]
    if np.any(rgb > alpha[..., None] + 1e-4):
        raise ValueError("Premultiplied RGB exceeds alpha; check preprocessing.")


def _build_rotations(premultiplied: np.ndarray, alpha: np.ndarray, rotation_steps: int) -> List[SpriteRotation]:
    """Generate rotated variants for a sprite."""

    rotations: List[SpriteRotation] = []
    if rotation_steps <= 1:
        rotations.append(
            SpriteRotation(
                angle_deg=0.0,
                premultiplied=premultiplied,
                alpha=alpha,
                size=(premultiplied.shape[1], premultiplied.shape[0]),
                bbox=_compute_bbox(alpha),
                norm=_compute_norm(alpha),
            )
        )
        return rotations

    for step in range(rotation_steps):
        angle = (360.0 * step) / rotation_steps
        rotated = _rotate_image(premultiplied, angle)
        rotated_alpha = _validate_alpha(rotated[..., 3])
        _validate_premultiplied(rotated, rotated_alpha)
        rotations.append(
            SpriteRotation(
                angle_deg=angle,
                premultiplied=rotated,
                alpha=rotated_alpha,
                size=(rotated.shape[1], rotated.shape[0]),
                bbox=_compute_bbox(rotated_alpha),
                norm=_compute_norm(rotated_alpha),
            )
        )
    return rotations


def _build_base(sprite: SpriteInput) -> SpriteBase:
    """Build base premultiplied sprite with metadata."""

    premultiplied, alpha = _premultiply(sprite.rgba)
    _validate_premultiplied(premultiplied, alpha)
    height, width = premultiplied.shape[:2]
    return SpriteBase(
        name=sprite.name,
        premultiplied=premultiplied,
        alpha=alpha,
        size=(width, height),
        bbox=_compute_bbox(alpha),
        norm=_compute_norm(alpha),
        source_path=sprite.source_path,
    )


def preprocess_sprites(
    sprites: Iterable[SpriteInput],
    levels: List[LevelConfig],
) -> List[SpritePyramid]:
    """Preprocess sprites into per-level, rotated variants.

    Notes on memory usage:
        Rotated atlases multiply memory roughly by (rotation_steps) per level.
        This design favors predictable GPU uploads later at the cost of RAM.
    """

    pyramids: List[SpritePyramid] = []
    for sprite in sprites:
        base = _build_base(sprite)
        level_entries: List[SpriteLevel] = []
        for level_index, level in enumerate(levels):
            scaled = _resize_linear(base.premultiplied, level.scale)
            scaled_alpha = _validate_alpha(scaled[..., 3])
            _validate_premultiplied(scaled, scaled_alpha)
            rotations = _build_rotations(scaled, scaled_alpha, level.rotation_steps)
            level_entries.append(
                SpriteLevel(
                    level_index=level_index,
                    scale=level.scale,
                    premultiplied=scaled,
                    alpha=scaled_alpha,
                    size=(scaled.shape[1], scaled.shape[0]),
                    bbox=_compute_bbox(scaled_alpha),
                    norm=_compute_norm(scaled_alpha),
                    rotations=rotations,
                )
            )
        pyramids.append(SpritePyramid(base=base, levels=level_entries))
    return pyramids
