"""Sprite compositing utilities using premultiplied alpha."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from sprite_recon.data import SpritePyramid


def _resize_linear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a linear-space RGBA image using PIL."""

    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _rotate_image(premultiplied: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate premultiplied RGBA image using PIL with expansion."""

    pil_image = Image.fromarray((np.clip(premultiplied, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    rotated = pil_image.rotate(angle_deg, resample=Image.BILINEAR, expand=True)
    return np.asarray(rotated).astype(np.float32) / 255.0


def _translate_image(premultiplied: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Apply sub-pixel translation using an affine transform."""

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return premultiplied
    pil_image = Image.fromarray((np.clip(premultiplied, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    translated = pil_image.transform(
        pil_image.size,
        Image.AFFINE,
        (1.0, 0.0, dx, 0.0, 1.0, dy),
        resample=Image.BILINEAR,
    )
    return np.asarray(translated).astype(np.float32) / 255.0


def _apply_scale_aspect(premultiplied: np.ndarray, scale: float, aspect_ratio: float) -> np.ndarray:
    """Apply uniform scale and aspect ratio adjustments."""

    height, width = premultiplied.shape[:2]
    x_scale = scale * aspect_ratio
    y_scale = scale / aspect_ratio if aspect_ratio != 0 else scale
    new_width = max(1, int(width * x_scale))
    new_height = max(1, int(height * y_scale))
    return _resize_linear(premultiplied, new_width, new_height)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB in [0,1] to HSV (vectorized)."""

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0.0, delta / maxc)

    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    rc = ((maxc - r) / delta) * mask
    gc = ((maxc - g) / delta) * mask
    bc = ((maxc - b) / delta) * mask

    hue = np.where((maxc == r) & mask, (bc - gc), hue)
    hue = np.where((maxc == g) & mask, 2.0 + rc - bc, hue)
    hue = np.where((maxc == b) & mask, 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    return np.stack([hue, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV in [0,1] to RGB (vectorized)."""

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    out = np.zeros_like(hsv)
    conds = [i_mod == k for k in range(6)]
    out[..., 0] = np.select(conds, [v, q, p, p, t, v])
    out[..., 1] = np.select(conds, [t, v, v, q, p, p])
    out[..., 2] = np.select(conds, [p, p, t, v, v, q])
    return out


def _apply_hsv_adjust(premultiplied: np.ndarray, hsv_adjust: Tuple[float, float, float]) -> np.ndarray:
    """Apply HSV adjustments to a premultiplied sprite.

    HSV adjustment operates on unpremultiplied RGB to avoid skewing colors
    by alpha. Premultiplication is restored after adjustment.
    """

    adjusted = premultiplied.copy()
    alpha = adjusted[..., 3:4]
    safe_alpha = np.where(alpha <= 1e-6, 1.0, alpha)
    rgb = adjusted[..., :3] / safe_alpha

    hsv = _rgb_to_hsv(np.clip(rgb, 0.0, 1.0))
    hsv[..., 0] = (hsv[..., 0] + hsv_adjust[0]) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] + hsv_adjust[1], 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] + hsv_adjust[2], 0.0, 1.0)
    rgb_adjusted = _hsv_to_rgb(hsv)

    adjusted[..., :3] = rgb_adjusted * alpha
    return adjusted


def render_sprite(
    sprite: SpritePyramid,
    level_index: int,
    position: Tuple[float, float],
    rotation_deg: float,
    scale: float,
    aspect_ratio: float,
    hsv_adjust: Tuple[float, float, float],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Render a sprite to a local patch without blending."""

    level = sprite.levels[level_index]
    base_template = _apply_scale_aspect(level.premultiplied, scale, aspect_ratio)
    if abs(rotation_deg) > 1e-6:
        base_template = _rotate_image(base_template, rotation_deg)
    base_template = _apply_hsv_adjust(base_template, hsv_adjust)

    center = (int(round(position[0])), int(round(position[1])))
    base_template = _translate_image(
        base_template, position[0] - center[0], position[1] - center[1]
    )

    height, width = base_template.shape[:2]
    x0 = center[0] - width // 2
    y0 = center[1] - height // 2
    x1 = x0 + width
    y1 = y0 + height
    return base_template, (x0, y0, x1, y1)


def composite_sprite(
    canvas: np.ndarray,
    sprite: SpritePyramid,
    level_index: int,
    position: Tuple[float, float],
    rotation_deg: float,
    scale: float,
    aspect_ratio: float,
    hsv_adjust: Tuple[float, float, float],
    opacity: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Composite a sprite onto the canvas and return the affected bounds.

    Returns the updated canvas and bounding box (x0, y0, x1, y1).
    """

    base_template, (x0, y0, x1, y1) = render_sprite(
        sprite=sprite,
        level_index=level_index,
        position=position,
        rotation_deg=rotation_deg,
        scale=scale,
        aspect_ratio=aspect_ratio,
        hsv_adjust=hsv_adjust,
    )

    if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
        raise ValueError("Sprite placement is out of canvas bounds.")

    sprite_rgba = base_template.copy()
    sprite_rgba[..., :3] *= opacity
    sprite_rgba[..., 3] *= opacity

    canvas_patch = canvas[y0:y1, x0:x1]
    src_rgb = sprite_rgba[..., :3]
    src_a = sprite_rgba[..., 3:4]
    dst_rgb = canvas_patch[..., :3]
    dst_a = canvas_patch[..., 3:4]

    out_rgb = src_rgb + dst_rgb * (1.0 - src_a)
    out_a = src_a + dst_a * (1.0 - src_a)
    canvas_patch[..., :3] = out_rgb
    canvas_patch[..., 3:4] = out_a
    canvas[y0:y1, x0:x1] = canvas_patch

    return canvas, (x0, y0, x1, y1)
