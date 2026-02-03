"""Local refinement of top-K candidate sprite placements.

Refinement operates on local patches only. It evaluates small, deterministic
perturbations of geometry and optionally brightness/HSV to improve fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from sprite_recon.config.schema import Config, LevelConfig
from sprite_recon.data import SpritePyramid
from sprite_recon.gpu_scoring import CandidateResult


@dataclass(frozen=True)
class RefinedCandidate:
    """Refined candidate with continuous parameters and measured improvement."""

    sprite_id: int
    level_index: int
    rotation_deg: float
    scale: float
    aspect_ratio: float
    hsv_adjust: Tuple[float, float, float]
    position: Tuple[float, float]
    opacity: float
    improvement: float


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


def _extract_patch(residual: np.ndarray, center: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    """Extract a residual patch centered at integer coordinates."""

    width, height = size
    half_w = width // 2
    half_h = height // 2
    x0 = center[0] - half_w
    y0 = center[1] - half_h
    x1 = x0 + width
    y1 = y0 + height
    return residual[y0:y1, x0:x1]


def _fit_opacity(residual_patch: np.ndarray, template: np.ndarray) -> Tuple[float, float]:
    """Compute optimal opacity and improvement for a fixed template.

    Uses least-squares: c* = <R, S> / <S, S>. Improvement is (dot^2)/denom.
    """

    residual_rgb = residual_patch[..., :3]
    template_rgb = template[..., :3]
    dot = float(np.sum(residual_rgb * template_rgb))
    denom = float(np.sum(template_rgb * template_rgb))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0, 0.0
    opacity = max(0.0, min(1.0, dot / denom))
    improvement = (dot * dot) / denom
    if not np.isfinite(improvement) or improvement < 0.0:
        return opacity, 0.0
    return opacity, improvement


def _apply_value_adjust(template: np.ndarray, value_scale: float) -> np.ndarray:
    """Apply a simple value/brightness adjustment to premultiplied RGB."""

    adjusted = template.copy()
    adjusted[..., :3] = np.clip(adjusted[..., :3] * value_scale, 0.0, 1.0)
    return adjusted


def _refine_candidate(
    candidate: CandidateResult,
    residual: np.ndarray,
    sprite_pyramids: Sequence[SpritePyramid],
    level_config: LevelConfig,
    config: Config,
    max_iters: int,
) -> RefinedCandidate:
    """Refine a single candidate with local, deterministic search."""

    descriptor = candidate.candidate
    sprite = sprite_pyramids[descriptor.sprite_id]
    level = sprite.levels[descriptor.level_index]
    rotation = level.rotations[descriptor.rotation_id]
    base_scale = level_config.uniform_scales[descriptor.scale_id]
    base_aspect = level_config.aspect_ratios[descriptor.aspect_ratio_id]
    base_rotation = rotation.angle_deg
    base_hsv = config.hsv_presets[descriptor.hsv_id]

    position = (float(descriptor.x), float(descriptor.y))
    scale = base_scale
    aspect = base_aspect
    rotation_deg = base_rotation
    hsv_adjust = base_hsv

    best_opacity = 0.0
    best_improvement = 0.0

    for _ in range(max_iters):
        improved = False
        for dx in (-0.5, 0.0, 0.5):
            for dy in (-0.5, 0.0, 0.5):
                test_position = (position[0] + dx, position[1] + dy)
                center = (int(round(test_position[0])), int(round(test_position[1])))
                base_template = _apply_scale_aspect(rotation.premultiplied, scale, aspect)
                if abs(rotation_deg - base_rotation) > 1e-6:
                    base_template = _rotate_image(base_template, rotation_deg - base_rotation)
                template = _translate_image(base_template, test_position[0] - center[0], test_position[1] - center[1])

                height, width = template.shape[:2]
                if (
                    center[0] - width // 2 < 0
                    or center[1] - height // 2 < 0
                    or center[0] + (width - width // 2) > residual.shape[1]
                    or center[1] + (height - height // 2) > residual.shape[0]
                ):
                    continue

                patch = _extract_patch(residual, center, (width, height))
                opacity, improvement = _fit_opacity(patch, template)

                if config.enable_hsv_refine:
                    for delta in (-config.hsv_refine_step, 0.0, config.hsv_refine_step):
                        value_scale = max(0.0, 1.0 + hsv_adjust[2] + delta)
                        adjusted = _apply_value_adjust(template, value_scale)
                        test_opacity, test_improvement = _fit_opacity(patch, adjusted)
                        if test_improvement > improvement:
                            improvement = test_improvement
                            opacity = test_opacity
                            hsv_adjust = (hsv_adjust[0], hsv_adjust[1], hsv_adjust[2] + delta)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_opacity = opacity
                    position = test_position
                    improved = True

        for delta_rot in (-1.0, 0.0, 1.0):
            test_rotation = rotation_deg + delta_rot
            base_template = _apply_scale_aspect(rotation.premultiplied, scale, aspect)
            template = _rotate_image(base_template, test_rotation - base_rotation)
            center = (int(round(position[0])), int(round(position[1])))
            height, width = template.shape[:2]
            if (
                center[0] - width // 2 < 0
                or center[1] - height // 2 < 0
                or center[0] + (width - width // 2) > residual.shape[1]
                or center[1] + (height - height // 2) > residual.shape[0]
            ):
                continue
            patch = _extract_patch(residual, center, (width, height))
            opacity, improvement = _fit_opacity(patch, template)
            if improvement > best_improvement:
                best_improvement = improvement
                best_opacity = opacity
                rotation_deg = test_rotation
                improved = True

        for delta_scale in (-0.05, 0.0, 0.05):
            test_scale = max(0.01, scale + delta_scale)
            base_template = _apply_scale_aspect(rotation.premultiplied, test_scale, aspect)
            template = _rotate_image(base_template, rotation_deg - base_rotation)
            center = (int(round(position[0])), int(round(position[1])))
            height, width = template.shape[:2]
            if (
                center[0] - width // 2 < 0
                or center[1] - height // 2 < 0
                or center[0] + (width - width // 2) > residual.shape[1]
                or center[1] + (height - height // 2) > residual.shape[0]
            ):
                continue
            patch = _extract_patch(residual, center, (width, height))
            opacity, improvement = _fit_opacity(patch, template)
            if improvement > best_improvement:
                best_improvement = improvement
                best_opacity = opacity
                scale = test_scale
                improved = True

        for delta_aspect in (-0.05, 0.0, 0.05):
            test_aspect = max(0.1, aspect + delta_aspect)
            base_template = _apply_scale_aspect(rotation.premultiplied, scale, test_aspect)
            template = _rotate_image(base_template, rotation_deg - base_rotation)
            center = (int(round(position[0])), int(round(position[1])))
            height, width = template.shape[:2]
            if (
                center[0] - width // 2 < 0
                or center[1] - height // 2 < 0
                or center[0] + (width - width // 2) > residual.shape[1]
                or center[1] + (height - height // 2) > residual.shape[0]
            ):
                continue
            patch = _extract_patch(residual, center, (width, height))
            opacity, improvement = _fit_opacity(patch, template)
            if improvement > best_improvement:
                best_improvement = improvement
                best_opacity = opacity
                aspect = test_aspect
                improved = True

        if not improved:
            break

    return RefinedCandidate(
        sprite_id=descriptor.sprite_id,
        level_index=descriptor.level_index,
        rotation_deg=rotation_deg,
        scale=scale,
        aspect_ratio=aspect,
        hsv_adjust=hsv_adjust,
        position=position,
        opacity=best_opacity,
        improvement=best_improvement,
    )


def refine_candidates(
    candidates: Iterable[CandidateResult],
    residual: np.ndarray,
    sprite_pyramids: Sequence[SpritePyramid],
    level_config: LevelConfig,
    config: Config,
    max_iters: int,
) -> List[RefinedCandidate]:
    """Refine the top-K candidates with local continuous search.

    The work scales with sprite area due to patch-based evaluation.
    """

    if not np.isfinite(residual).all():
        raise ValueError("Residual contains non-finite values.")

    refined: List[RefinedCandidate] = []
    for candidate in candidates:
        refined.append(
            _refine_candidate(
                candidate=candidate,
                residual=residual,
                sprite_pyramids=sprite_pyramids,
                level_config=level_config,
                config=config,
                max_iters=max_iters,
            )
        )
    return refined
