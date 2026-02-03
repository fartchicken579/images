"""Candidate generation and scoring.

This module is structured for GPU execution, with a CPU fallback used now.
The CPU path mirrors the expected GPU kernel behavior to keep interfaces stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from sprite_recon.config.schema import LevelConfig
from sprite_recon.data import SpritePyramid


@dataclass(frozen=True)
class CandidateDescriptor:
    """Discrete candidate parameters for scoring."""

    sprite_id: int
    level_index: int
    rotation_id: int
    scale_id: int
    aspect_ratio_id: int
    hsv_id: int
    x: int
    y: int


@dataclass(frozen=True)
class CandidateResult:
    """Score for a candidate, returned to the CPU after GPU scoring."""

    candidate: CandidateDescriptor
    score: float


@dataclass
class CandidateScorer:
    """Interface for candidate scoring backends."""

    backend: str = "cpu"

    def score(
        self,
        residual: np.ndarray,
        sprite_pyramids: Sequence[SpritePyramid],
        level_index: int,
        level_config: LevelConfig,
        hsv_presets: Sequence[Tuple[float, float, float]],
        top_k: int,
    ) -> List[CandidateResult]:
        """Score candidates and return the top-K results."""

        if self.backend != "cpu":
            raise NotImplementedError("GPU backend is not implemented yet.")
        return score_candidates(residual, sprite_pyramids, level_index, level_config, hsv_presets, top_k)


def _resize_linear(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a linear-space RGBA image using PIL."""

    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _apply_scale_aspect(
    premultiplied: np.ndarray,
    scale: float,
    aspect_ratio: float,
) -> np.ndarray:
    """Apply uniform scale and aspect ratio adjustments to a sprite."""

    height, width = premultiplied.shape[:2]
    x_scale = scale * aspect_ratio
    y_scale = scale / aspect_ratio if aspect_ratio != 0 else scale
    new_width = max(1, int(width * x_scale))
    new_height = max(1, int(height * y_scale))
    return _resize_linear(premultiplied, new_width, new_height)


def _extract_patch(residual: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Extract a residual patch centered at (x, y)."""

    half_w = width // 2
    half_h = height // 2
    x0 = x - half_w
    y0 = y - half_h
    x1 = x0 + width
    y1 = y0 + height
    return residual[y0:y1, x0:x1]


def _score_patch(residual_patch: np.ndarray, template: np.ndarray) -> float:
    """Compute upper-bound L2 improvement score for a candidate."""

    residual_rgb = residual_patch[..., :3]
    template_rgb = template[..., :3]
    dot = float(np.sum(residual_rgb * template_rgb))
    denom = float(np.sum(template_rgb * template_rgb))
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    score = (dot * dot) / denom
    if not np.isfinite(score) or score < 0.0:
        return 0.0
    return score


def _iter_grid_positions(residual: np.ndarray, width: int, height: int, stride: int) -> Iterable[Tuple[int, int]]:
    """Yield candidate centers that keep the sprite fully in bounds."""

    if stride <= 0:
        raise ValueError("Grid stride must be positive.")
    max_y, max_x = residual.shape[:2]
    half_w = width // 2
    half_h = height // 2
    x_start = half_w
    y_start = half_h
    x_end = max_x - (width - half_w)
    y_end = max_y - (height - half_h)
    for y in range(y_start, y_end, stride):
        for x in range(x_start, x_end, stride):
            yield x, y


def _generate_candidates(
    residual: np.ndarray,
    sprite_pyramids: Sequence[SpritePyramid],
    level_index: int,
    level_config: LevelConfig,
    hsv_presets: Sequence[Tuple[float, float, float]],
) -> Iterable[CandidateDescriptor]:
    """Yield discrete candidate parameter sets for a given pyramid level."""

    for sprite_id, sprite in enumerate(sprite_pyramids):
        if level_index >= len(sprite.levels):
            raise ValueError("Level index exceeds available sprite pyramid levels.")
        level = sprite.levels[level_index]
        for rotation_id, rotation in enumerate(level.rotations):
            for scale_id, _ in enumerate(level_config.uniform_scales):
                for aspect_id, _ in enumerate(level_config.aspect_ratios):
                    for hsv_id, _ in enumerate(hsv_presets):
                        for x, y in _iter_grid_positions(
                            residual,
                            rotation.size[0],
                            rotation.size[1],
                            level_config.grid_stride,
                        ):
                            yield CandidateDescriptor(
                                sprite_id=sprite_id,
                                level_index=level.level_index,
                                rotation_id=rotation_id,
                                scale_id=scale_id,
                                aspect_ratio_id=aspect_id,
                                hsv_id=hsv_id,
                                x=x,
                                y=y,
                            )


def score_candidates(
    residual: np.ndarray,
    sprite_pyramids: Sequence[SpritePyramid],
    level_index: int,
    level_config: LevelConfig,
    hsv_presets: Sequence[Tuple[float, float, float]],
    top_k: int,
) -> List[CandidateResult]:
    """Score candidates and return the top-K results.

    This CPU implementation mirrors the intended GPU kernel:
      - Each candidate computes a single scalar score.
      - Scoring depends only on residual and sprite templates.
      - No canvas or residual mutations occur.
      - GPU kernels will read residual and sprite atlases with coalesced access.
    """

    if residual.ndim != 3 or residual.shape[-1] < 3:
        raise ValueError("Residual must be HxWxC with at least 3 channels.")
    if not np.isfinite(residual).all():
        raise ValueError("Residual contains non-finite values.")

    results: List[CandidateResult] = []
    for candidate in _generate_candidates(residual, sprite_pyramids, level_index, level_config, hsv_presets):
        sprite = sprite_pyramids[candidate.sprite_id]
        level = sprite.levels[candidate.level_index]
        rotation = level.rotations[candidate.rotation_id]
        scale = level_config.uniform_scales[candidate.scale_id]
        aspect = level_config.aspect_ratios[candidate.aspect_ratio_id]

        # HSV presets are included in the descriptor for later use; scoring ignores them for now.
        template = _apply_scale_aspect(rotation.premultiplied, scale, aspect)
        height, width = template.shape[:2]

        if (
            candidate.x - width // 2 < 0
            or candidate.y - height // 2 < 0
            or candidate.x + (width - width // 2) > residual.shape[1]
            or candidate.y + (height - height // 2) > residual.shape[0]
        ):
            continue

        patch = _extract_patch(residual, candidate.x, candidate.y, width, height)
        score = _score_patch(patch, template)
        results.append(CandidateResult(candidate=candidate, score=score))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[: max(0, top_k)]
