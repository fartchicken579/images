"""Core data structures used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpriteInput:
    """Raw sprite input in linear RGBA (not premultiplied)."""

    name: str
    rgba: np.ndarray
    source_path: Path


@dataclass
class SpriteBase:
    """Preprocessed sprite stored in premultiplied-alpha form."""

    name: str
    premultiplied: np.ndarray
    alpha: np.ndarray
    size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    norm: float
    source_path: Path


@dataclass
class SpriteRotation:
    """Rotated sprite variant for a specific angle."""

    angle_deg: float
    premultiplied: np.ndarray
    alpha: np.ndarray
    size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    norm: float


@dataclass
class SpriteLevel:
    """Scaled sprite and its rotated variants for a pyramid level."""

    level_index: int
    scale: float
    premultiplied: np.ndarray
    alpha: np.ndarray
    size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    norm: float
    rotations: List[SpriteRotation]


@dataclass
class SpritePyramid:
    """All per-level variants for a sprite, suitable for GPU upload later."""

    base: SpriteBase
    levels: List[SpriteLevel]


@dataclass
class SpritePlacement:
    """Parameters describing a placed sprite instance."""

    sprite_index: int
    position: Tuple[float, float]
    rotation: float
    scale: float
    aspect_ratio: float
    hsv_adjust: Tuple[float, float, float]
    opacity: float


@dataclass
class Canvas:
    """Canvas holding the current composite at a given pyramid level."""

    level: int
    image: np.ndarray


@dataclass
class Residual:
    """Residual image (target - canvas) at a given pyramid level."""

    level: int
    image: np.ndarray


@dataclass
class ReconstructionState:
    """Container for the current state of the reconstruction process."""

    canvas: Canvas
    residual: Residual
    level_scale: float
    sprite_count: int = 0
    last_gain: float = 0.0
    last_gain_rate: Optional[float] = None
