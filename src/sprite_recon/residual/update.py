"""Residual update utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def recompute_residual(target: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Recompute the full residual from target and canvas."""

    if target.shape != canvas.shape:
        raise ValueError("Target and canvas shapes must match.")
    return target - canvas


def update_residual_patch(
    residual: np.ndarray,
    target: np.ndarray,
    canvas: np.ndarray,
    bounds: Tuple[int, int, int, int],
    sanity_check: bool = False,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Update residual for a bounding box region.

    If sanity_check is True, recompute full residual and validate bounds.
    """

    x0, y0, x1, y1 = bounds
    residual[y0:y1, x0:x1] = target[y0:y1, x0:x1] - canvas[y0:y1, x0:x1]

    if sanity_check:
        full = recompute_residual(target, canvas)
        diff = np.max(np.abs(full - residual))
        if diff > tolerance:
            raise ValueError(f"Residual patch update drifted (max diff {diff}).")
    return residual
