"""Initialize canvas and residual structures."""

from __future__ import annotations

import numpy as np

from sprite_recon.data import Canvas, ReconstructionState, Residual


def initialize_state(target_level: np.ndarray, level_scale: float) -> ReconstructionState:
    """Initialize canvas and residual for a pyramid level."""

    canvas_image = np.zeros_like(target_level)
    residual_image = target_level - canvas_image
    canvas = Canvas(level=0, image=canvas_image)
    residual = Residual(level=0, image=residual_image)
    return ReconstructionState(canvas=canvas, residual=residual, level_scale=level_scale)
