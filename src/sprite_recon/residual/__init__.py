"""Residual handling utilities."""

from sprite_recon.residual.state import initialize_state
from sprite_recon.residual.update import recompute_residual, update_residual_patch

__all__ = ["initialize_state", "recompute_residual", "update_residual_patch"]
