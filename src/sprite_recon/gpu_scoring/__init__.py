"""GPU scoring interface and CPU fallback implementation."""

from sprite_recon.gpu_scoring.scorer import (
    CandidateDescriptor,
    CandidateResult,
    CandidateScorer,
    score_candidates,
)

__all__ = [
    "CandidateDescriptor",
    "CandidateResult",
    "CandidateScorer",
    "score_candidates",
]
