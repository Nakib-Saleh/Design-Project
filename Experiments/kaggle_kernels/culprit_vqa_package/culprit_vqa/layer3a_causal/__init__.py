"""Layer 3a — causal ground truth (proposal §6.4, the centerpiece).

Exact Shapley attribution over the 2^k perturbation lattice, plus the
leave-one-out baseline and pairwise interaction indices it is designed to
beat under interacting factors (RQ1).
"""

from culprit_vqa.layer3a_causal.attribution import (
    efficiency_residual,
    is_null_attribution,
    normalize_attribution,
)
from culprit_vqa.layer3a_causal.probability import build_p_function, estimate_p
from culprit_vqa.layer3a_causal.shapley import (
    interaction_indices,
    leave_one_out,
    shapley_values,
)

__all__ = [
    "estimate_p",
    "build_p_function",
    "shapley_values",
    "leave_one_out",
    "interaction_indices",
    "normalize_attribution",
    "is_null_attribution",
    "efficiency_residual",
]
