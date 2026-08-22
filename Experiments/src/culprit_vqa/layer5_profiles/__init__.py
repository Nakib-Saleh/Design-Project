"""Layer 5 — failure profiles (proposal §6.7).

Per-language/culture/model causal-factor mixture histograms and an
interaction atlas — minimal dict-based aggregation for this scaffold;
richer aggregation/visualization is future thesis work.
"""

from culprit_vqa.layer5_profiles.aggregate import FailureProfile, build_failure_profile
from culprit_vqa.layer5_profiles.interaction_atlas import build_interaction_atlas

__all__ = ["FailureProfile", "build_failure_profile", "build_interaction_atlas"]
