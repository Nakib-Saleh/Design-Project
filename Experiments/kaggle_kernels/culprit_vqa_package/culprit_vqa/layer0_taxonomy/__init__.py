"""Layer 0 — the formal 5-axis factor taxonomy (proposal §4).

This is the formal object the rest of the pipeline is built on: a
perturbation or natural-failure hypothesis is a point in a 5-axis factor
space (Modality x Relevance x Conflict x Culturality x Knowledge). The
legacy 4-pillar tree is retained only as a presentation/mapping view
(see `pillar_mapping.py`) — claims and measurements are made at the
factor level, never the pillar level.
"""

from culprit_vqa.layer0_taxonomy.axes import (
    Conflict,
    Culturality,
    Knowledge,
    Modality,
    Relevance,
)
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates
from culprit_vqa.layer0_taxonomy.operators import (
    OPERATOR_REGISTRY,
    get_operator,
    list_operators,
)
from culprit_vqa.layer0_taxonomy.pillar_mapping import (
    LEAF_TO_FACTOR_TABLE,
    Pillar,
    pillar_for_factor,
)

__all__ = [
    "Modality",
    "Relevance",
    "Conflict",
    "Culturality",
    "Knowledge",
    "FactorCoordinates",
    "Factor",
    "OPERATOR_REGISTRY",
    "get_operator",
    "list_operators",
    "Pillar",
    "pillar_for_factor",
    "LEAF_TO_FACTOR_TABLE",
]
