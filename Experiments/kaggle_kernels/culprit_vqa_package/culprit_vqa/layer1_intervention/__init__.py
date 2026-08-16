"""Layer 1 — factorial intervention engine (proposal §6.1-§6.2).

Given a base VQA item and k <= 3 assigned perturbation factors, generates
the full 2^k lattice of perturbation conditions and validates each
condition with a three-way check (semantic / evidence / prior-shift).
"""

from culprit_vqa.layer1_intervention.items import (
    BBox,
    EvidenceRegion,
    Item,
    PerturbationEffect,
)
from culprit_vqa.layer1_intervention.lattice import Condition, generate_lattice
from culprit_vqa.layer1_intervention.validity import ValidityReport, run_validity_checks

__all__ = [
    "BBox",
    "EvidenceRegion",
    "PerturbationEffect",
    "Item",
    "Condition",
    "generate_lattice",
    "ValidityReport",
    "run_validity_checks",
]
