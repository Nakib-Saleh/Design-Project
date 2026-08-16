"""Three-way perturbation validity check (proposal §6.2).

Every generated condition must pass semantic preservation and evidence
preservation to be *accepted*. Prior-shift is measured and recorded as a
covariate but never gates acceptance — "semantically irrelevant but
statistically informative" perturbations are one of the phenomena under
study, not a reason to discard data.
"""

from dataclasses import dataclass

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition
from culprit_vqa.layer1_intervention.validity.evidence import (
    EvidenceCheckFn,
    evidence_preservation_check,
)
from culprit_vqa.layer1_intervention.validity.prior_shift import (
    PriorFn,
    prior_shift_measurement,
    stub_text_only_prior,
)
from culprit_vqa.layer1_intervention.validity.semantic import (
    SemanticCheckFn,
    semantic_preservation_check,
)


@dataclass(frozen=True)
class ValidityReport:
    condition_id: str
    semantic_ok: bool
    evidence_ok: bool
    prior_shift: float
    accepted: bool


def run_validity_checks(
    item: Item,
    condition: Condition,
    semantic_fn: SemanticCheckFn = semantic_preservation_check,
    evidence_fn: EvidenceCheckFn = evidence_preservation_check,
    prior_fn: PriorFn = stub_text_only_prior,
) -> ValidityReport:
    """Run the three-way check on one condition.

    `accepted` is semantic_ok AND evidence_ok only — prior_shift is
    recorded as a covariate and never affects acceptance (proposal §6.2).
    """
    semantic_result = semantic_fn(item, condition)
    evidence_result = evidence_fn(item, condition)
    shift = prior_shift_measurement(item, condition, prior_fn=prior_fn)
    return ValidityReport(
        condition_id=condition.condition_id,
        semantic_ok=semantic_result,
        evidence_ok=evidence_result,
        prior_shift=shift,
        accepted=semantic_result and evidence_result,
    )


__all__ = [
    "ValidityReport",
    "run_validity_checks",
    "semantic_preservation_check",
    "evidence_preservation_check",
    "prior_shift_measurement",
    "stub_text_only_prior",
    "SemanticCheckFn",
    "EvidenceCheckFn",
    "PriorFn",
]
