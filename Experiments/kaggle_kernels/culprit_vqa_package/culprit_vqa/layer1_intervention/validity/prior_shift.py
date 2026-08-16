"""Prior-shift measurement (proposal §6.2, check 3 of 3).

The change in the text-only answer prior induced by a perturbation is
measured on the language backbone and recorded as a covariate — it is
NEVER grounds for rejecting a condition. "Semantically irrelevant but
statistically informative" perturbations are precisely one of the
phenomena CULPRIT-VQA studies, not something to filter out.
"""

import hashlib
from typing import Callable

from culprit_vqa.layer0_taxonomy.axes import Conflict, Culturality, Modality
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition, make_condition_id

PriorFn = Callable[[Item, Condition], float]


def stub_text_only_prior(item: Item, condition: Condition) -> float:
    """Deterministic stand-in for a real text-only-LM query
    P(answer | question+context, no image).

    TODO(real LM): replace with an actual text-only backbone call, per
    proposal §6.2. This stub is hash-seeded (so it is reproducible across
    runs) and nudged by whether any applied factor is culturally-loaded
    or textual/contradictory — the two factor properties the proposal
    calls out as most likely to shift a text-only prior.
    """
    digest = hashlib.sha256(condition.condition_id.encode("utf-8")).digest()
    base = (int.from_bytes(digest[:4], "big") % 1000) / 1000.0  # in [0, 1)
    nudge = 0.0
    for factor in condition.applied_factors:
        coords = factor.coordinates
        if coords.culturality is Culturality.CULTURALLY_LOADED:
            nudge += 0.15
        if coords.modality is Modality.TEXTUAL and coords.conflict is Conflict.CONTRADICTORY:
            nudge += 0.1
    return max(0.0, min(1.0, base * 0.5 + nudge))


def prior_shift_measurement(
    item: Item, condition: Condition, prior_fn: PriorFn = stub_text_only_prior
) -> float:
    """P(answer | perturbed context) - P(answer | clean context), i.e. the
    shift in text-only prior induced by `condition` relative to the
    control (empty-set) condition for the same item."""
    control = Condition(item.item_id, frozenset(), make_condition_id(item.item_id, frozenset()))
    return prior_fn(item, condition) - prior_fn(item, control)
