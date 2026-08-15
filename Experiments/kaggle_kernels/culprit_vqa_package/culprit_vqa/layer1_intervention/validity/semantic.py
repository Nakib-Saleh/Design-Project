"""Semantic preservation check (proposal §6.2, check 1 of 3).

In production this is "judge model + native-speaker human audit on a 10%
stratified sample" (proposal §6.2). For the synthetic-fixture scaffold,
each factor's PerturbationEffect can declare a fixture-authored
`perturbed_answer` — the answer the item would have if that factor's
effect actually changed it. The check accepts a condition iff the
combination of applied factors leaves the ground-truth answer unchanged.
"""

from typing import Callable

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition

SemanticCheckFn = Callable[[Item, Condition], bool]


def semantic_preservation_check(item: Item, condition: Condition) -> bool:
    """Accept iff every applied factor's fixture-authored `perturbed_answer`
    (when set) matches `item.answer` (case-insensitive). A factor with no
    `perturbed_answer` declared is assumed not to alter the ground truth.
    """
    for factor in condition.applied_factors:
        effect = item.perturbation_effects.get(factor.id)
        if effect is None or effect.perturbed_answer is None:
            continue
        if effect.perturbed_answer.strip().lower() != item.answer.strip().lower():
            return False
    return True
