"""Factorial lattice generation (proposal §6.1).

Each item is assigned k <= 3 factors; all 2^k subsets are generated as
conditions. This is the design that makes interaction measurable and
Shapley attribution exact (proposal §6.4).
"""

import itertools
from dataclasses import dataclass
from typing import Sequence

from culprit_vqa.layer0_taxonomy.factors import Factor
from culprit_vqa.layer1_intervention.items import Item

MAX_FACTORS_PER_ITEM = 3


@dataclass(frozen=True)
class Condition:
    """One point in an item's factorial lattice: applying exactly the
    factors in `applied_factors` (possibly the empty set — the control)."""

    item_id: str
    applied_factors: frozenset[Factor]
    condition_id: str


def make_condition_id(item_id: str, subset: frozenset[Factor]) -> str:
    """Deterministic condition id for `item_id` applying exactly `subset`.
    Public so other modules (e.g. validity checks needing the control
    condition for a given item) can construct condition ids consistently
    without duplicating the naming scheme."""
    if not subset:
        return f"{item_id}::control"
    return f"{item_id}::" + "+".join(sorted(f.id for f in subset))


def generate_lattice(item: Item, factors: Sequence[Factor]) -> list[Condition]:
    """Generate the full 2^k lattice of conditions for `item` given its
    assigned `factors` (k = len(factors), k <= 3).

    Returns exactly 2**k conditions, one per subset of `factors`
    (including the empty set / control and the full set).
    """
    k = len(factors)
    if k > MAX_FACTORS_PER_ITEM:
        raise ValueError(
            f"k <= {MAX_FACTORS_PER_ITEM} factors per item (proposal §6.1); got k={k}"
        )
    conditions = []
    for r in range(k + 1):
        for subset in itertools.combinations(factors, r):
            fs = frozenset(subset)
            conditions.append(Condition(item.item_id, fs, make_condition_id(item.item_id, fs)))
    return conditions
