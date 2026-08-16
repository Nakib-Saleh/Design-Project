"""Exact Shapley attribution over the factorial lattice (proposal §6.4).

    phi_m(x) = sum_{S subseteq M\\{m}} [|S|!(k-|S|-1)!/k!] * (p(S) - p(S union {m}))

phi_m > 0 means factor m harms correctness, averaged over all contexts of
the other factors. Shapley is the unique attribution satisfying
efficiency/symmetry/null-player over the lattice — with interacting
factors, leave-one-out double-counts or hides shared effects (kept here
only as a baseline for comparison, per §6.4's "why Shapley, not
leave-one-out").
"""

import itertools
import math
from typing import Callable, Sequence

from culprit_vqa.layer0_taxonomy.factors import Factor

PFunction = Callable[[frozenset[Factor]], float]


def shapley_values(p_fn: PFunction, factors: Sequence[Factor]) -> dict[Factor, float]:
    """Exact Shapley value of each factor in `factors` over p_fn's lattice."""
    k = len(factors)
    phi: dict[Factor, float] = {}
    for m in factors:
        others = [f for f in factors if f != m]
        total = 0.0
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                S = frozenset(subset)
                weight = math.factorial(len(S)) * math.factorial(k - len(S) - 1) / math.factorial(k)
                total += weight * (p_fn(S) - p_fn(S | {m}))
        phi[m] = total
    return phi


def leave_one_out(p_fn: PFunction, factors: Sequence[Factor]) -> dict[Factor, float]:
    """Baseline attribution: p(M \\ {m}) - p(M) for each factor m.

    Biased under interaction — kept only as the comparison point Shapley
    is designed to beat (RQ1)."""
    M = frozenset(factors)
    return {m: p_fn(M - {m}) - p_fn(M) for m in factors}


def interaction_indices(
    p_fn: PFunction, factors: Sequence[Factor]
) -> dict[frozenset[Factor], float]:
    """Pairwise Shapley interaction index (Grabisch-Roubens) for every
    factor pair:

        I(i,j) = sum_{S subseteq M\\{i,j}} [|S|!(k-|S|-2)!/(k-1)!]
                 * (p(S+i+j) - p(S+i) - p(S+j) + p(S))

    Quantifies compositional (super-/sub-additive) effects between factor
    pairs — the interaction the plain per-factor Shapley value alone does
    not surface. Requires k >= 2; returns {} otherwise.
    """
    k = len(factors)
    if k < 2:
        return {}
    result: dict[frozenset[Factor], float] = {}
    for i, j in itertools.combinations(factors, 2):
        rest = [f for f in factors if f != i and f != j]
        total = 0.0
        for r in range(len(rest) + 1):
            for subset in itertools.combinations(rest, r):
                S = frozenset(subset)
                weight = (
                    math.factorial(len(S)) * math.factorial(k - len(S) - 2) / math.factorial(k - 1)
                )
                total += weight * (p_fn(S | {i, j}) - p_fn(S | {i}) - p_fn(S | {j}) + p_fn(S))
        result[frozenset({i, j})] = total
    return result
