"""Normalized attribution alpha and the efficiency-axiom invariant (proposal §6.4).

    alpha_m(x) = max(0, phi_m(x)) / sum_j max(0, phi_j(x))

alpha_m(x) is the normalized causal contribution of factor m to the
correctness drop, under the intervention distribution of the lattice —
nothing else. When no factor harms correctness (sum of positive phi is
zero), there is no cause to attribute: this is the null/abstain case,
returned as an all-zero vector rather than raising ZeroDivisionError.
"""

from typing import Sequence

from culprit_vqa.layer0_taxonomy.factors import Factor


def normalize_attribution(phi: dict[Factor, float]) -> dict[Factor, float]:
    """Normalize positive Shapley contributions to sum to 1.

    Explicit null case: if every phi_m <= 0 (no factor harms
    correctness), returns an all-zero dict rather than dividing by zero —
    "no factor in the space caused this" is itself a valid, distinct
    outcome (see proposal §11: "injected factor causes nothing" is a
    finding, not an error).
    """
    positive = {f: max(0.0, v) for f, v in phi.items()}
    total = sum(positive.values())
    if total == 0.0:
        return {f: 0.0 for f in phi}
    return {f: v / total for f, v in positive.items()}


def is_null_attribution(alpha: dict[Factor, float]) -> bool:
    """True iff `alpha` is the all-zero/no-cause-identified vector."""
    return all(v == 0.0 for v in alpha.values())


def efficiency_residual(phi: dict[Factor, float], p_fn, factors: Sequence[Factor]) -> float:
    """Testable invariant: sum(phi) - (p(empty) - p(M)) — should be ~0 for
    any exact Shapley computation (the efficiency axiom).

    Note the direction: phi_m is defined as p(S) - p(S union {m}) (proposal
    §6.4), i.e. the *harm* attributed to m — the negative of the standard
    Shapley value of the value function p. Standard Shapley efficiency says
    sum_m phi_m^std(p) = p(M) - p(empty); since our phi_m = -phi_m^std, the
    correct identity for THIS sign convention is
    sum_m phi_m = p(empty) - p(M)
    (the total correctness drop from the control condition to the fully
    perturbed condition, i.e. total harm attributed across all factors).
    """
    M = frozenset(factors)
    return sum(phi.values()) - (p_fn(frozenset()) - p_fn(M))
