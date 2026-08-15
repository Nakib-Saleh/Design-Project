"""Property-style tests of the Shapley axioms over synthetic p(S) functions
(proposal §6.4): efficiency, symmetry, null player, and a pure-interaction
case where only p(M) differs from every proper-subset value."""

from hypothesis import given
from hypothesis import strategies as st

from culprit_vqa.layer0_taxonomy.axes import Conflict, Culturality, Modality, Relevance
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates
from culprit_vqa.layer3a_causal.attribution import efficiency_residual
from culprit_vqa.layer3a_causal.shapley import shapley_values

TOL = 1e-9


def _factor(fid: str) -> Factor:
    return Factor(
        id=fid,
        coordinates=FactorCoordinates(
            Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
        ),
    )


def _make_p_fn(factors, values_by_ids: dict[frozenset[str], float]):
    def p_fn(subset: frozenset[Factor]) -> float:
        key = frozenset(f.id for f in subset)
        return values_by_ids[key]

    return p_fn


def _all_subset_id_keys(factor_ids: list[str]) -> list[frozenset[str]]:
    import itertools

    keys = []
    for r in range(len(factor_ids) + 1):
        for combo in itertools.combinations(factor_ids, r):
            keys.append(frozenset(combo))
    return keys


# --- Efficiency: holds for ANY p(S) values, by construction of Shapley. ---


@given(
    values=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=8, max_size=8)
)
def test_efficiency_holds_for_arbitrary_p_values_k3(values):
    factors = [_factor("a"), _factor("b"), _factor("c")]
    keys = _all_subset_id_keys(["a", "b", "c"])
    values_by_ids = dict(zip(keys, values))
    p_fn = _make_p_fn(factors, values_by_ids)

    phi = shapley_values(p_fn, factors)
    residual = efficiency_residual(phi, p_fn, factors)
    assert abs(residual) < TOL


# --- Symmetry: interchangeable factors get equal Shapley values. ---


def test_symmetric_factors_get_equal_shapley_values():
    a, b, c = _factor("a"), _factor("b"), _factor("c")

    def p_fn(subset: frozenset[Factor]) -> float:
        # p(S) depends only on |S| -- every factor is interchangeable.
        return 1.0 - 0.15 * len(subset)

    phi = shapley_values(p_fn, [a, b, c])
    assert abs(phi[a] - phi[b]) < TOL
    assert abs(phi[b] - phi[c]) < TOL


# --- Null player: a factor that never changes p(S) gets phi == 0. ---


def test_null_player_gets_zero_shapley_value():
    a, b, c = _factor("a"), _factor("b"), _factor("c")

    def p_fn(subset: frozenset[Factor]) -> float:
        # c is a null player: p(S) never depends on whether c in S.
        relevant = subset & {a, b}
        return 1.0 - 0.2 * len(relevant)

    phi = shapley_values(p_fn, [a, b, c])
    assert abs(phi[c]) < TOL
    # a and b are symmetric non-null players and should still get equal, nonzero credit.
    assert abs(phi[a] - phi[b]) < TOL
    assert phi[a] > 0


# --- Pure interaction: only p(M) differs; credit splits symmetrically. ---


def test_pure_interaction_splits_credit_symmetrically():
    a, b, c = _factor("a"), _factor("b"), _factor("c")
    keys = _all_subset_id_keys(["a", "b", "c"])
    full = frozenset({"a", "b", "c"})
    values_by_ids = {key: 0.9 for key in keys}
    values_by_ids[full] = 0.5  # only the full combination breaks the model

    p_fn = _make_p_fn([a, b, c], values_by_ids)
    phi = shapley_values(p_fn, [a, b, c])

    # All three factors are symmetric in this construction -> equal credit.
    assert abs(phi[a] - phi[b]) < TOL
    assert abs(phi[b] - phi[c]) < TOL
    # Efficiency still holds exactly.
    residual = efficiency_residual(phi, p_fn, [a, b, c])
    assert abs(residual) < TOL
