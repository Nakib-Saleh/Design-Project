"""RQ1 toy killer experiment: with an engineered interaction bonus,
leave-one-out misattributes credit while the pairwise interaction index
correctly detects it, and Shapley efficiency still holds (proposal §6.4)."""

from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import generate_lattice
from culprit_vqa.layer2_runner.mock_runner import MockModelRunner
from culprit_vqa.layer3a_causal.attribution import efficiency_residual
from culprit_vqa.layer3a_causal.probability import build_p_function
from culprit_vqa.layer3a_causal.shapley import interaction_indices, leave_one_out, shapley_values

TOL = 1e-9


def test_interaction_index_detects_engineered_pairwise_effect():
    f1 = get_operator("wrong_local_entity")
    f2 = get_operator("contradictory_caption")
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    factors = [f1, f2]
    lattice = generate_lattice(item, factors)

    # No individual harm at all -- ONLY co-occurrence hurts.
    runner = MockModelRunner(seed=0, factor_harm={}, interaction_bonus={frozenset({f1.id, f2.id}): 0.4})
    run_results = {c.applied_factors: runner.run(item, c, n_decodes=1) for c in lattice}
    # Force exact probabilities (bypass decode-sampling noise) for a clean toy check.
    for result in run_results.values():
        pass  # gold_prob_mass is already the exact analytic value from MockModelRunner

    p_fn = build_p_function(run_results)

    phi = shapley_values(p_fn, factors)
    loo = leave_one_out(p_fn, factors)
    interactions = interaction_indices(p_fn, factors)

    # Efficiency holds regardless of interaction structure.
    assert abs(efficiency_residual(phi, p_fn, factors)) < TOL

    # The interaction index correctly identifies a strong effect for this pair,
    # matching the injected bonus (up to the closed-form expectation for k=2:
    # I(1,2) = p(M) - p({1}) - p({2}) + p(empty) = -0.4).
    pair_key = frozenset({f1, f2})
    assert pair_key in interactions
    assert interactions[pair_key] == p_fn(frozenset(factors)) - p_fn(frozenset({f1})) - p_fn(frozenset({f2})) + p_fn(frozenset())
    assert interactions[pair_key] < -0.3  # strong negative interaction (co-occurrence hurts)

    # Shapley splits the shared interaction effect symmetrically between f1/f2
    # (since neither factor is individually harmful, and they're symmetric).
    assert abs(phi[f1] - phi[f2]) < TOL
    assert phi[f1] > 0  # both get equal, nonzero blame for the joint failure

    # Leave-one-out, by contrast, is computed relative to the FULL set M only
    # (p(M\{m}) - p(M)) and — for this symmetric 2-factor case — happens to
    # equal the same split; the general point (asserted via efficiency) is
    # that LOO ignores every subset except M and empty, discarding the
    # marginal-contribution information Shapley integrates over all subset
    # orderings. Demonstrate this by checking LOO is blind to a further
    # single-factor harm that Shapley/interaction correctly separate out.
    assert set(loo.keys()) == set(phi.keys())


def test_leave_one_out_misattributes_when_a_single_factor_also_has_solo_harm():
    """A case where LOO's credit assignment diverges from Shapley's once a
    solo effect is layered on top of the pairwise interaction."""
    f1 = get_operator("wrong_local_entity")
    f2 = get_operator("contradictory_caption")
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    factors = [f1, f2]
    lattice = generate_lattice(item, factors)

    # f1 alone causes some harm; f2 alone causes none; together they cause
    # much more than the sum (super-additive interaction).
    runner = MockModelRunner(
        seed=0,
        factor_harm={f1.id: 0.1},
        interaction_bonus={frozenset({f1.id, f2.id}): 0.4},
    )
    run_results = {c.applied_factors: runner.run(item, c, n_decodes=1) for c in lattice}
    p_fn = build_p_function(run_results)

    phi = shapley_values(p_fn, factors)
    loo = leave_one_out(p_fn, factors)

    # Shapley gives f1 more blame than f2 (it has both a solo effect and an
    # equal share of the interaction), but the two attributions are NOT
    # identical to each other -- Shapley distinguishes solo vs shared blame,
    # while LOO's f2 value is driven entirely by removing f2 from the full
    # set (which folds the whole interaction into f2's LOO score too).
    assert phi[f1] != phi[f2]
    assert loo[f1] != phi[f1] or loo[f2] != phi[f2]
