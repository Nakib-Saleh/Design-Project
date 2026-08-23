from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import generate_lattice
from culprit_vqa.layer2_runner.mock_runner import MockModelRunner


def _item():
    return Item(item_id="item", image_ref="ref", question="q?", answer="a")


def test_determinism_across_repeated_calls():
    factor = get_operator("diffusion_irrelevant_object")
    item = _item()
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors]
    runner = MockModelRunner(seed=42, factor_harm={factor.id: 0.3})

    result1 = runner.run(item, condition, n_decodes=8)
    result2 = runner.run(item, condition, n_decodes=8)

    assert result1.correct_flags == result2.correct_flags
    assert result1.sampled_answers == result2.sampled_answers
    assert result1.gold_prob_mass == result2.gold_prob_mass


def test_factor_harm_strictly_decreases_gold_prob_mass():
    factor = get_operator("diffusion_irrelevant_object")
    item = _item()
    lattice = generate_lattice(item, [factor])
    control = next(c for c in lattice if not c.applied_factors)
    perturbed = next(c for c in lattice if c.applied_factors)

    runner = MockModelRunner(seed=0, factor_harm={factor.id: 0.3})
    control_result = runner.run(item, control)
    perturbed_result = runner.run(item, perturbed)

    assert perturbed_result.gold_prob_mass < control_result.gold_prob_mass


def test_interaction_bonus_only_applies_when_both_factors_present():
    f1 = get_operator("wrong_local_entity")
    f2 = get_operator("contradictory_caption")
    item = _item()
    lattice = generate_lattice(item, [f1, f2])
    by_subset = {c.applied_factors: c for c in lattice}

    runner = MockModelRunner(
        seed=0,
        factor_harm={},
        interaction_bonus={frozenset({f1.id, f2.id}): 0.4},
    )

    p_empty = runner.run(item, by_subset[frozenset()]).gold_prob_mass
    p_f1 = runner.run(item, by_subset[frozenset({f1})]).gold_prob_mass
    p_f2 = runner.run(item, by_subset[frozenset({f2})]).gold_prob_mass
    p_both = runner.run(item, by_subset[frozenset({f1, f2})]).gold_prob_mass

    # No individual harm configured, so singles match the base rate.
    assert p_f1 == p_empty
    assert p_f2 == p_empty
    # The bonus only kicks in when both are present.
    assert p_both == p_empty - 0.4
