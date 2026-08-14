import pytest

from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import generate_lattice


def _item(item_id="item"):
    return Item(item_id=item_id, image_ref="ref", question="q?", answer="a")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_lattice_size_is_2_pow_k(k):
    factors = [
        get_operator(op_id)
        for op_id in ["diffusion_irrelevant_object", "irrelevant_plausible_fact", "text_overlay_wrong_answer"][:k]
    ]
    lattice = generate_lattice(_item(), factors)
    assert len(lattice) == 2 ** k


def test_every_condition_is_a_valid_subset_of_assigned_factors():
    factors = [get_operator("diffusion_irrelevant_object"), get_operator("irrelevant_plausible_fact")]
    lattice = generate_lattice(_item(), factors)
    factor_set = set(factors)
    for condition in lattice:
        assert condition.applied_factors <= factor_set


def test_empty_and_full_set_each_appear_exactly_once():
    factors = [get_operator("diffusion_irrelevant_object"), get_operator("irrelevant_plausible_fact")]
    lattice = generate_lattice(_item(), factors)
    subsets = [c.applied_factors for c in lattice]
    assert subsets.count(frozenset()) == 1
    assert subsets.count(frozenset(factors)) == 1


def test_four_factors_raises():
    factors = [
        get_operator(op_id)
        for op_id in [
            "diffusion_irrelevant_object",
            "irrelevant_plausible_fact",
            "text_overlay_wrong_answer",
            "wrong_local_entity",
        ]
    ]
    with pytest.raises(ValueError):
        generate_lattice(_item(), factors)
