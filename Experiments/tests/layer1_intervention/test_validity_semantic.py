from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item, PerturbationEffect
from culprit_vqa.layer1_intervention.lattice import generate_lattice
from culprit_vqa.layer1_intervention.validity.semantic import semantic_preservation_check


def _make_item(perturbed_answer):
    factor = get_operator("wrong_local_entity")
    item = Item(
        item_id="item",
        image_ref="ref",
        question="q?",
        answer="mango",
        perturbation_effects={factor.id: PerturbationEffect(perturbed_answer=perturbed_answer)},
    )
    return item, factor


def test_answer_changing_perturbation_is_rejected():
    item, factor = _make_item(perturbed_answer="banana")
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors]
    assert semantic_preservation_check(item, condition) is False


def test_answer_preserving_perturbation_is_accepted():
    item, factor = _make_item(perturbed_answer="Mango")  # case-insensitive match
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors]
    assert semantic_preservation_check(item, condition) is True


def test_factor_with_no_declared_effect_is_accepted():
    factor = get_operator("wrong_local_entity")
    item = Item(item_id="item", image_ref="ref", question="q?", answer="mango")
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors]
    assert semantic_preservation_check(item, condition) is True
