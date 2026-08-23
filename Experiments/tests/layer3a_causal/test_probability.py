import pytest

from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import generate_lattice
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3a_causal.probability import build_p_function, estimate_p


def _result(gold_prob_mass, correct_flags):
    return RunResult(
        item_id="item",
        condition_id="item::control",
        sampled_answers=["a" if c else "wrong" for c in correct_flags],
        correct_flags=correct_flags,
        cot_traces=["trace"] * len(correct_flags),
        gold_prob_mass=gold_prob_mass,
    )


def test_estimate_p_logit_mass_uses_gold_prob_mass():
    result = _result(0.73, [True, False, True])
    assert estimate_p(result, method="logit_mass") == 0.73


def test_estimate_p_decode_frequency_uses_correct_flag_fraction():
    result = _result(0.5, [True, True, False, False])
    assert estimate_p(result, method="decode_frequency") == 0.5


def test_estimate_p_unknown_method_raises():
    result = _result(0.5, [True])
    with pytest.raises(ValueError):
        estimate_p(result, method="bogus")


def test_build_p_function_looks_up_correct_subset():
    factor = get_operator("diffusion_irrelevant_object")
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    lattice = generate_lattice(item, [factor])
    control = next(c for c in lattice if not c.applied_factors)
    perturbed = next(c for c in lattice if c.applied_factors)

    results_by_subset = {
        control.applied_factors: _result(0.9, [True] * 8),
        perturbed.applied_factors: _result(0.4, [True, False, False, False, False, False, False, False]),
    }
    p_fn = build_p_function(results_by_subset)
    assert p_fn(frozenset()) == 0.9
    assert p_fn(frozenset({factor})) == 0.4
