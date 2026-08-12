from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item, PerturbationEffect
from culprit_vqa.layer1_intervention.lattice import make_condition_id
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3b_signals.uptake import UptakeSignal


def _make_item_and_condition_id():
    factor = get_operator("diffusion_irrelevant_object")
    item = Item(
        item_id="item",
        image_ref="ref",
        question="q?",
        answer="mango",
        perturbation_effects={factor.id: PerturbationEffect(distractor_keywords=("bicycle",))},
    )
    condition_id = make_condition_id(item.item_id, frozenset({factor}))
    return item, condition_id


def _result(condition_id, trace):
    return RunResult(
        item_id="item",
        condition_id=condition_id,
        sampled_answers=["wrong"],
        correct_flags=[False],
        cot_traces=[trace],
        gold_prob_mass=0.5,
    )


def test_uptake_is_one_when_trace_mentions_distractor_keyword():
    item, condition_id = _make_item_and_condition_id()
    clean = _result("item::control", "the answer is mango")
    perturbed = _result(condition_id, "I noticed a bicycle in the background, so the answer is wrong")
    assert UptakeSignal().compute(item, clean, perturbed) == 1.0


def test_uptake_is_zero_when_trace_does_not_mention_distractor_keyword():
    item, condition_id = _make_item_and_condition_id()
    clean = _result("item::control", "the answer is mango")
    perturbed = _result(condition_id, "the answer is still mango")
    assert UptakeSignal().compute(item, clean, perturbed) == 0.0
