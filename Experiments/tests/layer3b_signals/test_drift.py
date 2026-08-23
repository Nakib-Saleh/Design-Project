from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3b_signals.drift import DriftSignal


def _result(trace):
    return RunResult(
        item_id="item",
        condition_id="cond",
        sampled_answers=["a"],
        correct_flags=[True],
        cot_traces=[trace],
        gold_prob_mass=0.9,
    )


def test_identical_traces_have_near_zero_drift():
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    clean = _result("the answer is mango because of the fruit shape")
    perturbed = _result("the answer is mango because of the fruit shape")
    drift = DriftSignal().compute(item, clean, perturbed)
    assert drift < 1e-9


def test_disjoint_traces_have_high_drift():
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    clean = _result("alpha bravo charlie delta")
    perturbed = _result("echo foxtrot golf hotel")
    drift = DriftSignal().compute(item, clean, perturbed)
    assert drift > 1.5  # near the max of (1-cos)+(1-rougeL) = 2.0


def test_partial_overlap_gives_intermediate_drift():
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    clean = _result("the mango is on the table")
    perturbed = _result("the mango is on the shelf")
    drift = DriftSignal().compute(item, clean, perturbed)
    assert 0.0 < drift < 1.5
