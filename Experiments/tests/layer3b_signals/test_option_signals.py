"""S2/S3/S4 on the option distribution.

These three signals were constant 0.0 across all 1000 items of the first
real attribution run. The tests below pin down both halves of the fix:
that they now respond to real evidence, and that they stay at 0.0 in the
cases where responding would be an invention.
"""

import pytest

from culprit_vqa.layer1_intervention.items import Item, PerturbationEffect
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3b_signals.option_evidence import conflict, distractor_option_indices
from culprit_vqa.layer3b_signals.stubs import AttrRatioSignal, NLIConflictSignal
from culprit_vqa.layer3b_signals.uptake import UptakeSignal

FACTOR = "text_overlay_wrong_answer"
NO_TARGET = "salience_recomposition"


def _item(**effects):
    return Item(
        item_id="it",
        image_ref="ref",
        question="q?",
        answer="mango",
        perturbation_effects=effects or {
            FACTOR: PerturbationEffect(distractor_keywords=("banana",), distractor_option_idx=1)
        },
    )


def _res(condition_id, probs, predicted=None, trace="A"):
    if predicted is None and probs:
        predicted = max(range(len(probs)), key=lambda i: probs[i])
    return RunResult(
        item_id="it",
        condition_id=condition_id,
        sampled_answers=["x"],
        correct_flags=[False],
        cot_traces=[trace],
        gold_prob_mass=probs[0] if probs else 0.0,
        extra_signals={"option_probs": probs, "predicted_idx": predicted} if probs else {},
    )


CLEAN = "it::control"
PERT = f"it::{FACTOR}"


# ---------------------------------------------------------------- S4 DRR
def test_uptake_fires_when_the_distractor_option_captures_the_answer():
    item = _item()
    clean = _res(CLEAN, [0.8, 0.1, 0.1])          # gold (idx 0) winning
    perturbed = _res(PERT, [0.2, 0.7, 0.1])       # distractor (idx 1) now wins
    assert UptakeSignal().compute(item, clean, perturbed) == 1.0


def test_uptake_is_zero_when_the_answer_moves_but_not_to_the_distractor():
    item = _item()
    clean = _res(CLEAN, [0.8, 0.1, 0.1])
    perturbed = _res(PERT, [0.2, 0.1, 0.7])       # idx 2 wins, not the distractor
    assert UptakeSignal().compute(item, clean, perturbed) == 0.0


def test_uptake_is_zero_when_the_distractor_was_already_winning():
    """Capture means the perturbation moved the answer. If the model
    already preferred that option, the cue proved nothing."""
    item = _item()
    clean = _res(CLEAN, [0.2, 0.7, 0.1])
    perturbed = _res(PERT, [0.1, 0.8, 0.1])
    assert UptakeSignal().compute(item, clean, perturbed) == 0.0


def test_uptake_is_zero_for_a_factor_with_no_option_level_target():
    item = _item(**{NO_TARGET: PerturbationEffect(distractor_keywords=("cropped",))})
    clean = _res(CLEAN, [0.8, 0.1, 0.1])
    perturbed = _res(f"it::{NO_TARGET}", [0.1, 0.8, 0.1])
    assert UptakeSignal().compute(item, clean, perturbed) == 0.0


def test_uptake_falls_back_to_trace_keywords_when_no_option_probs():
    """The mock runner supplies no distribution; the synthetic pipeline
    must keep measuring something rather than silently going dead."""
    item = _item()
    clean = _res(CLEAN, None, trace="the answer is mango")
    perturbed = _res(PERT, None, trace="I see a banana so probably that")
    assert UptakeSignal().compute(item, clean, perturbed) == 1.0


# ---------------------------------------------------------- S2 attr ratio
def test_attr_ratio_is_the_probability_moved_onto_the_distractor():
    item = _item()
    clean = _res(CLEAN, [0.8, 0.1, 0.1])
    perturbed = _res(PERT, [0.2, 0.7, 0.1])
    assert AttrRatioSignal().compute(item, clean, perturbed) == pytest.approx(0.6)


def test_attr_ratio_is_negative_when_the_cue_repels():
    item = _item()
    clean = _res(CLEAN, [0.5, 0.4, 0.1])
    perturbed = _res(PERT, [0.8, 0.1, 0.1])
    assert AttrRatioSignal().compute(item, clean, perturbed) == pytest.approx(-0.3)


def test_attr_ratio_is_zero_for_a_factor_with_no_option_level_target():
    item = _item(**{NO_TARGET: PerturbationEffect(distractor_keywords=("cropped",))})
    clean = _res(CLEAN, [0.8, 0.1, 0.1])
    perturbed = _res(f"it::{NO_TARGET}", [0.1, 0.8, 0.1])
    assert AttrRatioSignal().compute(item, clean, perturbed) == 0.0


def test_attr_ratio_never_reads_the_gold_option():
    """Non-circularity guard. phi is a function of p(gold); if this signal
    moved when only p(gold) moved, Layer 4 could reconstruct its own
    target instead of predicting it. Distractor mass is held fixed here
    while everything else is rearranged."""
    item = _item()
    clean = _res(CLEAN, [0.6, 0.2, 0.2])
    perturbed = _res(PERT, [0.1, 0.2, 0.7])  # gold collapses, distractor unchanged
    assert AttrRatioSignal().compute(item, clean, perturbed) == pytest.approx(0.0)


# ------------------------------------------------------------- S3 conflict
def test_conflict_is_maximal_for_a_tie_and_zero_for_certainty():
    assert conflict([0.5, 0.5]) == pytest.approx(1.0)
    assert conflict([1.0, 0.0]) == pytest.approx(0.0)


def test_nli_conflict_rises_when_the_perturbation_creates_a_two_horse_race():
    item = _item()
    clean = _res(CLEAN, [0.9, 0.05, 0.05])        # margin 0.85 -> conflict 0.15
    perturbed = _res(PERT, [0.45, 0.45, 0.10])    # margin 0.00 -> conflict 1.00
    assert NLIConflictSignal().compute(item, clean, perturbed) == pytest.approx(0.85)


def test_nli_conflict_is_live_even_without_an_option_level_target():
    """The one of the three that applies to every factor, including a
    salience crop -- which is why it is worth having at all."""
    item = _item(**{NO_TARGET: PerturbationEffect(distractor_keywords=("cropped",))})
    clean = _res(CLEAN, [0.9, 0.05, 0.05])
    perturbed = _res(f"it::{NO_TARGET}", [0.45, 0.45, 0.10])
    assert NLIConflictSignal().compute(item, clean, perturbed) > 0.5


def test_nli_conflict_is_zero_without_option_probs():
    item = _item()
    assert NLIConflictSignal().compute(item, _res(CLEAN, None), _res(PERT, None)) == 0.0


# --------------------------------------------------------------- plumbing
def test_distractor_indices_recovered_from_a_composed_condition_id():
    item = Item(
        item_id="it", image_ref="r", question="q", answer="a",
        perturbation_effects={
            FACTOR: PerturbationEffect(distractor_option_idx=1),
            NO_TARGET: PerturbationEffect(distractor_option_idx=None),
        },
    )
    combined = f"it::{NO_TARGET}+{FACTOR}"
    assert distractor_option_indices(item, combined) == [1]


def test_empty_perturbation_effects_yields_no_targets():
    """The exact state of the first attribution run: operators ran, but
    nothing was recorded, so every option-level signal was dead."""
    item = Item(item_id="it", image_ref="r", question="q", answer="a", perturbation_effects={})
    assert distractor_option_indices(item, PERT) == []
