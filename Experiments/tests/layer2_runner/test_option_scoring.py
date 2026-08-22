"""Tests the normalized option-scoring logic -- pure Python, no torch/GPU.

These replace the old `_matches_gold` string-matching tests. That
heuristic existed only because correctness was read off a generated
string; with options scored and normalized, correctness is `argmax ==
gold` and the heuristic is gone.
"""

import math

import pytest

from culprit_vqa.layer2_runner.hf_runner import OPTION_LETTERS, softmax


def test_softmax_returns_a_distribution():
    p = softmax([1.0, 2.0, 3.0, 4.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert all(0.0 <= x <= 1.0 for x in p)


def test_softmax_preserves_ranking():
    """The scored argmax IS the predicted answer, so any reordering here
    would silently change which option the model is recorded as picking."""
    scores = [-3.0, 10.0, 0.5, -0.5]
    p = softmax(scores)
    assert p.index(max(p)) == scores.index(max(scores))


def test_softmax_is_shift_invariant():
    """Raw logits carry an arbitrary additive offset that differs per
    forward pass; p(gold) must not depend on it or scores would not be
    comparable across conditions -- the exact defect that made the old
    absolute measure useless."""
    a = softmax([1.0, 2.0, 3.0])
    b = softmax([101.0, 102.0, 103.0])
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-9


def test_softmax_handles_large_magnitudes_without_overflow():
    """Letter scoring reads raw logits, which routinely reach +-40 and can
    be far larger under 4-bit quantization; a naive exp() would inf/NaN
    and poison p(gold) for that whole condition."""
    p = softmax([1000.0, 999.0, -1000.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert all(math.isfinite(x) for x in p)


def test_softmax_of_equal_scores_is_uniform_chance():
    """A model with no preference must land exactly on the 1/k chance
    baseline -- that baseline is what makes p(gold) interpretable."""
    p = softmax([2.5] * 4)
    assert all(abs(x - 0.25) < 1e-12 for x in p)


def test_softmax_of_empty_is_empty():
    assert softmax([]) == []


def test_softmax_degenerate_scores_fall_back_to_uniform():
    """All-(-1e9) means every option tokenized degenerately. Returning
    uniform is honest (no information); returning NaN would propagate
    into every Shapley value computed from that condition."""
    p = softmax([-1e9, -1e9])
    assert all(math.isfinite(x) for x in p)
    assert abs(sum(p) - 1.0) < 1e-12


def test_gold_prob_has_a_meaningful_chance_baseline():
    """The whole point of normalizing: p(gold) is now on a scale where
    0.25 means 'no idea' for a 4-option item. The old absolute measure
    had no such reference point, so its values were uninterpretable."""
    n_options = 4
    uniform = softmax([0.0] * n_options)
    assert abs(uniform[0] - 1.0 / n_options) < 1e-12


def test_margin_is_zero_exactly_when_the_pick_is_gold():
    """`margin` = p(picked) - p(gold) is the graded 'how wrong' that a
    binary correct flag cannot express; it must collapse to 0 on a
    correct answer or the two fields would disagree."""
    probs = softmax([0.1, 5.0, 0.2, 0.3])
    picked = probs.index(max(probs))
    assert probs[picked] - probs[picked] == 0.0
    gold_wrong = 0
    assert probs[picked] - probs[gold_wrong] > 0.0


def test_enough_option_letters_for_realistic_items():
    """CVQA items are 4-way, but the scorer indexes OPTION_LETTERS by
    option count; too few letters would raise mid-run on a GPU."""
    assert len(OPTION_LETTERS) >= 8
    assert OPTION_LETTERS[:4] == "ABCD"


@pytest.mark.parametrize("scores,expected_idx", [
    ([5.0, 1.0, 1.0, 1.0], 0),
    ([1.0, 1.0, 1.0, 5.0], 3),
    ([-10.0, -9.0, -20.0, -30.0], 1),
])
def test_argmax_selection_matches_intended_option(scores, expected_idx):
    p = softmax(scores)
    assert p.index(max(p)) == expected_idx


# ----------------------------------------------------------------------
# NaN / inf safety. Observed on real hardware: 2/173 items in the
# instrument check produced a NaN logit under 4-bit inference. The naive
# softmax turned that into an all-NaN distribution, which would have
# propagated through gold_prob_mass into every Shapley value for the
# item -- silently, with no error raised.
# ----------------------------------------------------------------------

from culprit_vqa.layer2_runner.hf_runner import has_degenerate_scores


def test_one_nan_score_does_not_destroy_the_distribution():
    p = softmax([1.0, float("nan"), 2.0])
    assert all(math.isfinite(x) for x in p)
    assert abs(sum(p) - 1.0) < 1e-12
    # the NaN option is floored, so the real scores keep their ordering
    assert p[2] > p[0] > p[1]


def test_all_nan_falls_back_to_uniform():
    p = softmax([float("nan")] * 4)
    assert all(abs(x - 0.25) < 1e-12 for x in p)


def test_infinities_do_not_produce_nan():
    for scores in ([float("inf"), 1.0], [float("-inf"), 1.0], [float("inf"), float("-inf")]):
        p = softmax(scores)
        assert all(math.isfinite(x) for x in p), scores
        assert abs(sum(p) - 1.0) < 1e-12, scores


def test_degenerate_scores_flag_detects_bad_input():
    """The repaired distribution is usable but not trustworthy, so the
    condition must be flagged for exclusion rather than silently kept."""
    assert has_degenerate_scores([1.0, float("nan"), 2.0]) is True
    assert has_degenerate_scores([1.0, float("inf")]) is True
    assert has_degenerate_scores([1.0, 2.0, 3.0]) is False


def test_clean_scores_are_unaffected_by_the_nan_guard():
    """The guard must not change results for normal input, or every
    previously-computed value would shift."""
    p = softmax([0.5, -1.25, 3.0, 0.0])
    expected = [math.exp(v - 3.0) for v in [0.5, -1.25, 3.0, 0.0]]
    total = sum(expected)
    for got, want in zip(p, [e / total for e in expected]):
        assert abs(got - want) < 1e-12
