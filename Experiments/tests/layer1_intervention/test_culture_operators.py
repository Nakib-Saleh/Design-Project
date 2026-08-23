"""The replacement cultural factors.

`wrong_local_entity` measured +1.1% (p=0.20) at n=698 -- inert. These
tests pin the structural properties that made it inert and that the
replacements are meant to avoid, so a future edit cannot quietly
regress one of them back into an ignorable aside.
"""

import random

from PIL import Image

from culprit_vqa.layer1_intervention.real_operators import (
    CULTURE_OPERATOR_IDS,
    REAL_OPERATORS,
    apply_cultural_framing_shift,
    apply_cultural_text_overlay,
    apply_entity_swap_in_question,
    apply_native_language_question,
    apply_wrong_local_entity,
    describe_factor_effects,
)

OPTIONS = ["Torii", "Pagoda", "Temple", "Shrine"]
META = {"subset": ("Indonesian", "Indonesia"), "native_question": "Apa nama bangunan ini?"}


def _img():
    return Image.new("RGB", (64, 64), (120, 130, 140))


def _rng():
    return random.Random(0)


# ------------------------------------------------------- native language
def test_native_language_swaps_in_the_real_native_question():
    out = apply_native_language_question(_img(), "What is this building?", OPTIONS, 0, _rng(), META)
    assert out.question == "Apa nama bangunan ini?"
    assert out.image is not None


def test_native_language_is_a_noop_without_a_distinct_native_question():
    """Must be detectable as not-applied, so the manipulation check can
    report the subset it actually ran on instead of diluting the effect
    with untouched items."""
    q = "What is this building?"
    out = apply_native_language_question(_img(), q, OPTIONS, 0, _rng(), {"subset": ("English", "USA")})
    assert out.question == q
    assert out.distractor_keywords == ()


# -------------------------------------------------------- framing shift
def test_framing_shift_embeds_the_claim_in_the_question_stem():
    q = "What is the name of this structure?"
    out = apply_cultural_framing_shift(_img(), q, OPTIONS, 0, _rng(), META)
    assert out.question != q
    assert out.question.lower().startswith("in this traditional ")
    # The original question survives as a clause rather than being replaced.
    assert "name of this structure" in out.question


def test_framing_shift_never_asserts_the_items_own_culture():
    """Asserting the true culture would make the intervention a no-op
    dressed up as a perturbation."""
    for seed in range(30):
        out = apply_cultural_framing_shift(
            _img(), "What is this?", OPTIONS, 0, random.Random(seed), META
        )
        assert "indonesian" not in out.question.lower()


def test_wrong_local_entity_remains_a_parenthetical_for_contrast():
    """Documents the defect being fixed: the old operator leaves the
    question fully answerable with the aside ignored."""
    q = "What is the name of this structure?"
    out = apply_wrong_local_entity(_img(), q, OPTIONS, 0, _rng(), META)
    assert out.question.endswith(q)
    assert out.question.startswith("(")


# ------------------------------------------------------- culture overlay
def test_culture_overlay_changes_pixels_and_leaves_the_question_alone():
    img = _img()
    out = apply_cultural_text_overlay(img, "What is this?", OPTIONS, 0, _rng(), META)
    assert out.question == "What is this?"
    assert list(out.image.getdata()) != list(img.getdata())


def test_culture_overlay_targets_no_answer_option():
    """The point of this operator is to separate 'insensitive to culture'
    from 'insensitive to text'. If it named an option it would just be a
    second answer-injection factor."""
    out = apply_cultural_text_overlay(_img(), "What is this?", OPTIONS, 0, _rng(), META)
    assert out.distractor_option_idx is None


def test_culture_overlay_never_names_the_items_own_country():
    for seed in range(30):
        out = apply_cultural_text_overlay(
            _img(), "What is this?", OPTIONS, 0, random.Random(seed), META
        )
        assert "indonesia" not in out.distractor_keywords


# ----------------------------------------------------------- entity swap
def test_entity_swap_replaces_the_entity_rather_than_appending():
    q = "Is this building found in Jakarta?"
    out = apply_entity_swap_in_question(_img(), q, OPTIONS, 0, _rng(), META)
    assert "Jakarta" not in out.question
    assert out.question != q


def test_entity_swap_is_a_noop_when_no_entity_is_present():
    q = "what is the name of this structure?"
    out = apply_entity_swap_in_question(_img(), q, OPTIONS, 0, _rng(), META)
    assert out.question == q


def test_entity_swap_ignores_a_leading_question_word():
    """'What' at position 0 is capitalised by sentence case, not because
    it is an entity."""
    q = "What is this?"
    out = apply_entity_swap_in_question(_img(), q, OPTIONS, 0, _rng(), META)
    assert out.question == q


# ------------------------------------------------------------- registry
def test_every_culture_operator_is_registered_and_uniform():
    img = _img()
    for fid in CULTURE_OPERATOR_IDS:
        assert fid in REAL_OPERATORS
        out = REAL_OPERATORS[fid](img, "What is this?", OPTIONS, 0, _rng(), META)
        assert isinstance(out.question, str)
        assert isinstance(out.distractor_keywords, tuple)
        assert out.image is not None


def test_operators_run_without_metadata():
    """The manipulation-check kernel builds items before culture metadata
    is guaranteed present; nothing may crash on a missing subset."""
    for fid in CULTURE_OPERATOR_IDS:
        out = REAL_OPERATORS[fid](_img(), "What is this?", OPTIONS, 0, _rng(), None)
        assert isinstance(out.question, str)


def test_describe_factor_effects_records_the_injected_option():
    effects = describe_factor_effects(
        _img(), "What fruit?", ["mango", "banana", "apple", "grape"], 0,
        ["text_overlay_wrong_answer", "salience_recomposition"],
        seed_key_for=lambda fid: f"seed::it::it::{fid}",
        metadata=META,
    )
    assert effects["text_overlay_wrong_answer"].distractor_option_idx != 0
    assert effects["text_overlay_wrong_answer"].distractor_option_idx is not None
    assert effects["salience_recomposition"].distractor_option_idx is None


def test_describe_factor_effects_matches_what_the_singleton_run_injects():
    """The recorded distractor must be the option the model was actually
    shown. A seed mismatch here would turn S2/S4 into noise rather than
    leaving them at zero -- strictly worse than the bug being fixed."""
    options = ["mango", "banana", "apple", "grape"]
    key = "0::it::it::text_overlay_wrong_answer"
    effects = describe_factor_effects(
        _img(), "What fruit?", options, 0, ["text_overlay_wrong_answer"],
        seed_key_for=lambda fid: key, metadata=META,
    )
    direct = REAL_OPERATORS["text_overlay_wrong_answer"](
        _img(), "What fruit?", options, 0, random.Random(key), META
    )
    assert effects["text_overlay_wrong_answer"].distractor_option_idx == direct.distractor_option_idx
