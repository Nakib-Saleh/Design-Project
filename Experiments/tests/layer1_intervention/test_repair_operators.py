from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.repair_operators import (
    PLACEBO_REPAIRS,
    REAL_REPAIRS,
    REPAIR_BY_ID,
    REPAIRS,
    upscale_image,
)


class _FakeImage:
    """Minimal stand-in for a PIL image -- `upscale_image` only needs
    `.size` and `.resize`, so the test suite stays free of a Pillow
    dependency and of any real image file."""

    def __init__(self, w, h):
        self.size = (w, h)

    def resize(self, size, resample=None):
        return _FakeImage(*size)


def _item(question="What is this?", **metadata):
    md = {"category": "Cooking and food", "subset": "('Bengali', 'India')"}
    md.update(metadata)
    return Item(
        item_id="i1", image_ref="ref", question=question, answer="a",
        language="('Bengali', 'India')", metadata=md,
    )


def test_every_repair_preserves_the_original_question_text():
    """A repair must ADD context, never replace the question -- otherwise
    a 'fix' could work by asking something easier, which would not be
    evidence about the original failure. The native-language repair is
    exempt: substituting the question is exactly what it tests."""
    item = _item()
    for repair in REPAIRS:
        if repair.repair_id == "ask_in_native_language":
            continue
        out = repair.question_fn(item)
        assert item.question in out, repair.repair_id


def test_repair_ids_are_unique_and_registry_partitions_cleanly():
    ids = [r.repair_id for r in REPAIRS]
    assert len(ids) == len(set(ids))
    assert set(REAL_REPAIRS) | set(PLACEBO_REPAIRS) == set(REPAIRS)
    assert not (set(REAL_REPAIRS) & set(PLACEBO_REPAIRS))
    assert REPAIR_BY_ID["chain_of_thought"].axis == "reasoning"


def test_five_real_repairs_cover_five_distinct_axes():
    """The whole point of the expansion: each real repair probes a
    different cause, so a flip is readable as evidence for one axis
    rather than 'something helped'."""
    assert len(REAL_REPAIRS) == 5
    axes = [r.axis for r in REAL_REPAIRS]
    assert len(set(axes)) == 5
    assert "none" not in axes


def test_placebos_exist_and_are_shape_matched_to_real_repairs():
    """Placebos must mimic the FORM of the real repairs (one prefix, one
    suffix) or they would not control for the right thing -- a prefix
    placebo cannot bound the noise floor of a suffix repair."""
    assert len(PLACEBO_REPAIRS) == 2
    item = _item()
    outs = [r.question_fn(item) for r in PLACEBO_REPAIRS]
    prefixed = [o for o in outs if o.endswith(item.question)]
    suffixed = [o for o in outs if o.startswith(item.question)]
    assert len(prefixed) == 1
    assert len(suffixed) == 1


def test_placebos_carry_no_item_specific_information():
    """A placebo that leaked the category or culture would be a weak real
    repair, not a control -- it would raise the measured noise floor and
    hide genuine effects."""
    item = _item()
    for repair in PLACEBO_REPAIRS:
        out = repair.question_fn(item)
        assert item.metadata["category"] not in out
        assert item.metadata["subset"] not in out
        assert repair.axis == "none"
        assert repair.image_fn is None


def test_category_and_cultural_hints_supply_different_information():
    """These are two separate repairs precisely because knowing the topic
    and knowing the culture can fail independently -- if they carried the
    same text, the two axes could never be told apart."""
    item = _item()
    cat = REPAIR_BY_ID["add_category_hint"].question_fn(item)
    cul = REPAIR_BY_ID["supply_cultural_context"].question_fn(item)
    assert "Cooking and food" in cat and "Cooking and food" not in cul
    assert "Bengali" in cul and "Bengali" not in cat


def test_native_language_repair_substitutes_the_native_question():
    item = _item(native_question="এটা কি?")
    out = REPAIR_BY_ID["ask_in_native_language"].question_fn(item)
    assert out == "এটা কি?"


def test_native_language_repair_falls_back_when_no_native_text():
    """Must return the question unchanged (not raise, not invent text) so
    the audit can detect the no-op and exclude the item from this
    repair's success rate."""
    item = _item()
    out = REPAIR_BY_ID["ask_in_native_language"].question_fn(item)
    assert out == item.question


def test_only_the_visual_repair_edits_the_image():
    with_image = [r for r in REPAIRS if r.image_fn is not None]
    assert [r.repair_id for r in with_image] == ["enhance_visual_detail"]
    assert REPAIR_BY_ID["enhance_visual_detail"].max_image_side is not None


def test_upscale_hits_the_target_long_side_regardless_of_source_size():
    """Absolute target, not a multiplier: the runner downsamples every
    image to a fixed cap first, so a source-relative scale would make the
    repair's real strength vary arbitrarily with source resolution."""
    for w, h in [(400, 300), (4000, 3000), (300, 400)]:
        out = upscale_image(_FakeImage(w, h), target_long_side=1152)
        assert max(out.size) == 1152
        # aspect ratio preserved within rounding
        assert abs((out.size[0] / out.size[1]) - (w / h)) < 0.01


def test_upscale_target_exceeds_runner_default_cap():
    """If it didn't, the runner would re-cap the upscaled image straight
    back down and the visual repair would cost a GPU call while
    measuring nothing."""
    from culprit_vqa.layer2_runner.hf_runner import DEFAULT_MAX_IMAGE_SIDE

    assert REPAIR_BY_ID["enhance_visual_detail"].max_image_side > DEFAULT_MAX_IMAGE_SIDE


def test_upscale_handles_missing_image():
    assert upscale_image(None) is None
