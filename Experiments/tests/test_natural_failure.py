"""Tests the natural-failure audit orchestration logic with a fake runner
(no GPU needed) -- verifies skip-if-already-correct, flip detection,
confidence-delta recording, and no-op repair handling."""

import math

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.repair_operators import PLACEBO_REPAIRS, REPAIRS, Repair
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.natural_failure import run_natural_failure_audit


class FakeRunner:
    """Deterministic fake: 'clean' is always wrong with a low gold-prob;
    repairs named in `flip_repairs` make the listed items correct. Every
    repair raises gold_prob a little, so the confidence-delta path is
    exercised even where nothing flips."""

    def __init__(self, flip_repairs=None, clean_gold_prob=0.10):
        self.flip_repairs = flip_repairs or {}
        self.clean_gold_prob = clean_gold_prob
        self.calls = []

    def answer_with_question_override(
        self, item, question_text, run_label, image_override=None, max_image_side=None
    ):
        self.calls.append((run_label, question_text, image_override, max_image_side))
        repair_id = run_label.split("::", 1)[1]
        if repair_id == "clean":
            correct, gold_prob = False, self.clean_gold_prob
        else:
            correct = item.item_id in self.flip_repairs.get(repair_id, set())
            gold_prob = 0.8 if correct else self.clean_gold_prob + 0.05
        return RunResult(
            item_id=item.item_id, condition_id=run_label,
            sampled_answers=["x"], correct_flags=[correct], cot_traces=[f"trace for {run_label}"],
            gold_prob_mass=gold_prob, extra_signals={"gen_confidence": 0.7},
        )


class AlwaysCorrectRunner:
    def answer_with_question_override(
        self, item, question_text, run_label, image_override=None, max_image_side=None
    ):
        return RunResult(item.item_id, run_label, ["a"], [True], ["trace"], 0.9)


def _item(item_id, **metadata):
    md = {"pil_image": None, "options": ["a", "b"], "correct_idx": 0,
          "category": "cat", "subset": "('Bengali', 'India')"}
    md.update(metadata)
    return Item(
        item_id=item_id, image_ref="ref", question="q?", answer="a", language="en", metadata=md,
    )


def test_natural_failure_audit_skips_items_the_model_gets_right():
    records = run_natural_failure_audit([_item("i1")], AlwaysCorrectRunner())
    assert records == []


def test_natural_failure_audit_records_flip_correctly():
    runner = FakeRunner(flip_repairs={"add_category_hint": {"i1"}})
    records = run_natural_failure_audit([_item("i1"), _item("i2")], runner)

    assert len(records) == 2
    r1 = next(r for r in records if r.item_id == "i1")
    r2 = next(r for r in records if r.item_id == "i2")

    assert r1.clean_correct is False
    assert r1.repairs["add_category_hint"]["flipped"] is True
    assert r1.repairs["chain_of_thought"]["flipped"] is False
    assert all(not o["flipped"] for o in r2.repairs.values())


def test_every_registered_repair_is_tried_on_every_wrong_item():
    """The 'apply everything, let the winner define the label' design:
    skipping any repair would bias which cause can be discovered."""
    runner = FakeRunner()
    records = run_natural_failure_audit([_item("i1")], runner)
    assert set(records[0].repairs.keys()) == {r.repair_id for r in REPAIRS}
    # 1 clean call + one per repair. The native-language repair is a
    # no-op here (no native_question), so it costs no GPU call.
    assert len(runner.calls) == 1 + len(REPAIRS) - 1


def test_placebo_outcomes_are_recorded_and_flagged_for_analysis():
    """Placebos are useless unless the analysis can separate them from
    real repairs, so the flag has to survive into the record."""
    records = run_natural_failure_audit([_item("i1")], FakeRunner())
    placebo_ids = {r.repair_id for r in PLACEBO_REPAIRS}
    for rid, outcome in records[0].repairs.items():
        assert outcome["is_placebo"] == (rid in placebo_ids)


def test_confidence_delta_is_recorded_even_when_no_flip_occurs():
    """The whole reason for step 2: a repair that moves the model toward
    the right answer without crossing the decision boundary is real
    evidence, and a binary flip flag throws it away."""
    runner = FakeRunner(clean_gold_prob=0.10)
    record = run_natural_failure_audit([_item("i1")], runner)[0]

    assert record.clean_gold_prob_mass == 0.10
    cot = record.repairs["chain_of_thought"]
    assert cot["flipped"] is False
    assert abs(cot["gold_prob_mass"] - 0.15) < 1e-9
    assert abs(cot["delta_gold_prob"] - 0.05) < 1e-9


def test_confidence_delta_is_large_when_a_repair_flips():
    runner = FakeRunner(flip_repairs={"chain_of_thought": {"i1"}}, clean_gold_prob=0.10)
    record = run_natural_failure_audit([_item("i1")], runner)[0]
    assert record.repairs["chain_of_thought"]["delta_gold_prob"] > 0.5


def test_clean_confidence_is_recorded():
    record = run_natural_failure_audit([_item("i1")], FakeRunner())[0]
    assert record.clean_gen_confidence == 0.7


def test_noop_repair_is_marked_unapplied_and_costs_no_model_call():
    """The native-language repair silently falls back to the English
    question when no native text exists. Counting that as 'the language
    repair failed' would understate its true success rate on the items
    where it was actually tested -- and would waste a GPU call re-asking
    the identical question."""
    runner = FakeRunner()
    record = run_natural_failure_audit([_item("i1")], runner)[0]

    outcome = record.repairs["ask_in_native_language"]
    assert outcome["applied"] is False
    assert math.isnan(outcome["delta_gold_prob"])
    assert not any("ask_in_native_language" in c[0] for c in runner.calls)


def test_native_language_repair_is_applied_when_native_text_exists():
    runner = FakeRunner()
    record = run_natural_failure_audit([_item("i1", native_question="native q?")], runner)[0]

    assert record.repairs["ask_in_native_language"]["applied"] is True
    call = next(c for c in runner.calls if c[0].endswith("::ask_in_native_language"))
    assert call[1] == "native q?"


def test_visual_repair_passes_an_image_override_and_a_raised_resolution_cap():
    """Both halves matter: without the override the image is unchanged,
    and without the raised cap the runner downsamples the upscale away."""
    class Img:
        size = (800, 600)

        def resize(self, size, resample=None):
            out = Img()
            out.size = size
            return out

    runner = FakeRunner()
    run_natural_failure_audit([_item("i1", pil_image=Img())], runner)

    call = next(c for c in runner.calls if c[0].endswith("::enhance_visual_detail"))
    _, _, image_override, max_side = call
    assert image_override is not None
    assert max(image_override.size) == 1152
    assert max_side == 1152


def test_text_only_repairs_pass_no_image_override():
    runner = FakeRunner()
    run_natural_failure_audit([_item("i1")], runner)
    for label, _, image_override, _ in runner.calls:
        if not label.endswith("::enhance_visual_detail"):
            assert image_override is None


def test_a_custom_repair_set_can_be_injected():
    """Lets a cheap debug run use one repair without editing the
    registry -- and keeps the audit decoupled from the global set."""
    only_cot = tuple(r for r in REPAIRS if r.repair_id == "chain_of_thought")
    runner = FakeRunner()
    record = run_natural_failure_audit([_item("i1")], runner, repairs=only_cot)
    assert set(record[0].repairs) == {"chain_of_thought"}


def test_callbacks_fire_for_checkpointing():
    recorded, skipped = [], []

    run_natural_failure_audit(
        [_item("i1")], FakeRunner(), on_record=recorded.append, on_skip_correct=skipped.append
    )
    assert [r.item_id for r in recorded] == ["i1"]
    assert skipped == []

    run_natural_failure_audit(
        [_item("i2")], AlwaysCorrectRunner(), on_record=recorded.append, on_skip_correct=skipped.append
    )
    assert len(recorded) == 1  # unchanged -- i2 was correct, not recorded
    assert [it.item_id for it in skipped] == ["i2"]


def test_one_failing_item_does_not_sink_the_audit():
    class FlakyRunner(FakeRunner):
        def answer_with_question_override(self, item, question_text, run_label, **kw):
            if item.item_id == "bad":
                raise RuntimeError("simulated CUDA OOM")
            return super().answer_with_question_override(item, question_text, run_label, **kw)

    errors = []
    records = run_natural_failure_audit(
        [_item("bad"), _item("good")], FlakyRunner(), on_error=lambda i, e: errors.append(i.item_id)
    )
    assert [r.item_id for r in records] == ["good"]
    assert errors == ["bad"]


def test_repair_with_neither_function_is_treated_as_unapplied():
    """Guards the degenerate registry entry: a Repair with no question_fn
    and no image_fn changes nothing, and must not be silently recorded as
    a tested-and-failed repair."""
    empty = (Repair(repair_id="noop", axis="none", hypothesis="nothing"),)
    runner = FakeRunner()
    record = run_natural_failure_audit([_item("i1")], runner, repairs=empty)[0]
    assert record.repairs["noop"]["applied"] is False
    assert len(runner.calls) == 1  # clean only
