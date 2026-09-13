"""Natural-failure audit (proposal §7) -- the RQ4 headline experiment.

Unlike the rest of the pipeline (which measures causes of failures WE
injected), this measures whether we can diagnose a mistake the model made
on its own, with nothing altered. There is no injected factor to remove,
so attribution is validated constructively: each repair operator embodies
a hypothesis about the cause, and if it flips a wrong answer to correct,
that hypothesis has causal evidence for this specific failure.

Design note -- why every repair is applied to every failure rather than
classifying the failure first and applying a matched fix: a classifier
for natural failures would need ground-truth cause labels for natural
failures, which do not exist (that absence IS the research problem).
Applying all repairs and letting whichever one works define the label
keeps the causal direction the proposal rests on -- interventions
establish the truth, predictions only approximate it. A predictor of
"which repair will work" can be trained afterwards, against labels this
audit produces.

Each record captures both the binary flip AND the continuous change in
the model's probability mass on the gold answer. Flips alone are a very
lossy readout: a repair that moves the gold answer from hopeless to
nearly-chosen is real evidence about the cause, but scores identically
to one that changed nothing. The deltas are what make partial effects
and placebo comparisons measurable.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.repair_operators import REPAIRS, Repair


@dataclass
class RepairOutcome:
    """What one repair did to one naturally-wrong item."""

    repair_id: str
    axis: str
    is_placebo: bool
    correct: bool
    flipped: bool
    trace: str
    gold_prob_mass: float
    delta_gold_prob: float
    """gold_prob_mass under this repair minus gold_prob_mass on the clean
    run -- positive means the repair moved the model toward the right
    answer even if it did not flip it."""
    gen_confidence: float
    applied: bool = True
    """False when the repair was a no-op for this item (e.g. the native-
    language repair on an item with no native question). Such outcomes
    must be excluded from that repair's success rate rather than counted
    as failures -- the repair was never actually tested."""


@dataclass
class NaturalFailureRecord:
    item_id: str
    clean_correct: bool
    clean_trace: str
    clean_gold_prob_mass: float = 0.0
    clean_gen_confidence: float = float("nan")
    repairs: dict = field(default_factory=dict)  # repair_id -> RepairOutcome as dict


def _extract(result) -> tuple[bool, str, float, float]:
    correct = bool(result.correct_flags[0])
    trace = result.cot_traces[0]
    gold_prob = float(result.gold_prob_mass)
    confidence = float(result.extra_signals.get("gen_confidence", math.nan))
    return correct, trace, gold_prob, confidence


def _apply_repair(repair: Repair, item: Item) -> tuple[str, object, bool]:
    """Resolve a repair into (question_text, image_override, applied).

    `applied` is False when the repair produced no actual change -- which
    happens for `ask_in_native_language` on items whose native question
    is missing or identical to the English one. Running the model anyway
    is wasteful and, worse, silently pollutes that repair's success rate
    with items where it was never tested.
    """
    question = repair.question_fn(item) if repair.question_fn is not None else item.question
    image = None
    if repair.image_fn is not None:
        image = repair.image_fn(item.metadata.get("pil_image"))
    applied = (question != item.question) or (image is not None)
    return question, image, applied


def run_natural_failure_audit(
    items: list[Item],
    runner,
    repairs: Sequence[Repair] = REPAIRS,
    on_record: Callable[[NaturalFailureRecord], None] | None = None,
    on_skip_correct: Callable[[Item], None] | None = None,
    on_error: Callable[[Item, Exception], None] | None = None,
) -> list[NaturalFailureRecord]:
    """For each item: answer the clean (unmodified) question. If wrong,
    try every repair and record whether it flipped the answer to correct
    and how far it moved the gold-answer probability. Items the model
    already gets right are skipped -- they're not natural failures,
    there's nothing to diagnose.

    `runner` must provide `answer_with_question_override(item,
    question_text, run_label, image_override=None) -> RunResult`
    (HFVisionLanguageRunner does; MockModelRunner does not -- this audit
    is real-model-only by design).

    `on_record` (optional) is called with each NaturalFailureRecord as
    soon as it's produced -- lets a long-running real-GPU caller
    checkpoint to disk incrementally instead of only getting the full
    list at the very end. `on_skip_correct` (optional) is called for
    every item skipped because the model already got it right, for
    progress reporting.

    A single item's model call raising (e.g. a persistent CUDA OOM even
    after the runner's internal retry) is caught and reported via
    `on_error` rather than aborting the whole audit -- one bad image
    should not cost every item after it (the lesson from the main
    pipeline's first full-scale run).
    """
    records = []
    for item in items:
        try:
            clean_result = runner.answer_with_question_override(
                item, item.question, run_label=f"{item.item_id}::clean"
            )
            clean_correct, clean_trace, clean_gold_prob, clean_confidence = _extract(clean_result)

            if clean_correct:
                if on_skip_correct is not None:
                    on_skip_correct(item)
                continue

            repair_outcomes = {}
            for repair in repairs:
                question, image_override, applied = _apply_repair(repair, item)
                if not applied:
                    repair_outcomes[repair.repair_id] = _unapplied_outcome(repair)
                    continue
                repaired_result = runner.answer_with_question_override(
                    item,
                    question,
                    run_label=f"{item.item_id}::{repair.repair_id}",
                    image_override=image_override,
                    max_image_side=repair.max_image_side,
                )
                correct, trace, gold_prob, confidence = _extract(repaired_result)
                repair_outcomes[repair.repair_id] = {
                    "repair_id": repair.repair_id,
                    "axis": repair.axis,
                    "is_placebo": repair.is_placebo,
                    "correct": correct,
                    "flipped": correct and not clean_correct,
                    "trace": trace,
                    "gold_prob_mass": gold_prob,
                    "delta_gold_prob": gold_prob - clean_gold_prob,
                    "gen_confidence": confidence,
                    "applied": True,
                }

            record = NaturalFailureRecord(
                item_id=item.item_id,
                clean_correct=clean_correct,
                clean_trace=clean_trace,
                clean_gold_prob_mass=clean_gold_prob,
                clean_gen_confidence=clean_confidence,
                repairs=repair_outcomes,
            )
            records.append(record)
            if on_record is not None:
                on_record(record)
        except Exception as exc:  # noqa: BLE001 -- one bad item must not sink the audit
            if on_error is not None:
                on_error(item, exc)
    return records


def _unapplied_outcome(repair: Repair) -> dict:
    return {
        "repair_id": repair.repair_id,
        "axis": repair.axis,
        "is_placebo": repair.is_placebo,
        "correct": False,
        "flipped": False,
        "trace": "",
        "gold_prob_mass": float("nan"),
        "delta_gold_prob": float("nan"),
        "gen_confidence": float("nan"),
        "applied": False,
    }
