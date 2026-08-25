"""Thin stub signals — S2, S3, S5, S6 (proposal §6.5).

Each is a real, callable minimal implementation (or an honest constant
placeholder), never used as causal evidence. Every causal claim routes
through Layer 3a's interventions; these are amortization features only.
"""

import math

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3b_signals.option_evidence import (
    conflict,
    distractor_option_indices,
    mass_on,
    option_probs,
)
from culprit_vqa.layer3b_signals.uptake import distractor_keywords_for_condition


class AttrRatioSignal:
    """S2 — how much probability the perturbation moved ONTO the distractor.

    Measured as p(distractor | perturbed) - p(distractor | clean), so it
    is a change attributable to the intervention rather than a level that
    could reflect the item's own difficulty. Positive means the cue
    pulled belief toward the option it was advertising.

    Gold-blind by construction: it reads only the distractor's own
    probability, never the correct answer's. That is what keeps Layer 4
    from trivially reconstructing phi, which is a function of p(gold).

    Falls back to the original trace token-overlap ratio when the runner
    provides no option distribution (mock runner), and returns 0.0 for
    factors with no option-level target -- a salience crop advertises no
    particular answer, and inventing a number there would be noise.
    """

    name = "s2_attr_ratio"

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float:
        targets = distractor_option_indices(item, perturbed.condition_id)
        probs_pert = option_probs(perturbed)
        if targets and probs_pert is not None:
            return mass_on(probs_pert, targets) - mass_on(option_probs(clean), targets)

        if not perturbed.cot_traces:
            return 0.0
        trace_tokens = perturbed.cot_traces[0].lower().split()
        if not trace_tokens:
            return 0.0
        keywords = {kw.lower() for kw in distractor_keywords_for_condition(item, perturbed.condition_id)}
        if not keywords:
            return 0.0
        hits = sum(1 for tok in trace_tokens if tok in keywords)
        return hits / len(trace_tokens)


class NLIConflictSignal:
    """S3 — how much the intervention destabilized the model's belief.

    The construct in the proposal is contradiction between what the image
    supports and what the intervention asserts. With no NLI backbone
    available on the run environment (and nothing to run it on, since
    letter scoring emits no claim text), this measures the observable
    footprint that contradiction leaves: the model stops having a clear
    winner and becomes torn between two options.

    Reported as the CHANGE in top-2 conflict, perturbed minus clean, so a
    naturally ambiguous item does not read as high conflict on its own.
    Applies to every factor, including ones with no option-level target,
    which makes it the only one of S2/S3/S4 that is live for a salience
    crop.

    Gold-blind: the top-2 margin does not depend on which option is
    correct.

    TODO(thesis): with a real NLI backbone and generated rationales, swap
    this for an actual contradiction probability and compare the two.
    """

    name = "s3_nli_conflict"

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float:
        probs_pert = option_probs(perturbed)
        if probs_pert is None:
            return 0.0
        return conflict(probs_pert) - conflict(option_probs(clean))


class ProbeSignal:
    """S5 — approximated here as a generation-confidence proxy, NOT a
    trained linear probe on hidden states (a real probe needs its own
    labeled training set, which would be circular with the experiment
    this signal feeds into). The runner computes the model's own average
    top-token probability while generating its answer and stores it in
    `RunResult.extra_signals["gen_confidence"]` -- this signal just reads
    it back.

    Still auxiliary diagnostic ONLY per proposal §6.5 -- decodability/
    confidence != utilization, and this signal must never be used as
    causal evidence. Returns NaN whenever the runner didn't provide it
    (always true for MockModelRunner).

    TODO(thesis): replace with an actual trained linear probe on
    mid-layer hidden states once a labeled training set exists.
    """

    name = "s5_probe_aux"

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float:
        return perturbed.extra_signals.get("gen_confidence", math.nan)


class LanguageDeltaSignal:
    """S6 — local-vs-English answer-confidence difference (MMAC-style).

    Real signal: computed once per item (not per condition, since it's a
    property of the item/model pair, not of a specific perturbation) by
    running the model on both the native-language and English-translated
    clean question and comparing gold-answer probability mass. The
    orchestration script stores the result in
    `item.metadata["language_delta"]` before the pipeline runs; this
    signal just reads it back. Returns 0.0 if not precomputed (e.g. no
    native-language text available for this item).
    """

    name = "s6_language_delta"

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float:
        return item.metadata.get("language_delta", 0.0)
