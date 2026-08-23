"""Shared helpers for the option-distribution behavioral signals.

Why these exist
---------------
The first real attribution run produced three constant-zero signals
(S2 attr-ratio, S3 NLI-conflict, S4 uptake), which is most of the reason
the amortized attributor reached only rho=0.49 against a 0.70 target.
There were two independent causes:

1. `Item.perturbation_effects` was never populated on real CVQA items, so
   the signals had no record of what each factor injected.
2. Under letter scoring the model emits a SINGLE TOKEN ("A"), so every
   trace-text signal is structurally dead -- there are no words in a
   one-letter trace for a keyword to match. This is not fixable by
   populating keywords; it needs a different observable.

The observable used here is the per-option probability distribution the
runner already computes and stores in `RunResult.extra_signals`
("option_probs", "predicted_idx"). It costs nothing extra: it is the same
forward pass the correctness measure already uses.

Non-circularity
---------------
Layer 4 must predict Shapley blame from cheap per-condition evidence, so
these signals must not smuggle in the quantity being predicted. At k=2,
phi_m is a linear combination of p(gold) across the four conditions, and
the term p(gold | {m}) alone gets you halfway there analytically. So
NONE of the functions below reads the gold index. They look only at the
distractor's own probability, at the shape of the distribution, and at
which option won -- never at how the correct answer fared.
"""

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer2_runner.base import RunResult


def factor_ids_from_condition_id(item_id: str, condition_id: str) -> list[str]:
    """Recover the applied factor ids from a condition_id built by
    `layer1_intervention.lattice.make_condition_id`."""
    suffix = condition_id[len(item_id) + 2 :]  # strip "{item_id}::"
    if suffix == "control":
        return []
    return suffix.split("+")


def distractor_option_indices(item: Item, condition_id: str) -> list[int]:
    """Option indices the applied factors actively push the model toward.

    Empty for factors with no option-level target (a salience crop pushes
    toward no particular answer). Callers must treat empty as "this
    signal does not apply here" and return 0.0, not as evidence of
    absence.
    """
    out: list[int] = []
    for factor_id in factor_ids_from_condition_id(item.item_id, condition_id):
        effect = item.perturbation_effects.get(factor_id)
        if effect is not None and effect.distractor_option_idx is not None:
            out.append(effect.distractor_option_idx)
    return out


def option_probs(result: RunResult) -> list[float] | None:
    """The normalized per-option distribution, or None when the runner
    did not provide one (MockModelRunner, or a degenerate scoring call)."""
    probs = result.extra_signals.get("option_probs")
    if not probs:
        return None
    return list(probs)


def predicted_index(result: RunResult) -> int | None:
    idx = result.extra_signals.get("predicted_idx")
    return None if idx is None else int(idx)


def mass_on(probs: list[float] | None, indices: list[int]) -> float:
    """Total probability on `indices`, ignoring out-of-range entries."""
    if not probs or not indices:
        return 0.0
    return sum(probs[i] for i in set(indices) if 0 <= i < len(probs))


def conflict(probs: list[float] | None) -> float:
    """How torn the model is, as 1 - (top1 - top2).

    1.0 means the top two options are tied (maximum conflict); 0.0 means
    one option holds all the mass. Deliberately based on the top-2 margin
    rather than full entropy: entropy over k options confounds "torn
    between two readings" with "spread thinly over many", and it is the
    two-horse race that a conflicting cue actually produces.

    Gold-blind: it never asks which option is correct.
    """
    if not probs or len(probs) < 2:
        return 0.0
    top2 = sorted(probs, reverse=True)[:2]
    return max(0.0, 1.0 - (top2[0] - top2[1]))
