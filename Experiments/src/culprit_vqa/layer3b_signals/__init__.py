"""Layer 3b — behavioral signals, explicitly non-causal (proposal §6.5).

Traces are treated as behavioral evidence about what the model
expressed, never as ground truth about what caused the answer. All
causal claims route through Layer 3a's interventions; the signals here
exist only to amortize that expensive computation (Layer 4).

Known design tension (not solved in this scaffold): S1 (drift), S2
(attr-ratio) and S4 (uptake) are inherently *paired* — they compare a
clean RunResult against a perturbed one. Natural (non-injected) failures
have no such pair. A future `NaturalFailureSignal` variant will need a
single-pass fallback (e.g. comparing against a reconstructed "expected
clean" trace); this scaffold's `compute_signal_vector` always requires
both `clean` and `perturbed` RunResults.
"""

from culprit_vqa.layer3b_signals.base import Signal, SignalVector
from culprit_vqa.layer3b_signals.drift import DriftSignal
from culprit_vqa.layer3b_signals.stubs import (
    AttrRatioSignal,
    LanguageDeltaSignal,
    NLIConflictSignal,
    ProbeSignal,
)
from culprit_vqa.layer3b_signals.uptake import UptakeSignal
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition
from culprit_vqa.layer2_runner.base import RunResult

DEFAULT_SIGNALS: list[Signal] = [
    DriftSignal(),
    UptakeSignal(),
    AttrRatioSignal(),
    NLIConflictSignal(),
    ProbeSignal(),
    LanguageDeltaSignal(),
]


def compute_signal_vector(
    item: Item,
    condition: Condition,
    clean: RunResult,
    perturbed: RunResult,
    signals: list[Signal] = DEFAULT_SIGNALS,
) -> SignalVector:
    return SignalVector(
        item_id=item.item_id,
        condition_id=condition.condition_id,
        values={s.name: s.compute(item, clean, perturbed) for s in signals},
    )


__all__ = [
    "Signal",
    "SignalVector",
    "DriftSignal",
    "UptakeSignal",
    "AttrRatioSignal",
    "NLIConflictSignal",
    "ProbeSignal",
    "LanguageDeltaSignal",
    "DEFAULT_SIGNALS",
    "compute_signal_vector",
]
