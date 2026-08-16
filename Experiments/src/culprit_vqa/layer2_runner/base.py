"""ModelRunner interface (proposal §6.3).

Any real backend (open HF model with traces/hidden states, or a closed
API model, behavioral-only) implements this same interface so Layer 3a/3b
code never needs to know which kind of model produced a RunResult.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition


@dataclass
class RunResult:
    """The output of running one model on one (item, condition) pair."""

    item_id: str
    condition_id: str
    sampled_answers: list[str]
    correct_flags: list[bool]
    cot_traces: list[str]
    gold_prob_mass: float
    hidden_states: "np.ndarray | None" = None
    extra_signals: dict = field(default_factory=dict)
    """Lightweight runner-computed scalar diagnostics (e.g. a generation
    confidence proxy for S5), keyed by signal name. Not the same as
    Layer 3b's SignalVector -- these are raw per-condition numbers a
    Signal implementation can read, not the final signal vector itself."""


class ModelRunner(ABC):
    """Common interface for real and mock model backends."""

    @abstractmethod
    def run(self, item: Item, condition: Condition, n_decodes: int = 8) -> RunResult:
        """Run the model on `item` under `condition`, returning n_decodes
        sampled answers plus the gold-answer probability mass."""
        raise NotImplementedError
