"""Signal vector and Signal protocol (proposal §6.5).

Traces are behavioral evidence about what the model expressed, never
ground truth about what caused the answer — all causal claims route
through Layer 3a. Signals here are amortization features only.
"""

from dataclasses import dataclass
from typing import Protocol

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer2_runner.base import RunResult


@dataclass
class SignalVector:
    item_id: str
    condition_id: str
    values: dict[str, float]


class Signal(Protocol):
    name: str

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float: ...
