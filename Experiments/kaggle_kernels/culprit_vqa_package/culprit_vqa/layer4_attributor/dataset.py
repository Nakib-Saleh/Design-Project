"""Training examples for the amortized attributor (proposal §6.6).

Critically, the label is the MEASURED Shapley effect (`phi_m > 0`), never
"was factor m injected" — this is the scaffold-level enforcement of the
v1 -> v2 fix (proposal §14): a perturbation that was injected but caused
nothing must get weight ~= 0, not a positive label.
"""

from dataclasses import dataclass

import numpy as np

from culprit_vqa.layer1_intervention.lattice import make_condition_id
from culprit_vqa.pipeline import PipelineRecord


@dataclass
class AttributorExample:
    item_id: str
    factor_id: str
    features: dict[str, float]
    phi: float
    is_harmful: bool


def build_examples(records: list[PipelineRecord]) -> list[AttributorExample]:
    """For each item, for each assigned factor m, pair the control-vs-{m}
    signal vector with the measured phi_m and the is_harmful=(phi_m>0) label."""
    examples: list[AttributorExample] = []
    for record in records:
        for factor in record.factors:
            condition_id = make_condition_id(record.item.item_id, frozenset({factor}))
            signal_vector = record.signal_vectors.get(condition_id)
            if signal_vector is None:
                continue
            phi_m = record.phi.get(factor, 0.0)
            examples.append(
                AttributorExample(
                    item_id=record.item.item_id,
                    factor_id=factor.id,
                    features=dict(signal_vector.values),
                    phi=phi_m,
                    is_harmful=phi_m > 0.0,
                )
            )
    return examples


def to_arrays(
    examples: list[AttributorExample], feature_names: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, feature_names) with a fixed, sorted feature-name
    ordering (sorted from the examples themselves unless `feature_names`
    is given, so predict-time arrays align with the fit-time ordering)."""
    if feature_names is None:
        names: set[str] = set()
        for ex in examples:
            names.update(ex.features.keys())
        feature_names = sorted(names)

    def _row(ex: AttributorExample) -> list[float]:
        row = []
        for name in feature_names:
            value = ex.features.get(name, 0.0)
            row.append(0.0 if (value is None or value != value) else value)  # NaN-safe (S5)
        return row

    X = np.array([_row(ex) for ex in examples], dtype=float) if examples else np.zeros((0, len(feature_names)))
    y = np.array([1 if ex.is_harmful else 0 for ex in examples], dtype=int)
    return X, y, feature_names
