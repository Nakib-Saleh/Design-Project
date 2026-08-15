"""Top-level orchestration: wires Layers 1 -> 2 -> 3a -> 3b for one item.

`run_pipeline_for_item` is the backbone reused by both the end-to-end
smoke test and `scripts/demo_pipeline.py`.
"""

from dataclasses import dataclass, field

from culprit_vqa.layer0_taxonomy.factors import Factor
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition, generate_lattice
from culprit_vqa.layer1_intervention.validity import ValidityReport, run_validity_checks
from culprit_vqa.layer2_runner.base import ModelRunner, RunResult
from culprit_vqa.layer3a_causal.attribution import normalize_attribution
from culprit_vqa.layer3a_causal.probability import build_p_function
from culprit_vqa.layer3a_causal.shapley import (
    interaction_indices,
    leave_one_out,
    shapley_values,
)
from culprit_vqa.layer3b_signals import DEFAULT_SIGNALS, Signal, compute_signal_vector
from culprit_vqa.layer3b_signals.base import SignalVector


@dataclass
class PipelineRecord:
    """Everything computed for one item's factorial lattice."""

    item: Item
    factors: list[Factor]
    validity_reports: dict[str, ValidityReport] = field(default_factory=dict)
    run_results: dict[frozenset[Factor], RunResult] = field(default_factory=dict)
    phi: dict[Factor, float] = field(default_factory=dict)
    alpha: dict[Factor, float] = field(default_factory=dict)
    interactions: dict[frozenset[Factor], float] = field(default_factory=dict)
    loo: dict[Factor, float] = field(default_factory=dict)
    signal_vectors: dict[str, SignalVector] = field(default_factory=dict)


def run_pipeline_for_item(
    item: Item,
    factors: list[Factor],
    runner: ModelRunner,
    n_decodes: int = 8,
    signals: list[Signal] = DEFAULT_SIGNALS,
) -> PipelineRecord:
    """Run the full Layer 1 -> 2 -> 3a -> 3b pipeline for one item.

    1. Generate the 2^k lattice of conditions.
    2. Run the three-way validity check on every condition.
    3. Run the model on every condition.
    4. Compute Shapley values, leave-one-out, pairwise interactions, and
       normalized attribution over the lattice.
    5. Compute behavioral signal vectors comparing the control condition
       against each singleton-factor condition.
    """
    conditions: list[Condition] = generate_lattice(item, factors)

    validity_reports: dict[str, ValidityReport] = {}
    run_results: dict[frozenset[Factor], RunResult] = {}
    for condition in conditions:
        validity_reports[condition.condition_id] = run_validity_checks(item, condition)
        run_results[condition.applied_factors] = runner.run(item, condition, n_decodes=n_decodes)

    p_fn = build_p_function(run_results)
    phi = shapley_values(p_fn, factors)
    alpha = normalize_attribution(phi)
    loo = leave_one_out(p_fn, factors)
    interactions = interaction_indices(p_fn, factors)

    control_result = run_results[frozenset()]
    signal_vectors: dict[str, SignalVector] = {}
    for condition in conditions:
        if len(condition.applied_factors) != 1:
            continue
        perturbed_result = run_results[condition.applied_factors]
        signal_vectors[condition.condition_id] = compute_signal_vector(
            item, condition, control_result, perturbed_result, signals=signals
        )

    return PipelineRecord(
        item=item,
        factors=list(factors),
        validity_reports=validity_reports,
        run_results=run_results,
        phi=phi,
        alpha=alpha,
        interactions=interactions,
        loo=loo,
        signal_vectors=signal_vectors,
    )
