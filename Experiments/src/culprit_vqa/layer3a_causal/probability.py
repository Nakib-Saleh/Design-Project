"""p(S) estimation from a RunResult (proposal §6.4).

Logit/probability mass on the gold answer is preferred over sampled
decode frequency: it is continuous and far lower-variance than a Bernoulli
frequency estimate over a handful of decodes.
"""

from typing import Callable, Literal

from culprit_vqa.layer0_taxonomy.factors import Factor
from culprit_vqa.layer2_runner.base import RunResult

EstimationMethod = Literal["logit_mass", "decode_frequency"]


def estimate_p(run_result: RunResult, method: EstimationMethod = "logit_mass") -> float:
    """Estimate p(S) = P(correct | condition) from a RunResult.

    - "logit_mass" (default, preferred per proposal §6.4): the model's
      probability mass on the gold answer.
    - "decode_frequency": fraction of sampled decodes that were correct.
    """
    if method == "logit_mass":
        return run_result.gold_prob_mass
    if method == "decode_frequency":
        if not run_result.correct_flags:
            return 0.0
        return sum(run_result.correct_flags) / len(run_result.correct_flags)
    raise ValueError(f"Unknown estimation method: {method!r}")


def build_p_function(
    results_by_subset: dict[frozenset[Factor], RunResult],
    method: EstimationMethod = "logit_mass",
) -> Callable[[frozenset[Factor]], float]:
    """Build a p(S) function closed over a fully-populated lattice's
    RunResults, for use with `shapley_values` / `leave_one_out` /
    `interaction_indices`."""

    def p_fn(subset: frozenset[Factor]) -> float:
        return estimate_p(results_by_subset[subset], method)

    return p_fn
