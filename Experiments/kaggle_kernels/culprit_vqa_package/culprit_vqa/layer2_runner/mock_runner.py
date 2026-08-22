"""Deterministic mock model backend for synthetic/small-scale testing.

Correctness probability starts at a base rate and is decremented by each
applied factor's configured "harm", plus an optional extra decrement (or
increment, if negative) when specific factor pairs co-occur — this
`interaction_bonus` knob is what lets a test engineer a case where
leave-one-out misattributes credit but Shapley does not (the RQ1
"killer experiment" at toy scale, proposal §6.4).
"""

import hashlib

import numpy as np

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition
from culprit_vqa.layer2_runner.base import ModelRunner, RunResult

DEFAULT_BASE_CORRECTNESS = 0.9


class MockModelRunner(ModelRunner):
    """A fake model whose correctness probability is a deterministic
    function of which factors are applied to a condition.

    Deterministic per (item_id, condition_id, seed): repeated calls with
    the same condition give bit-identical results. This matters because
    exact Shapley computation over p(S) is only meaningful/testable when
    there is no re-sampling noise between calls to the same condition.
    """

    def __init__(
        self,
        seed: int = 0,
        base_correctness: float = DEFAULT_BASE_CORRECTNESS,
        factor_harm: dict[str, float] | None = None,
        interaction_bonus: dict[frozenset[str], float] | None = None,
    ):
        self.seed = seed
        self.base_correctness = base_correctness
        self.factor_harm = factor_harm or {}
        self.interaction_bonus = interaction_bonus or {}

    def _derive_seed(self, item_id: str, condition_id: str) -> int:
        digest = hashlib.sha256(f"{self.seed}::{item_id}::{condition_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def _gold_prob_mass(self, condition: Condition) -> float:
        p = self.base_correctness
        for factor in condition.applied_factors:
            p -= self.factor_harm.get(factor.id, 0.0)
        applied_ids = frozenset(f.id for f in condition.applied_factors)
        for pair, bonus in self.interaction_bonus.items():
            if pair <= applied_ids:
                p -= bonus
        return float(np.clip(p, 0.0, 1.0))

    def _make_trace(self, item: Item, condition: Condition, correct: bool) -> str:
        if correct:
            return f"considering the image and question, the answer is {item.answer}"
        distractor_words = []
        for factor in condition.applied_factors:
            effect = item.perturbation_effects.get(factor.id)
            if effect is not None:
                distractor_words.extend(effect.distractor_keywords)
        mention = " ".join(distractor_words) if distractor_words else "an unrelated detail"
        return f"noticing {mention}, the answer is wrong"

    def run(self, item: Item, condition: Condition, n_decodes: int = 8) -> RunResult:
        p = self._gold_prob_mass(condition)
        rng = np.random.default_rng(self._derive_seed(item.item_id, condition.condition_id))
        correct_flags = list(rng.random(n_decodes) < p)
        sampled_answers = [item.answer if c else "wrong" for c in correct_flags]
        cot_traces = [self._make_trace(item, condition, c) for c in correct_flags]
        return RunResult(
            item_id=item.item_id,
            condition_id=condition.condition_id,
            sampled_answers=sampled_answers,
            correct_flags=correct_flags,
            cot_traces=cot_traces,
            gold_prob_mass=p,
            hidden_states=None,
        )
