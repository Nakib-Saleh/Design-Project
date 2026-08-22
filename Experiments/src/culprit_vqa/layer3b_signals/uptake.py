"""S4 uptake — did the model actually take the distractor's bait?
(DRR/HFR style, proposal §6.5, Distract-Bench recipe.)

Two implementations of the same construct, chosen automatically:

* Preferred (real runs): distractor response rate. The model was CAPTURED
  if the perturbation made the distractor option win when it was not
  already winning. This is the literature's DRR and needs no trace text,
  which matters because letter-scored runs emit a single token.
* Fallback (mock runner / trace-producing runs): does the trace mention a
  distractor keyword. Kept so the synthetic pipeline and its tests keep
  measuring something real.
"""

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3b_signals.option_evidence import (
    distractor_option_indices,
    factor_ids_from_condition_id,
    predicted_index,
)

# Re-exported: several modules and tests import this from here.
_factor_ids_from_condition_id = factor_ids_from_condition_id


def distractor_keywords_for_condition(item: Item, condition_id: str) -> list[str]:
    keywords: list[str] = []
    for factor_id in factor_ids_from_condition_id(item.item_id, condition_id):
        effect = item.perturbation_effects.get(factor_id)
        if effect is not None:
            keywords.extend(effect.distractor_keywords)
    return keywords


class UptakeSignal:
    name = "s4_uptake"

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float:
        targets = distractor_option_indices(item, perturbed.condition_id)
        pred_pert = predicted_index(perturbed)
        if targets and pred_pert is not None:
            pred_clean = predicted_index(clean)
            captured = pred_pert in targets and pred_clean not in targets
            return 1.0 if captured else 0.0

        # Trace fallback.
        keywords = distractor_keywords_for_condition(item, perturbed.condition_id)
        if not keywords or not perturbed.cot_traces:
            return 0.0
        trace = perturbed.cot_traces[0].lower()
        return 1.0 if any(kw.lower() in trace for kw in keywords) else 0.0
