"""Evidence preservation check (proposal §6.2, check 2 of 3).

The decisive visual evidence region must be untouched: IoU between a
visual perturbation's touched region and the annotated evidence region
must stay below threshold, and textual perturbations must not remove
needed information.
"""

from typing import Callable

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition

EvidenceCheckFn = Callable[[Item, Condition], bool]

DEFAULT_IOU_THRESHOLD = 0.1


def evidence_preservation_check(
    item: Item, condition: Condition, iou_threshold: float = DEFAULT_IOU_THRESHOLD
) -> bool:
    """Accept iff no applied factor violates evidence preservation.

    - Visual/cross-modal factors with a `touched_bbox`: reject if its IoU
      with `item.evidence_region.bbox` exceeds `iou_threshold`.
    - Any factor with `removes_needed_text=True`: reject outright.
    - Factors with no evidence region declared, or an item with no
      annotated evidence region, are treated as non-violating (nothing to
      check against).
    """
    for factor in condition.applied_factors:
        effect = item.perturbation_effects.get(factor.id)
        if effect is None:
            continue
        if effect.removes_needed_text:
            return False
        if effect.touched_bbox is not None and item.evidence_region is not None:
            iou = effect.touched_bbox.iou(item.evidence_region.bbox)
            if iou > iou_threshold:
                return False
    return True
