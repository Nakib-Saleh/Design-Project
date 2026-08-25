"""The 4-pillar tree, retained as a presentation/mapping view only (proposal §4.2).

Claims and measurements are made at the *factor* level everywhere else in
this codebase. `pillar_for_factor` and `LEAF_TO_FACTOR_TABLE` exist purely
for communication and for mapping to prior-benchmark terminology — never
import them into Layer 1/3a/3b/4/5 causal computation.
"""

from enum import Enum

from culprit_vqa.layer0_taxonomy.axes import Culturality, Modality
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates
from culprit_vqa.layer0_taxonomy.axes import Conflict, Relevance


class Pillar(str, Enum):
    VISUAL = "visual_pillar"
    TEXTUAL = "textual_pillar"
    CROSS_MODAL = "cross_modal_pillar"
    CULTURAL = "cultural_pillar"


_MODALITY_TO_PILLAR = {
    Modality.VISUAL: Pillar.VISUAL,
    Modality.TEXTUAL: Pillar.TEXTUAL,
    Modality.CROSS_MODAL: Pillar.CROSS_MODAL,
}


def pillar_for_factor(factor: Factor) -> Pillar:
    """Map a factor to one of the 4 legacy pillars.

    Any culturally-loaded factor is presented under the Cultural pillar
    regardless of its modality — this is exactly the "cultural
    substitution is not a distinct leaf, it's textual x contradictory x
    culturally-loaded" resolution from proposal §4.1: the pillar view
    routes it to "Cultural" for communication, but the factor coordinates
    (used for all real computation) still carry the full modality info.
    """
    if factor.coordinates.culturality is Culturality.CULTURALLY_LOADED:
        return Pillar.CULTURAL
    return _MODALITY_TO_PILLAR[factor.coordinates.modality]


# Named legacy leaves -> their factor coordinates, per proposal §4.2's worked
# example: "salience manipulation" = visual x irrelevant x consistent x neutral.
LEAF_TO_FACTOR_TABLE: dict[str, FactorCoordinates] = {
    "salience_manipulation": FactorCoordinates(
        Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
    ),
    "textual_distraction": FactorCoordinates(
        Modality.TEXTUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
    ),
    "cross_modal_conflict": FactorCoordinates(
        Modality.CROSS_MODAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.NEUTRAL
    ),
    "cultural_substitution": FactorCoordinates(
        Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED
    ),
}
