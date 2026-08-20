from culprit_vqa.layer0_taxonomy.axes import Conflict, Culturality, Modality, Relevance
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates
from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer0_taxonomy.pillar_mapping import (
    LEAF_TO_FACTOR_TABLE,
    Pillar,
    pillar_for_factor,
)


def test_culturally_loaded_factor_maps_to_cultural_pillar_regardless_of_modality():
    visual_cultural = Factor(
        id="visual_cultural",
        coordinates=FactorCoordinates(
            Modality.VISUAL, Relevance.RELEVANT, Conflict.CONSISTENT, Culturality.CULTURALLY_LOADED
        ),
    )
    textual_cultural = get_operator("western_default_substitution")
    assert pillar_for_factor(visual_cultural) == Pillar.CULTURAL
    assert pillar_for_factor(textual_cultural) == Pillar.CULTURAL


def test_non_cultural_factor_maps_by_modality():
    visual = get_operator("diffusion_irrelevant_object")
    textual = get_operator("irrelevant_plausible_fact")
    cross_modal = get_operator("contradictory_caption")
    assert pillar_for_factor(visual) == Pillar.VISUAL
    assert pillar_for_factor(textual) == Pillar.TEXTUAL
    assert pillar_for_factor(cross_modal) == Pillar.CROSS_MODAL


def test_salience_manipulation_leaf_matches_proposal_worked_example():
    # proposal §4.2: "salience manipulation" = visual x irrelevant x consistent x neutral
    coords = LEAF_TO_FACTOR_TABLE["salience_manipulation"]
    assert coords == FactorCoordinates(
        Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
    )
