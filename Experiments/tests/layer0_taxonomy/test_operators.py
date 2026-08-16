import pytest

from culprit_vqa.layer0_taxonomy.axes import (
    Conflict,
    Culturality,
    Knowledge,
    Modality,
    Relevance,
)
from culprit_vqa.layer0_taxonomy.operators import OPERATOR_REGISTRY, get_operator, list_operators

# The exact 9 operators + axis coordinates from proposal §6.1's table.
EXPECTED_OPERATORS = {
    "diffusion_irrelevant_object": (Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL),
    "salience_recomposition": (Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL),
    "irrelevant_plausible_fact": (Modality.TEXTUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL),
    "wrong_local_entity": (Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED),
    "contradictory_caption": (Modality.CROSS_MODAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.NEUTRAL),
    "text_overlay_wrong_answer": (Modality.CROSS_MODAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.NEUTRAL),
    "western_default_substitution": (Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED),
    "biased_prior_phrasing": (Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONSISTENT, Culturality.CULTURALLY_LOADED),
    "long_tail_entity_swap": (Modality.VISUAL, Relevance.RELEVANT, Conflict.CONSISTENT, Culturality.CULTURALLY_LOADED),
}


def test_registry_has_exactly_the_nine_proposal_operators():
    assert set(OPERATOR_REGISTRY.keys()) == set(EXPECTED_OPERATORS.keys())


@pytest.mark.parametrize("operator_id", sorted(EXPECTED_OPERATORS))
def test_operator_axis_coordinates_match_proposal(operator_id):
    modality, relevance, conflict, culturality = EXPECTED_OPERATORS[operator_id]
    factor = get_operator(operator_id)
    assert factor.coordinates.modality == modality
    assert factor.coordinates.relevance == relevance
    assert factor.coordinates.conflict == conflict
    assert factor.coordinates.culturality == culturality


def test_long_tail_entity_swap_requires_knowledge():
    factor = get_operator("long_tail_entity_swap")
    assert factor.coordinates.knowledge == Knowledge.KNOWLEDGE_REQUIRED


def test_get_operator_raises_on_unknown_id():
    with pytest.raises(KeyError):
        get_operator("not_a_real_operator")


def test_list_operators_returns_all_registered_factors():
    assert {f.id for f in list_operators()} == set(OPERATOR_REGISTRY.keys())


# --- post-proposal extensions -----------------------------------------
# Added after the manipulation check found the culturally-loaded arm of
# the taxonomy had no working intervention. Kept separate from the nine
# above so the proposal's table stays verifiable as written.

def test_extensions_are_disjoint_from_the_proposal_registry():
    from culprit_vqa.layer0_taxonomy.operators import EXTENSION_OPERATORS

    assert set(EXTENSION_OPERATORS) & set(OPERATOR_REGISTRY) == set()


def test_every_extension_is_culturally_loaded():
    """The extensions exist to give the culturally-loaded axis a working
    operator; one that wasn't culturally loaded would not serve that."""
    from culprit_vqa.layer0_taxonomy.operators import EXTENSION_OPERATORS

    for factor in EXTENSION_OPERATORS.values():
        assert factor.coordinates.culturality is Culturality.CULTURALLY_LOADED


def test_get_operator_resolves_both_registries():
    from culprit_vqa.layer0_taxonomy.operators import ALL_OPERATORS, list_all_operators

    assert get_operator("cultural_text_overlay").id == "cultural_text_overlay"
    assert get_operator("salience_recomposition").id == "salience_recomposition"
    assert {f.id for f in list_all_operators()} == set(ALL_OPERATORS)


def test_list_operators_still_returns_only_the_proposal_nine():
    assert len(list_operators()) == 9


def test_every_real_operator_implementation_has_a_taxonomy_entry():
    """A factor the kernels can apply but not look up would crash the run
    at factor-resolution time, after the model is already loaded."""
    from culprit_vqa.layer1_intervention.real_operators import REAL_OPERATORS

    for factor_id in REAL_OPERATORS:
        assert get_operator(factor_id).id == factor_id
