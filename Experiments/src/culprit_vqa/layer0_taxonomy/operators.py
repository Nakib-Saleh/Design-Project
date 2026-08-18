"""The operator library — the 9 example operators from proposal §6.1,
each tagged with its 5-axis coordinates."""

from culprit_vqa.layer0_taxonomy.axes import (
    Conflict,
    Culturality,
    Knowledge,
    Modality,
    Relevance,
)
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates

OPERATOR_REGISTRY: dict[str, Factor] = {
    "diffusion_irrelevant_object": Factor(
        id="diffusion_irrelevant_object",
        coordinates=FactorCoordinates(
            Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
        ),
        description="Diffusion-inserted salient irrelevant object (Idis recipe)",
    ),
    "salience_recomposition": Factor(
        id="salience_recomposition",
        coordinates=FactorCoordinates(
            Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
        ),
        description="Salience re-composition (distractor enlarged/centered)",
    ),
    "irrelevant_plausible_fact": Factor(
        id="irrelevant_plausible_fact",
        coordinates=FactorCoordinates(
            Modality.TEXTUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
        ),
        description="Irrelevant plausible fact prepended (GSM-IC style)",
    ),
    "wrong_local_entity": Factor(
        id="wrong_local_entity",
        coordinates=FactorCoordinates(
            Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED
        ),
        description="Wrong same-category local entity mentioned",
    ),
    "contradictory_caption": Factor(
        id="contradictory_caption",
        coordinates=FactorCoordinates(
            Modality.CROSS_MODAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.NEUTRAL
        ),
        description="Contradictory caption (CLASH recipe)",
    ),
    "text_overlay_wrong_answer": Factor(
        id="text_overlay_wrong_answer",
        coordinates=FactorCoordinates(
            Modality.CROSS_MODAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.NEUTRAL
        ),
        description="Text overlay asserting wrong answer (VisualTextTrap recipe)",
    ),
    "western_default_substitution": Factor(
        id="western_default_substitution",
        coordinates=FactorCoordinates(
            Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED
        ),
        description='Western-default substitution in context ("this is pizza")',
    ),
    "biased_prior_phrasing": Factor(
        id="biased_prior_phrasing",
        coordinates=FactorCoordinates(
            Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONSISTENT, Culturality.CULTURALLY_LOADED
        ),
        description="Biased-prior phrasing, local vs English parallel",
    ),
    "long_tail_entity_swap": Factor(
        id="long_tail_entity_swap",
        coordinates=FactorCoordinates(
            Modality.VISUAL,
            Relevance.RELEVANT,
            Conflict.CONSISTENT,
            Culturality.CULTURALLY_LOADED,
            Knowledge.KNOWLEDGE_REQUIRED,
        ),
        description="Long-tail entity swap (CulturalGround)",
    ),
}


# Operators added AFTER the proposal, in response to a measured result:
# the manipulation check found `wrong_local_entity` inert (+1.1%, p=0.20,
# n=698), leaving the culturally-loaded arm of the taxonomy with no
# working intervention. These two are kept separate from
# OPERATOR_REGISTRY so the proposal's §6.1 table stays verifiable as
# written -- the extension is an empirical addition, not a retro-edit of
# the design.
#
# The other two culture operators are NOT here: they implement
# `biased_prior_phrasing` and `western_default_substitution`, which the
# proposal already declared above but which had never been built.
EXTENSION_OPERATORS: dict[str, Factor] = {
    "cultural_text_overlay": Factor(
        id="cultural_text_overlay",
        coordinates=FactorCoordinates(
            Modality.CROSS_MODAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED
        ),
        description="Wrong cultural origin asserted as an image caption "
                    "(cultural analogue of text_overlay_wrong_answer, naming no answer option)",
    ),
    "entity_swap_in_question": Factor(
        id="entity_swap_in_question",
        coordinates=FactorCoordinates(
            Modality.TEXTUAL, Relevance.RELEVANT, Conflict.CONTRADICTORY, Culturality.CULTURALLY_LOADED
        ),
        description="Named entity in the question replaced by a foreign-culture one "
                    "(replaces the referent rather than appending an aside)",
    ),
}

ALL_OPERATORS: dict[str, Factor] = {**OPERATOR_REGISTRY, **EXTENSION_OPERATORS}


def get_operator(operator_id: str) -> Factor:
    """Look up a registered operator by id, proposal operators first then
    extensions. Raises KeyError with a clear message if unknown."""
    try:
        return ALL_OPERATORS[operator_id]
    except KeyError:
        raise KeyError(
            f"Unknown operator id {operator_id!r}. "
            f"Registered operators: {sorted(ALL_OPERATORS)}"
        ) from None


def list_operators() -> list[Factor]:
    """The proposal's operators only — the canonical §6.1 table."""
    return list(OPERATOR_REGISTRY.values())


def list_all_operators() -> list[Factor]:
    """Proposal operators plus post-proposal extensions."""
    return list(ALL_OPERATORS.values())
