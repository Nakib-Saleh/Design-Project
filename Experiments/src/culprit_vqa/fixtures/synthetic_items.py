"""Five hand-built synthetic items exercising the pipeline's corners:

1. `item_baseline_k1`    — k=1 baseline, single harmful visual factor.
2. `item_interaction_k2` — k=2, engineered interaction (RQ1 toy killer experiment).
3. `item_full_k3`        — k=3, full 8-condition lattice.
4. `item_null_player_k1` — k=1, a factor that causes nothing (pure null player).
5. `item_language_k2`    — k=2, culturally-loaded pair with a parallel-language pointer.

Also exports the `factor_harm` / `interaction_bonus` configuration used to
build a `MockModelRunner` consistent with these items, and a convenience
`build_synthetic_runner()`.
"""

from culprit_vqa.layer0_taxonomy.factors import Factor
from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import BBox, EvidenceRegion, Item, PerturbationEffect
from culprit_vqa.layer2_runner.mock_runner import MockModelRunner

# Evidence sits in the top-left quadrant; visual perturbations below are
# placed in the bottom-right quadrant so they never overlap it (IoU = 0).
_EVIDENCE_BBOX = BBox(0.1, 0.1, 0.4, 0.4)
_FAR_BBOX = BBox(0.6, 0.6, 0.9, 0.9)

SYNTHETIC_FACTOR_HARM: dict[str, float] = {
    "diffusion_irrelevant_object": 0.4,
    "wrong_local_entity": 0.05,
    "contradictory_caption": 0.05,
    "irrelevant_plausible_fact": 0.15,
    "text_overlay_wrong_answer": 0.2,
    "salience_recomposition": 0.0,  # pure null player — see item 4
    "western_default_substitution": 0.25,
    "biased_prior_phrasing": 0.2,
}

SYNTHETIC_INTERACTION_BONUS: dict[frozenset[str], float] = {
    frozenset({"wrong_local_entity", "contradictory_caption"}): 0.4,
}


def _item_baseline_k1() -> tuple[Item, list[Factor]]:
    factor = get_operator("diffusion_irrelevant_object")
    item = Item(
        item_id="item_baseline_k1",
        image_ref="synthetic://baseline",
        question="What fruit is in the image?",
        answer="mango",
        evidence_region=EvidenceRegion(_EVIDENCE_BBOX, "the fruit"),
        language="en",
        perturbation_effects={
            factor.id: PerturbationEffect(
                touched_bbox=_FAR_BBOX,
                perturbed_answer="mango",
                distractor_keywords=("bicycle",),
            )
        },
    )
    return item, [factor]


def _item_interaction_k2() -> tuple[Item, list[Factor]]:
    f1 = get_operator("wrong_local_entity")
    f2 = get_operator("contradictory_caption")
    item = Item(
        item_id="item_interaction_k2",
        image_ref="synthetic://interaction",
        question="Which festival is depicted?",
        answer="pohela boishakh",
        evidence_region=EvidenceRegion(_EVIDENCE_BBOX, "the festival scene"),
        language="bn",
        perturbation_effects={
            f1.id: PerturbationEffect(
                perturbed_answer="pohela boishakh",
                distractor_keywords=("diwali",),
            ),
            f2.id: PerturbationEffect(
                touched_bbox=_FAR_BBOX,
                perturbed_answer="pohela boishakh",
                distractor_keywords=("unrelated caption",),
            ),
        },
    )
    return item, [f1, f2]


def _item_full_k3() -> tuple[Item, list[Factor]]:
    f1 = get_operator("diffusion_irrelevant_object")
    f2 = get_operator("irrelevant_plausible_fact")
    f3 = get_operator("text_overlay_wrong_answer")
    item = Item(
        item_id="item_full_k3",
        image_ref="synthetic://full_k3",
        question="What instrument is being played?",
        answer="sitar",
        evidence_region=EvidenceRegion(_EVIDENCE_BBOX, "the instrument"),
        language="hi",
        perturbation_effects={
            f1.id: PerturbationEffect(
                touched_bbox=_FAR_BBOX,
                perturbed_answer="sitar",
                distractor_keywords=("lamp",),
            ),
            f2.id: PerturbationEffect(
                perturbed_answer="sitar",
                distractor_keywords=("unrelated fact",),
            ),
            f3.id: PerturbationEffect(
                touched_bbox=_FAR_BBOX,
                perturbed_answer="sitar",
                distractor_keywords=("guitar",),
            ),
        },
    )
    return item, [f1, f2, f3]


def _item_null_player_k1() -> tuple[Item, list[Factor]]:
    factor = get_operator("salience_recomposition")
    item = Item(
        item_id="item_null_player_k1",
        image_ref="synthetic://null_player",
        question="What animal is shown?",
        answer="elephant",
        evidence_region=EvidenceRegion(_EVIDENCE_BBOX, "the animal"),
        language="en",
        perturbation_effects={
            factor.id: PerturbationEffect(
                touched_bbox=_FAR_BBOX,
                perturbed_answer="elephant",
                distractor_keywords=("shadow",),
            )
        },
    )
    return item, [factor]


def _item_language_k2() -> tuple[Item, list[Factor]]:
    f1 = get_operator("western_default_substitution")
    f2 = get_operator("biased_prior_phrasing")
    item = Item(
        item_id="item_language_k2",
        image_ref="synthetic://language",
        question="Ei khabarer nam ki?",
        answer="biryani",
        evidence_region=EvidenceRegion(_EVIDENCE_BBOX, "the dish"),
        language="bn",
        perturbation_effects={
            f1.id: PerturbationEffect(
                perturbed_answer="biryani",
                distractor_keywords=("pizza",),
            ),
            f2.id: PerturbationEffect(
                perturbed_answer="biryani",
                distractor_keywords=("western dish",),
            ),
        },
        metadata={"parallel_item_id": "item_language_k2_en"},
    )
    return item, [f1, f2]


def build_synthetic_items() -> list[tuple[Item, list[Factor]]]:
    """The 5 hand-built (Item, assigned factors) fixtures."""
    return [
        _item_baseline_k1(),
        _item_interaction_k2(),
        _item_full_k3(),
        _item_null_player_k1(),
        _item_language_k2(),
    ]


def build_synthetic_runner(seed: int = 0) -> MockModelRunner:
    """A MockModelRunner configured consistently with `build_synthetic_items`."""
    return MockModelRunner(
        seed=seed,
        factor_harm=SYNTHETIC_FACTOR_HARM,
        interaction_bonus=SYNTHETIC_INTERACTION_BONUS,
    )
