"""Real (non-synthetic) perturbation-application functions for a subset of
the operator library (proposal §6.1), used when running on real images
instead of hand-built fixtures.

Scope note (honest simplification vs. the full research design): visual
operators here use simple deterministic pixel compositing/cropping via
Pillow rather than a diffusion-based object-insertion pipeline — a
defensible approximation for a scaled smoke test, not the final research
instrument. Textual operators are template-based string edits. There is
no ground-truth evidence-region annotation available for real CVQA
images, so visual operators are constrained BY CONSTRUCTION to a
peripheral image region (a corner strip) rather than verified against an
annotated evidence bbox via IoU -- documented here rather than silently
assumed.

Every apply function has the uniform signature:
    apply(image: PIL.Image.Image, question: str, options: list[str],
          correct_idx: int, rng: random.Random, metadata: dict | None = None)
      -> OperatorOutcome

`metadata` carries the item's dataset fields (subset, native_question,
...) for operators that need to know the item's own culture or language.
Operators that don't need it ignore it, so the signature stays uniform.

None of these functions change which option is objectively correct, so
semantic preservation holds structurally (no perturbed_answer override is
needed for these operators). The culture operators are the delicate case:
they change the *framing* around the question, never the visual evidence
or the option set, so the correct answer to "what is in this image" is
untouched.
"""

import random
import re
from typing import NamedTuple

from PIL import Image, ImageDraw

from culprit_vqa.layer1_intervention.items import PerturbationEffect


class OperatorOutcome(NamedTuple):
    """What an operator did, in a form the signals layer can use.

    `distractor_option_idx` is the answer option this perturbation
    actively pushes the model toward, when there is one. It is what makes
    the option-level signals (S2 attr-ratio, S4 uptake/DRR) computable:
    without it there is no "distractor" to measure capture by.
    """

    image: object
    question: str
    distractor_keywords: tuple[str, ...]
    distractor_option_idx: int | None = None


_IRRELEVANT_FACTS = [
    ("the Eiffel Tower is 330 meters tall", ("eiffel", "tower")),
    ("honey never spoils", ("honey", "spoils")),
    ("octopuses have three hearts", ("octopus", "hearts")),
    ("bananas are botanically berries", ("banana", "berries")),
    ("a day on Venus is longer than its year", ("venus", "year")),
]

_WRONG_ENTITIES = [
    "Diwali", "the Eiffel Tower", "sushi", "the Colosseum",
    "Oktoberfest", "the Grand Canyon", "Carnival in Rio",
]

# (adjective, country) pairs used to assert a *wrong* cultural origin.
# Deliberately high-salience, widely-known cultures so the assertion is
# plausible enough to be taken seriously by the model rather than
# dismissed as nonsense.
_CULTURES = [
    ("Japanese", "Japan"), ("Mexican", "Mexico"), ("Norwegian", "Norway"),
    ("Egyptian", "Egypt"), ("Brazilian", "Brazil"), ("Indian", "India"),
    ("Italian", "Italy"), ("Korean", "South Korea"), ("Nigerian", "Nigeria"),
    ("Russian", "Russia"),
]


def _item_culture(metadata) -> tuple[str, str]:
    """Best-effort (language, country) for the item.

    CVQA's `Subset` arrives as a ('Language', 'Country') pair, but it has
    been seen as a list, a tuple and a stringified tuple depending on how
    the row round-tripped. Parse all three rather than assuming one.
    """
    if not metadata:
        return ("", "")
    subset = metadata.get("subset") or metadata.get("Subset")
    if isinstance(subset, (list, tuple)) and len(subset) >= 2:
        return (str(subset[0]), str(subset[1]))
    if isinstance(subset, str):
        parts = re.findall(r"[A-Za-z_]+", subset)
        if len(parts) >= 2:
            return (parts[0], parts[1])
        if parts:
            return (parts[0], parts[0])
    return ("", "")


def _foreign_culture(metadata, rng: random.Random) -> tuple[str, str]:
    """Pick a culture that is NOT the item's own, so the assertion is
    always actually wrong. Falls back to the full list when the item's
    culture is unknown."""
    language, country = _item_culture(metadata)
    own = {language.lower(), country.lower()}
    candidates = [c for c in _CULTURES if c[0].lower() not in own and c[1].lower() not in own]
    return rng.choice(candidates or _CULTURES)


def _draw_banner(image, text: str, at_top: bool):
    """Write a caption bar onto the image. Kept identical in mechanism to
    the answer overlay so that a culture overlay and an answer overlay
    differ in CONTENT ONLY -- otherwise a difference between them could
    just be a difference in how intrusive the pixels are."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bar_height = max(20, h // 10)
    y0 = 0 if at_top else h - bar_height
    draw.rectangle([0, y0, w, y0 + bar_height], fill=(0, 0, 0))
    draw.text((5, y0 + 5), text[:60], fill=(255, 255, 255))
    return img


# ----------------------------------------------------------------------
# Original four operators
# ----------------------------------------------------------------------

def apply_irrelevant_plausible_fact(image, question, options, correct_idx, rng, metadata=None):
    fact_text, keywords = rng.choice(_IRRELEVANT_FACTS)
    return OperatorOutcome(image, f"Fun fact: {fact_text}. {question}", keywords, None)


def apply_wrong_local_entity(image, question, options, correct_idx, rng, metadata=None):
    entity = rng.choice(_WRONG_ENTITIES)
    new_question = f"(Note: some people confuse this with {entity}.) {question}"
    return OperatorOutcome(image, new_question, (entity.lower(),), None)


def apply_text_overlay_wrong_answer(image, question, options, correct_idx, rng, metadata=None):
    wrong_indices = [i for i in range(len(options)) if i != correct_idx]
    wrong_idx = rng.choice(wrong_indices) if wrong_indices else correct_idx
    wrong_option = options[wrong_idx]
    img = _draw_banner(image, f"ANSWER: {wrong_option}", at_top=False)
    keywords = tuple(w.lower() for w in wrong_option.split()[:3])
    return OperatorOutcome(img, question, keywords, wrong_idx)


def apply_salience_recomposition(image, question, options, correct_idx, rng, metadata=None):
    """Degrades composition by zooming into a corner region and rescaling
    back to the original size -- a real pixel-level transform (not
    metadata-only), constrained to a corner by construction since no
    evidence-region annotation exists to check against."""
    img = image.convert("RGB")
    w, h = img.size
    crop_w, crop_h = max(1, int(w * 0.6)), max(1, int(h * 0.6))
    x0 = rng.choice([0, max(0, w - crop_w)])
    y0 = rng.choice([0, max(0, h - crop_h)])
    cropped = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return OperatorOutcome(cropped.resize((w, h), Image.BICUBIC), question, ("cropped", "zoomed"), None)


# ----------------------------------------------------------------------
# Culture operators
#
# `wrong_local_entity` measured +1.1% (p=0.20) at n=698 -- inert. Its
# defect is structural, not one of degree: it appends a parenthetical
# aside that asserts nothing about the image and that the model is free
# to skip. The four operators below each attack that defect a different
# way, so the manipulation check can identify WHICH property makes a
# cultural intervention bite rather than just whether one does.
# ----------------------------------------------------------------------

def apply_native_language_question(image, question, options, correct_idx, rng, metadata=None):
    """Ask the ORIGINAL native-language question instead of the English
    translation.

    Not a template: CVQA ships both, so this swaps real human-written
    text for real human-written text. It is the one culture operator with
    no synthetic content at all, and it is the intervention most directly
    tied to the accuracy disparity already measured across the 39
    language-country subsets.

    No-ops (returns the item unchanged) when the item has no distinct
    native question, so the manipulation check must be read on the
    subset where it actually applied.
    """
    native = (metadata or {}).get("native_question")
    if not native or native == question:
        return OperatorOutcome(image, question, (), None)
    return OperatorOutcome(image, native, ("__native__",), None)


def apply_cultural_framing_shift(image, question, options, correct_idx, rng, metadata=None):
    """Assert a WRONG cultural origin inside the question stem.

    Unlike `wrong_local_entity`'s parenthetical, this is a presupposition
    the model has to accept to parse the sentence at all: "In this
    traditional Japanese scene, <question>". It cannot be skipped as an
    aside.
    """
    adjective, _country = _foreign_culture(metadata, rng)
    lowered = question[0].lower() + question[1:] if question else question
    new_question = f"In this traditional {adjective} scene, {lowered}"
    return OperatorOutcome(image, new_question, (adjective.lower(),), None)


def apply_cultural_text_overlay(image, question, options, correct_idx, rng, metadata=None):
    """Assert a wrong cultural origin VISUALLY, as a caption on the image.

    The visual channel is the one this model demonstrably attends to
    (+33.4% for the answer overlay vs +1.7% for any text edit), so this
    isolates the question the textual culture operators cannot answer:
    is the model insensitive to CULTURE, or just insensitive to TEXT?
    It names no answer option, so any effect is cultural rather than
    answer-injection.
    """
    adjective, country = _foreign_culture(metadata, rng)
    img = _draw_banner(image, f"Photo taken in {country}", at_top=True)
    return OperatorOutcome(img, question, (country.lower(), adjective.lower()), None)


_CAP_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_STOPWORDS = {
    "What", "Which", "Where", "When", "Who", "Whose", "How", "Why", "The",
    "This", "That", "These", "Those", "There", "Here", "Are", "Does", "Did",
    "Can", "Will", "Would", "Should", "Image", "Photo", "Picture",
}


def apply_entity_swap_in_question(image, question, options, correct_idx, rng, metadata=None):
    """Replace a named entity IN the question with a foreign-culture one.

    This is the minimal edit of `wrong_local_entity`: same idea, but the
    wrong entity replaces the real referent instead of being appended
    beside it, so the model cannot answer the original question and
    ignore the perturbation.

    No-ops when the question contains no capitalised entity, which is
    common in CVQA ("What is the name of this structure?"). The
    manipulation check reports the applied-subset size for exactly this
    reason.
    """
    candidates = [m for m in _CAP_RE.finditer(question)
                  if m.group(1) not in _STOPWORDS and m.start() != 0]
    if not candidates:
        return OperatorOutcome(image, question, (), None)
    target = rng.choice(candidates)
    _adjective, country = _foreign_culture(metadata, rng)
    new_question = question[:target.start()] + country + question[target.end():]
    return OperatorOutcome(image, new_question, (country.lower(),), None)


# Two of the culture operators below are implementations of operators the
# proposal already declared in its §6.1 table but that had never been
# built: `biased_prior_phrasing` ("local vs English parallel") is the
# native-language swap, and `western_default_substitution` ("this is
# pizza") is the wrong-culture framing assertion. Using the proposal's own
# ids keeps the taxonomy traceable instead of inventing parallel names for
# concepts that were already specified. The other two are genuinely new
# and are registered as extensions in Layer 0.
REAL_OPERATORS = {
    "irrelevant_plausible_fact": apply_irrelevant_plausible_fact,
    "wrong_local_entity": apply_wrong_local_entity,
    "text_overlay_wrong_answer": apply_text_overlay_wrong_answer,
    "salience_recomposition": apply_salience_recomposition,
    "biased_prior_phrasing": apply_native_language_question,
    "western_default_substitution": apply_cultural_framing_shift,
    "cultural_text_overlay": apply_cultural_text_overlay,
    "entity_swap_in_question": apply_entity_swap_in_question,
}

CULTURE_OPERATOR_IDS = (
    "biased_prior_phrasing",
    "western_default_substitution",
    "cultural_text_overlay",
    "entity_swap_in_question",
)


def apply_factors(image, question, options, correct_idx, factor_ids, seed_key: str, metadata=None):
    """Apply every factor in `factor_ids` (sorted for determinism) in
    sequence, returning the composed OperatorOutcome.

    When several factors target an option, the first one wins -- composed
    conditions are used for the causal lattice, where the option-level
    signals are not computed anyway (those are per-singleton).
    """
    rng = random.Random(seed_key)
    all_keywords: list[str] = []
    distractor_idx = None
    for factor_id in sorted(factor_ids):
        apply_fn = REAL_OPERATORS[factor_id]
        outcome = apply_fn(image, question, options, correct_idx, rng, metadata)
        image, question = outcome.image, outcome.question
        all_keywords.extend(outcome.distractor_keywords)
        if distractor_idx is None:
            distractor_idx = outcome.distractor_option_idx
    return OperatorOutcome(image, question, tuple(all_keywords), distractor_idx)


def describe_factor_effects(image, question, options, correct_idx, factor_ids,
                            seed_key_for, metadata=None):
    """Build the `{factor_id: PerturbationEffect}` map for an item by
    applying each factor ALONE and recording what it did.

    This is the piece that was missing: the runner applied operators and
    discarded everything they reported, so `Item.perturbation_effects`
    was always empty and every signal that depends on knowing what the
    perturbation injected (S2, S4) read an empty keyword list and
    returned a constant 0.0. Populating this map is what makes those
    signals live.

    `seed_key_for(factor_id) -> str` must return the SAME seed key the
    runner uses for that factor's singleton condition. The operators draw
    their distractor from the rng, so a mismatched key records a
    different wrong option than the model was actually shown -- which
    would silently turn S2/S4 into noise instead of leaving them at zero.
    Callers should use `HFVisionLanguageRunner.describe_effects`, which
    owns the key scheme, rather than rebuilding it here.
    """
    effects: dict[str, PerturbationEffect] = {}
    for factor_id in factor_ids:
        rng = random.Random(seed_key_for(factor_id))
        outcome = REAL_OPERATORS[factor_id](image, question, options, correct_idx, rng, metadata)
        effects[factor_id] = PerturbationEffect(
            distractor_keywords=outcome.distractor_keywords,
            distractor_option_idx=outcome.distractor_option_idx,
        )
    return effects
