"""Repair operators for the natural-failure audit (proposal §7).

Natural (non-injected) failures have no perturbation to remove, so
attribution is validated constructively: each repair is a hypothesis
about what's wrong, and if applying it flips a wrong answer to correct,
that repair's hypothesized cause has causal evidence for this failure.

This is the "fix everything, let whichever fix works define the label"
design: rather than classifying a failure first and then choosing a fix
(which would require ground-truth labels for natural failures that by
definition do not exist), every repair is applied to every failure and
the diagnosis is whichever repair causally helped. Each repair is
therefore deliberately tied to ONE axis of the Layer 0 taxonomy, so a
flip is readable as evidence for that axis.

Placebo repairs are registered alongside the real ones and marked
`is_placebo=True`. They add text of comparable length but carry no
information relevant to the question. They exist because a raw flip
rate is uninterpretable on its own: greedy decoding is sensitive to any
prompt change, so some fraction of "fixes" are just perturbation noise.
The placebo flip rate estimates that floor, and a real repair only
counts as evidence if it beats it.

A repair may edit the question text (`question_fn`), the image
(`image_fn`), or both. Image-editing repairs are what make the visual
axis a genuine hypothesis rather than a prompt tweak.
"""

from dataclasses import dataclass
from typing import Callable

from culprit_vqa.layer1_intervention.items import Item


@dataclass(frozen=True)
class Repair:
    """One repair hypothesis.

    `question_fn` takes the whole `Item` (not just the question string)
    because some repairs need context the question alone doesn't carry --
    the native-language variant, the cultural subset, the category.
    `image_fn` takes and returns a PIL image. Either may be None; at
    least one must be set or the repair is a no-op.

    `axis` names the Layer 0 taxonomy axis this repair probes, so a flip
    can be read as causal evidence for that axis. Placebos carry
    axis="none".
    """

    repair_id: str
    axis: str
    hypothesis: str
    question_fn: Callable[[Item], str] | None = None
    image_fn: Callable | None = None
    max_image_side: int | None = None
    """Per-repair resolution cap handed to the runner. Only meaningful
    for image-editing repairs: the runner caps every image at 768px by
    default, which would silently undo an upscale (every real CVQA image
    already exceeds 768), so an upscale repair must raise its own cap or
    it measures nothing."""
    is_placebo: bool = False


# ----------------------------------------------------------------------
# Question-text repairs
# ----------------------------------------------------------------------


def apply_add_category_hint(item: Item) -> str:
    """Knowledge axis: the model may have had the visual facts but not
    known what KIND of thing it was being asked about. Names the domain
    without naming the answer."""
    category = item.metadata.get("category", "general knowledge")
    return f"(This is a question about: {category}.) {item.question}"


def apply_supply_cultural_context(item: Item) -> str:
    """Culturality axis: distinct from the knowledge hint above -- this
    supplies the cultural/regional SETTING rather than the topic. Knowing
    "this is a food question" and knowing "this is Bengali food from
    India" are different pieces of information, and CVQA items are
    culture-specific by construction, so they can fail independently."""
    subset = item.metadata.get("subset", item.language)
    return (
        f"(This image and question come from the following cultural context: {subset}. "
        f"Answer using knowledge of that culture.) {item.question}"
    )


def apply_ask_in_native_language(item: Item) -> str:
    """Language axis: the audit asks the English-translated question by
    default. If the translation lost or distorted something, asking the
    original native-language question should recover it. Falls back to
    the English question when no native text is available, which makes
    this repair a silent no-op for those items -- the analysis must
    exclude items with no native question rather than count them as
    'language repair failed'."""
    native = item.metadata.get("native_question")
    return native if native else item.question


def apply_chain_of_thought(item: Item) -> str:
    """Reasoning axis: the model may have had everything it needed and
    still jumped to a snap answer. Asks it to reason before committing."""
    return f"{item.question}\nLet's think step by step before choosing the final answer."


def apply_look_closely(item: Item) -> str:
    """Visual axis (text half). Paired with `upscale_image` below -- the
    instruction alone would just be another prompt tweak; the point of
    the pair is to give the model both more pixels and a reason to use
    them."""
    return (
        f"{item.question}\nLook carefully at the fine details in the image "
        f"before choosing your answer."
    )


# ----------------------------------------------------------------------
# Image repairs
# ----------------------------------------------------------------------


VISUAL_REPAIR_MAX_SIDE = 1152
"""1.5x the runner's 768px default cap. Chosen, not maximized: 2.25x the
pixels means ~2.25x the vision tokens and activation memory, which a
16GB T4 running a 4-bit 3B model can absorb, and the runner's
half-resolution OOM retry covers the tail. Going higher is what OOM'd
the first full run."""


def upscale_image(image, target_long_side: int = VISUAL_REPAIR_MAX_SIDE):
    """Visual axis (image half): resample the image so its long side hits
    `target_long_side`, so a dynamic-resolution vision tower allocates
    more tokens to it and small or low-contrast evidence has a better
    chance of surviving tokenization. This adds no information the image
    didn't already contain -- it only changes how much of it reaches the
    model.

    Note this sets an absolute target rather than multiplying the source
    size: the runner downsamples every image to a fixed 768px cap
    anyway, so the model's actual baseline is 768 regardless of how large
    the source file was. Scaling relative to the raw source would make
    the repair's real strength vary arbitrarily by source resolution.
    """
    if image is None:
        return None
    from PIL import Image

    w, h = image.size
    scale = target_long_side / max(w, h)
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample)


# ----------------------------------------------------------------------
# Placebo controls -- length-matched, information-free
# ----------------------------------------------------------------------


def apply_placebo_irrelevant_prefix(item: Item) -> str:
    """Placebo matched in shape to the two prefix repairs
    (`add_category_hint`, `supply_cultural_context`): a parenthetical
    preamble of comparable length that is true but tells the model
    nothing it could use."""
    return (
        f"(This question is part of a multiple-choice visual survey. "
        f"Exactly one option is intended to be correct.) {item.question}"
    )


def apply_placebo_neutral_suffix(item: Item) -> str:
    """Placebo matched in shape to the two suffix repairs
    (`chain_of_thought`, `look_closely`): a trailing instruction of
    comparable length that requests nothing the task didn't already
    require."""
    return f"{item.question}\nPlease read the question and then provide your answer."


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

REPAIRS: tuple[Repair, ...] = (
    Repair(
        repair_id="add_category_hint",
        axis="knowledge",
        hypothesis="The model lacked topical context about what is being asked.",
        question_fn=apply_add_category_hint,
    ),
    Repair(
        repair_id="supply_cultural_context",
        axis="culturality",
        hypothesis="The model lacked the cultural/regional setting the item assumes.",
        question_fn=apply_supply_cultural_context,
    ),
    Repair(
        repair_id="ask_in_native_language",
        axis="modality_text",
        hypothesis="The English translation lost or distorted the question.",
        question_fn=apply_ask_in_native_language,
    ),
    Repair(
        repair_id="chain_of_thought",
        axis="reasoning",
        hypothesis="The model took a reasoning shortcut rather than lacking information.",
        question_fn=apply_chain_of_thought,
    ),
    Repair(
        repair_id="enhance_visual_detail",
        axis="modality_visual",
        hypothesis="The decisive visual evidence was too small or too weakly attended.",
        question_fn=apply_look_closely,
        image_fn=upscale_image,
        max_image_side=VISUAL_REPAIR_MAX_SIDE,
    ),
    Repair(
        repair_id="placebo_irrelevant_prefix",
        axis="none",
        hypothesis="Control: measures flips caused by adding a prefix at all.",
        question_fn=apply_placebo_irrelevant_prefix,
        is_placebo=True,
    ),
    Repair(
        repair_id="placebo_neutral_suffix",
        axis="none",
        hypothesis="Control: measures flips caused by adding a suffix at all.",
        question_fn=apply_placebo_neutral_suffix,
        is_placebo=True,
    ),
)

REPAIR_BY_ID: dict[str, Repair] = {r.repair_id: r for r in REPAIRS}

REAL_REPAIRS: tuple[Repair, ...] = tuple(r for r in REPAIRS if not r.is_placebo)
PLACEBO_REPAIRS: tuple[Repair, ...] = tuple(r for r in REPAIRS if r.is_placebo)
