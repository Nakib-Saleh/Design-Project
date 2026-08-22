"""Item, evidence region, and per-factor perturbation effects."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    """An axis-aligned box, e.g. over an image, in arbitrary consistent units."""

    x0: float
    y0: float
    x1: float
    y1: float

    def area(self) -> float:
        w = max(0.0, self.x1 - self.x0)
        h = max(0.0, self.y1 - self.y0)
        return w * h

    def intersection(self, other: "BBox") -> float:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def iou(self, other: "BBox") -> float:
        """Standard intersection-over-union; 0.0 if there is no overlap or
        either box is degenerate (zero area)."""
        inter = self.intersection(other)
        if inter == 0.0:
            return 0.0
        union = self.area() + other.area() - inter
        if union <= 0.0:
            return 0.0
        return inter / union


@dataclass(frozen=True)
class EvidenceRegion:
    """The decisive visual evidence region for an item's answer."""

    bbox: BBox
    description: str = ""


@dataclass(frozen=True)
class PerturbationEffect:
    """What a given factor actually does to a specific item.

    This is fixture/generation-time authored metadata used by the
    validity checks (Layer 1) and by the behavioral signals (Layer 3b) —
    it is never inspected by Layer 3a's causal computation, which only
    ever looks at model outputs (RunResult).
    """

    touched_bbox: BBox | None = None
    removes_needed_text: bool = False
    perturbed_answer: str | None = None
    distractor_keywords: tuple[str, ...] = ()
    # Index of the answer option this factor actively pushes the model
    # toward, when the factor targets a specific option (e.g. an overlay
    # that writes one wrong option onto the image). None for factors with
    # no option-level target, such as a salience crop -- those legitimately
    # have no distractor to be captured by, and the option-level signals
    # (S2/S4) return 0.0 for them rather than inventing a value.
    distractor_option_idx: int | None = None


@dataclass
class Item:
    """A base VQA item plus the per-factor perturbation effects that can
    be applied to it."""

    item_id: str
    image_ref: str
    question: str
    answer: str
    evidence_region: EvidenceRegion | None = None
    language: str = "en"
    perturbation_effects: dict[str, PerturbationEffect] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
