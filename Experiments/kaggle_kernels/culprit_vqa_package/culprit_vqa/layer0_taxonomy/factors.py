"""FactorCoordinates and Factor — the formal objects of the taxonomy (proposal §4.1).

"Factors compose; leaves don't" — a Factor is a point in the 5-axis space,
not a leaf of the legacy tree. Two perturbation operators that appear as
distinct leaves in the presentation tree (e.g. a cultural-substitution
leaf and a cross-modal-conflict leaf) may in fact be the *same* point in
factor-coordinate space, or differ on exactly one axis — that is the
resolution to the v1 taxonomy-overlap criticism (see PROPOSAL_CULPRIT_VQA.md §4.2).
"""

from dataclasses import dataclass

from culprit_vqa.layer0_taxonomy.axes import (
    Conflict,
    Culturality,
    Knowledge,
    Modality,
    Relevance,
)


@dataclass(frozen=True)
class FactorCoordinates:
    """A point in the 5-axis factor space."""

    modality: Modality
    relevance: Relevance
    conflict: Conflict
    culturality: Culturality
    knowledge: Knowledge = Knowledge.PERCEPTUAL


@dataclass(frozen=True, eq=False)
class Factor:
    """A named, instantiable perturbation factor (an "operator") tagged
    with its 5-axis coordinates.

    Identity is by `id` alone: two `Factor` objects constructed at
    different times (e.g. once in the operator registry, once rebuilt
    from a serialized record) must compare equal and hash the same as
    long as their id matches, even if `description` text differs
    slightly. This matters because `Factor` instances are used as
    dict/set keys throughout Layer 1 (lattice subsets) and Layer 3a
    (Shapley sums over factor subsets).
    """

    id: str
    coordinates: FactorCoordinates
    description: str = ""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Factor) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Factor(id={self.id!r})"
