import dataclasses

import pytest

from culprit_vqa.layer0_taxonomy.axes import (
    Conflict,
    Culturality,
    Knowledge,
    Modality,
    Relevance,
)
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates


def _coords():
    return FactorCoordinates(
        Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
    )


def test_factor_equality_and_hash_are_by_id_only():
    a = Factor(id="same_id", coordinates=_coords(), description="first description")
    b = Factor(id="same_id", coordinates=_coords(), description="a totally different description")
    assert a == b
    assert hash(a) == hash(b)


def test_factors_with_different_ids_are_not_equal():
    a = Factor(id="id_a", coordinates=_coords())
    b = Factor(id="id_b", coordinates=_coords())
    assert a != b


def test_factor_dedupes_correctly_in_a_set():
    a = Factor(id="dup", coordinates=_coords(), description="desc 1")
    b = Factor(id="dup", coordinates=_coords(), description="desc 2")
    assert {a, b} == {a}
    assert len({a, b}) == 1


def test_factor_coordinates_are_immutable():
    coords = _coords()
    with pytest.raises(dataclasses.FrozenInstanceError):
        coords.modality = Modality.TEXTUAL  # type: ignore[misc]


def test_factor_default_knowledge_axis_is_perceptual():
    coords = _coords()
    assert coords.knowledge == Knowledge.PERCEPTUAL
