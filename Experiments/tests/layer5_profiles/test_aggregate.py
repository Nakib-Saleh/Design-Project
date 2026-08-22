from culprit_vqa.layer0_taxonomy.axes import Conflict, Culturality, Modality, Relevance
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer5_profiles.aggregate import build_failure_profile
from culprit_vqa.layer5_profiles.interaction_atlas import build_interaction_atlas
from culprit_vqa.pipeline import PipelineRecord


def _factor(fid):
    return Factor(
        id=fid,
        coordinates=FactorCoordinates(
            Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
        ),
    )


def _record(item_id, language, alpha, interactions=None):
    item = Item(item_id=item_id, image_ref="ref", question="q?", answer="a", language=language)
    factors = list(alpha.keys())
    return PipelineRecord(item=item, factors=factors, alpha=alpha, interactions=interactions or {})


def test_build_failure_profile_groups_and_averages_correctly():
    a, b = _factor("a"), _factor("b")
    records = [
        _record("item1", "en", {a: 0.8, b: 0.2}),
        _record("item2", "en", {a: 0.4, b: 0.6}),
        _record("item3", "bn", {a: 1.0, b: 0.0}),
    ]
    profiles = build_failure_profile(records, group_by=lambda r: r.item.language)

    assert set(profiles.keys()) == {"en", "bn"}
    assert profiles["en"].n_items == 2
    assert abs(profiles["en"].factor_mixture["a"] - 0.6) < 1e-9
    assert abs(profiles["en"].factor_mixture["b"] - 0.4) < 1e-9
    assert profiles["bn"].n_items == 1
    for profile in profiles.values():
        for value in profile.factor_mixture.values():
            assert 0.0 <= value <= 1.0


def test_build_interaction_atlas_nonempty_when_k_geq_2():
    a, b = _factor("a"), _factor("b")
    records = [
        _record("item1", "en", {a: 0.5, b: 0.5}, interactions={frozenset({a, b}): -0.4}),
        _record("item2", "en", {a: 0.5, b: 0.5}, interactions={frozenset({a, b}): -0.2}),
    ]
    atlas = build_interaction_atlas(records)
    key = frozenset({"a", "b"})
    assert key in atlas
    assert abs(atlas[key] - (-0.3)) < 1e-9


def test_build_interaction_atlas_empty_for_k1_records():
    a = _factor("a")
    records = [_record("item1", "en", {a: 1.0})]
    atlas = build_interaction_atlas(records)
    assert atlas == {}
