"""Full-wiring smoke test: 5 synthetic items -> MockModelRunner -> Layer 3a
(efficiency axiom) -> Layer 3b signals -> Layer 4 (train/predict on a
held-out split) -> Layer 5 (non-empty failure profile).

This asserts PLUMBING, not attribution accuracy — 5 synthetic items are
far too few to demand any correlation threshold (proposal's own RQ3
target, rho >= 0.7, is a held-out-real-data target, not a toy-fixture one).
"""

import math

from culprit_vqa.layer3a_causal.attribution import efficiency_residual, is_null_attribution
from culprit_vqa.layer3a_causal.probability import build_p_function
from culprit_vqa.layer4_attributor.dataset import build_examples
from culprit_vqa.layer4_attributor.logistic import LogisticAttributor
from culprit_vqa.layer5_profiles.aggregate import build_failure_profile
from culprit_vqa.layer5_profiles.interaction_atlas import build_interaction_atlas

TOL = 1e-6


def test_every_record_has_full_lattice_validity_and_valid_shapley_output(pipeline_records):
    for record in pipeline_records:
        k = len(record.factors)
        assert len(record.run_results) == 2 ** k
        assert len(record.validity_reports) == 2 ** k

        p_fn = build_p_function(record.run_results)
        residual = efficiency_residual(record.phi, p_fn, record.factors)
        assert abs(residual) < 1e-9, f"{record.item.item_id}: efficiency residual {residual}"

        total_alpha = sum(record.alpha.values())
        assert math.isclose(total_alpha, 1.0, abs_tol=TOL) or is_null_attribution(record.alpha)

        # Every singleton condition should have produced a signal vector.
        assert len(record.signal_vectors) == k


def test_null_player_item_produces_null_attribution(pipeline_records):
    record = next(r for r in pipeline_records if r.item.item_id == "item_null_player_k1")
    assert is_null_attribution(record.alpha)


def test_interaction_item_has_nonempty_interaction_atlas_entry(pipeline_records):
    record = next(r for r in pipeline_records if r.item.item_id == "item_interaction_k2")
    assert len(record.interactions) > 0


def test_layer4_trains_on_held_out_split_and_predicts_valid_alpha(pipeline_records):
    train_records, held_out = pipeline_records[:-1], pipeline_records[-1]
    train_examples = build_examples(train_records)
    held_out_examples = build_examples([held_out])

    attributor = LogisticAttributor()
    attributor.fit(train_examples)  # must not raise
    predicted_alpha = attributor.predict_alpha_for_item(held_out_examples)

    assert set(predicted_alpha.keys()) == {ex.factor_id for ex in held_out_examples}
    total = sum(predicted_alpha.values())
    assert math.isclose(total, 1.0, abs_tol=TOL) or math.isclose(total, 0.0, abs_tol=TOL)
    assert all(v == v for v in predicted_alpha.values())  # never NaN


def test_layer5_failure_profile_is_nonempty(pipeline_records):
    profiles = build_failure_profile(pipeline_records, group_by=lambda r: r.item.language)
    assert len(profiles) > 0
    for profile in profiles.values():
        assert profile.n_items > 0
        for value in profile.factor_mixture.values():
            assert 0.0 <= value <= 1.0

    atlas = build_interaction_atlas(pipeline_records)
    assert len(atlas) > 0  # item_interaction_k2 and item_full_k3 both have k>=2
