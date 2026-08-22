from culprit_vqa.layer0_taxonomy.axes import Conflict, Culturality, Modality, Relevance
from culprit_vqa.layer0_taxonomy.factors import Factor, FactorCoordinates
from culprit_vqa.layer3a_causal.attribution import is_null_attribution, normalize_attribution


def _factor(fid):
    return Factor(
        id=fid,
        coordinates=FactorCoordinates(
            Modality.VISUAL, Relevance.IRRELEVANT, Conflict.CONSISTENT, Culturality.NEUTRAL
        ),
    )


def test_all_zero_phi_normalizes_to_all_zero_alpha_without_error():
    a, b = _factor("a"), _factor("b")
    alpha = normalize_attribution({a: 0.0, b: 0.0})
    assert alpha == {a: 0.0, b: 0.0}
    assert is_null_attribution(alpha)


def test_all_negative_phi_normalizes_to_all_zero_alpha():
    a, b = _factor("a"), _factor("b")
    alpha = normalize_attribution({a: -0.3, b: -0.1})
    assert alpha == {a: 0.0, b: 0.0}
    assert is_null_attribution(alpha)


def test_mixed_sign_phi_normalizes_only_the_positive_part():
    a, b, c = _factor("a"), _factor("b"), _factor("c")
    alpha = normalize_attribution({a: 0.6, b: -0.2, c: 0.4})
    assert alpha[b] == 0.0
    assert abs(alpha[a] - 0.6) < 1e-9
    assert abs(alpha[c] - 0.4) < 1e-9
    assert abs(sum(alpha.values()) - 1.0) < 1e-9
    assert not is_null_attribution(alpha)


def test_single_positive_factor_gets_all_the_credit():
    a, b = _factor("a"), _factor("b")
    alpha = normalize_attribution({a: 0.25, b: 0.0})
    assert alpha[a] == 1.0
    assert alpha[b] == 0.0
