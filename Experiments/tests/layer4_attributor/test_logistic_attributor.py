import math

from culprit_vqa.layer4_attributor.dataset import AttributorExample
from culprit_vqa.layer4_attributor.evaluate import expected_calibration_error, spearman_correlation
from culprit_vqa.layer4_attributor.logistic import LogisticAttributor


def _examples(n_harmful=6, n_not_harmful=6):
    examples = []
    for i in range(n_harmful):
        examples.append(
            AttributorExample(
                item_id=f"item_h{i}",
                factor_id="f_harm",
                features={"s1_drift": 1.5 + 0.01 * i, "s4_uptake": 1.0},
                phi=0.3,
                is_harmful=True,
            )
        )
    for i in range(n_not_harmful):
        examples.append(
            AttributorExample(
                item_id=f"item_n{i}",
                factor_id="f_safe",
                features={"s1_drift": 0.1 + 0.01 * i, "s4_uptake": 0.0},
                phi=-0.1,
                is_harmful=False,
            )
        )
    return examples


def test_fit_and_predict_run_without_error_on_separable_synthetic_data():
    examples = _examples()
    attributor = LogisticAttributor()
    attributor.fit(examples)
    probs = attributor.predict_proba_harmful(examples)
    assert len(probs) == len(examples)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_alpha_for_item_sums_to_one_or_is_null():
    examples = _examples()
    attributor = LogisticAttributor()
    attributor.fit(examples)

    held_out = [
        AttributorExample(item_id="held", factor_id="f_harm", features={"s1_drift": 1.6, "s4_uptake": 1.0}, phi=0.0, is_harmful=False),
        AttributorExample(item_id="held", factor_id="f_safe", features={"s1_drift": 0.1, "s4_uptake": 0.0}, phi=0.0, is_harmful=False),
    ]
    alpha = attributor.predict_alpha_for_item(held_out)
    total = sum(alpha.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9) or math.isclose(total, 0.0, abs_tol=1e-9)
    assert all(v == v for v in alpha.values())  # never NaN


def test_degenerate_all_same_label_training_does_not_raise():
    """All-harmful (or all-not-harmful) training data is a single-class
    degenerate case sklearn's LogisticRegression cannot fit directly —
    the attributor must fall back gracefully, never raise or return NaN."""
    examples = [
        AttributorExample(item_id=f"item{i}", factor_id="f", features={"s1_drift": 0.5}, phi=-0.1, is_harmful=False)
        for i in range(5)
    ]
    attributor = LogisticAttributor()
    attributor.fit(examples)  # must not raise

    held_out = [AttributorExample(item_id="held", factor_id="f", features={"s1_drift": 0.5}, phi=0.0, is_harmful=False)]
    probs = attributor.predict_proba_harmful(held_out)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0

    alpha = attributor.predict_alpha_for_item(held_out)
    assert all(v == v for v in alpha.values())  # never NaN


def test_predict_alpha_for_item_empty_list_returns_empty_dict():
    attributor = LogisticAttributor()
    attributor.fit(_examples())
    assert attributor.predict_alpha_for_item([]) == {}


def test_evaluate_helpers_run_without_exceptions_on_toy_inputs():
    rho = spearman_correlation([0.1, 0.4, 0.9], [0.2, 0.3, 0.8])
    assert -1.0 <= rho <= 1.0

    ece = expected_calibration_error(probs=[0.1, 0.9, 0.4, 0.6], labels=[0, 1, 0, 1])
    assert 0.0 <= ece <= 1.0
