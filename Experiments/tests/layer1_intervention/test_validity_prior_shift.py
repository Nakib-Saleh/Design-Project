from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import generate_lattice
from culprit_vqa.layer1_intervention.validity import run_validity_checks
from culprit_vqa.layer1_intervention.validity.prior_shift import prior_shift_measurement


def _item_and_condition():
    factor = get_operator("western_default_substitution")
    item = Item(item_id="item", image_ref="ref", question="q?", answer="a")
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors]
    return item, condition


def test_prior_shift_never_affects_acceptance():
    item, condition = _item_and_condition()

    def prior_fn_huge_shift(item, condition):
        return 1.0 if condition.applied_factors else 0.0

    def prior_fn_no_shift(item, condition):
        return 0.5

    report_huge = run_validity_checks(item, condition, prior_fn=prior_fn_huge_shift)
    report_none = run_validity_checks(item, condition, prior_fn=prior_fn_no_shift)

    # The prior_shift VALUE differs...
    assert report_huge.prior_shift != report_none.prior_shift
    # ...but acceptance is identical (driven only by semantic/evidence checks).
    assert report_huge.accepted == report_none.accepted


def test_prior_shift_measured_relative_to_control():
    item, condition = _item_and_condition()

    def prior_fn(item, condition):
        return 0.8 if condition.applied_factors else 0.3

    shift = prior_shift_measurement(item, condition, prior_fn=prior_fn)
    assert shift == 0.8 - 0.3
