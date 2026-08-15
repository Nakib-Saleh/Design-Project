from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import BBox, EvidenceRegion, Item, PerturbationEffect
from culprit_vqa.layer1_intervention.lattice import generate_lattice
from culprit_vqa.layer1_intervention.validity.evidence import evidence_preservation_check

EVIDENCE_BBOX = BBox(0.1, 0.1, 0.4, 0.4)


def _item_with_touched_bbox(touched_bbox):
    factor = get_operator("diffusion_irrelevant_object")
    item = Item(
        item_id="item",
        image_ref="ref",
        question="q?",
        answer="a",
        evidence_region=EvidenceRegion(EVIDENCE_BBOX),
        perturbation_effects={factor.id: PerturbationEffect(touched_bbox=touched_bbox, perturbed_answer="a")},
    )
    return item, factor


def test_perturbation_overlapping_evidence_is_rejected():
    overlapping_bbox = BBox(0.15, 0.15, 0.45, 0.45)  # overlaps EVIDENCE_BBOX heavily
    item, factor = _item_with_touched_bbox(overlapping_bbox)
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors == frozenset({factor})]
    assert evidence_preservation_check(item, condition) is False


def test_perturbation_far_from_evidence_is_accepted():
    far_bbox = BBox(0.6, 0.6, 0.9, 0.9)
    item, factor = _item_with_touched_bbox(far_bbox)
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors == frozenset({factor})]
    assert evidence_preservation_check(item, condition) is True


def test_removes_needed_text_is_rejected():
    factor = get_operator("irrelevant_plausible_fact")
    item = Item(
        item_id="item",
        image_ref="ref",
        question="q?",
        answer="a",
        evidence_region=EvidenceRegion(EVIDENCE_BBOX),
        perturbation_effects={factor.id: PerturbationEffect(removes_needed_text=True, perturbed_answer="a")},
    )
    [condition] = [c for c in generate_lattice(item, [factor]) if c.applied_factors == frozenset({factor})]
    assert evidence_preservation_check(item, condition) is False


def test_control_condition_is_always_accepted():
    item, factor = _item_with_touched_bbox(BBox(0.15, 0.15, 0.45, 0.45))
    [control] = [c for c in generate_lattice(item, [factor]) if not c.applied_factors]
    assert evidence_preservation_check(item, control) is True
