from PIL import Image

from culprit_vqa.layer1_intervention.real_operators import (
    REAL_OPERATORS,
    apply_factors,
    apply_salience_recomposition,
    apply_text_overlay_wrong_answer,
)


def _image():
    return Image.new("RGB", (64, 64), color=(120, 130, 140))


def test_text_overlay_produces_a_different_image_without_changing_question():
    img = _image()
    options = ["mango", "banana", "apple", "grape"]
    new_img, new_q, keywords, distractor_idx = apply_text_overlay_wrong_answer(img, "What fruit?", options, 0, __import__("random").Random(0))
    assert new_img.size == img.size
    assert list(new_img.getdata()) != list(img.getdata())  # pixels actually changed
    assert new_q == "What fruit?"
    assert len(keywords) > 0
    assert distractor_idx is not None and distractor_idx != 0  # a wrong option, recorded


def test_salience_recomposition_preserves_size_and_changes_pixels():
    img = Image.new("RGB", (100, 80))
    for x in range(100):
        for y in range(80):
            img.putpixel((x, y), (x % 256, y % 256, 0))
    new_img, new_q, keywords, distractor_idx = apply_salience_recomposition(img, "q?", ["a", "b"], 0, __import__("random").Random(1))
    assert new_img.size == img.size
    assert list(new_img.getdata()) != list(img.getdata())


def test_apply_factors_composes_multiple_operators_deterministically():
    img = _image()
    options = ["mango", "banana", "apple", "grape"]
    factor_ids = ["irrelevant_plausible_fact", "wrong_local_entity"]

    img1, q1, kw1, _ = apply_factors(img, "What fruit?", options, 0, factor_ids, seed_key="item1::cond1")
    img2, q2, kw2, _ = apply_factors(img, "What fruit?", options, 0, factor_ids, seed_key="item1::cond1")

    assert q1 == q2  # deterministic given the same seed key
    assert kw1 == kw2
    assert q1 != "What fruit?"  # actually modified
    assert list(img1.getdata()) == list(img2.getdata())


def test_all_registered_operators_are_callable_with_uniform_signature():
    img = _image()
    options = ["mango", "banana", "apple", "grape"]
    for factor_id, fn in REAL_OPERATORS.items():
        new_img, new_q, keywords, _ = fn(img, "What fruit?", options, 0, __import__("random").Random(0))
        assert new_img is not None
        assert isinstance(new_q, str)
        assert isinstance(keywords, tuple)
