"""End-to-end guard: the signal vector must not be constant.

The first real attribution run (1000 items) produced s2_attr_ratio,
s3_nli_conflict and s4_uptake as EXACTLY 0.0 on every row. Nothing failed
and nothing warned -- the amortized attributor simply trained on three
columns of zeros and reached rho=0.49 against a 0.70 target on what was
effectively one feature.

That is the failure mode this module exists to catch: signals that are
silently dead rather than wrong. It drives the real pipeline with a stub
runner that emits realistic option distributions, and asserts the
resulting features actually vary.
"""

import random

import pytest

from culprit_vqa.layer0_taxonomy.operators import get_operator
from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.real_operators import describe_factor_effects
from culprit_vqa.layer2_runner.base import RunResult
from culprit_vqa.layer3b_signals import DEFAULT_SIGNALS
from culprit_vqa.pipeline import run_pipeline_for_item

OPTIONS = ["Torii", "Pagoda", "Temple", "Shrine"]
CORRECT = 0
FACTOR_IDS = ["text_overlay_wrong_answer", "salience_recomposition"]


class StubRunner:
    """Emits a plausible option distribution per condition.

    Deliberately makes the overlay factor capture its own injected option
    on some items and not others, so a signal that responds correctly
    will vary and one that is hard-wired to a constant will not.
    """

    def __init__(self, item, effects, seed=0):
        self.item = item
        self.effects = effects
        self.rng = random.Random(seed)

    def run(self, item, condition, n_decodes=1):
        applied = {f.id for f in condition.applied_factors}
        probs = [0.70, 0.14, 0.09, 0.07]

        if "text_overlay_wrong_answer" in applied:
            target = self.effects["text_overlay_wrong_answer"].distractor_option_idx
            if target is not None and self.rng.random() < 0.6:
                probs = [0.15] * 4
                probs[target] = 0.55  # captured by the injected option
            else:
                probs = [0.45, 0.40, 0.10, 0.05]  # torn, but gold holds
        if "salience_recomposition" in applied:
            probs = [0.40, 0.38, 0.12, 0.10]  # blurred evidence -> less certain

        total = sum(probs)
        probs = [p / total for p in probs]
        predicted = max(range(len(probs)), key=lambda i: probs[i])
        return RunResult(
            item_id=item.item_id,
            condition_id=condition.condition_id,
            sampled_answers=[OPTIONS[predicted]],
            correct_flags=[predicted == CORRECT],
            cot_traces=["ABCD"[predicted]],  # letter scoring emits ONE token
            gold_prob_mass=probs[CORRECT],
            extra_signals={
                "option_probs": probs,
                "predicted_idx": predicted,
                "gen_confidence": probs[predicted],
            },
        )


def _item(index, image):
    return Item(
        item_id=f"live_{index:03d}",
        image_ref="ref",
        question="What is the name of this structure?",
        answer=OPTIONS[CORRECT],
        language="('Indonesian', 'Indonesia')",
        metadata={
            "pil_image": image,
            "options": OPTIONS,
            "correct_idx": CORRECT,
            "subset": ("Indonesian", "Indonesia"),
            "native_question": "Apa nama bangunan ini?",
            "language_delta": 0.1 * (index % 5),
        },
    )


def _run_many(n=25):
    from PIL import Image

    image = Image.new("RGB", (64, 64), (100, 110, 120))
    factors = [get_operator(fid) for fid in FACTOR_IDS]
    collected: dict[str, list[float]] = {}
    for i in range(n):
        item = _item(i, image)
        effects = describe_factor_effects(
            image, item.question, OPTIONS, CORRECT, FACTOR_IDS,
            seed_key_for=lambda fid, i=i: f"0::{item.item_id}::{item.item_id}::{fid}",
            metadata=item.metadata,
        )
        item.perturbation_effects = effects
        record = run_pipeline_for_item(
            item, factors, StubRunner(item, effects, seed=i), n_decodes=1
        )
        for vec in record.signal_vectors.values():
            for name, value in vec.values.items():
                collected.setdefault(name, []).append(value)
    return collected


@pytest.fixture(scope="module")
def signal_columns():
    return _run_many()


@pytest.mark.parametrize("name", [s.name for s in DEFAULT_SIGNALS if s.name != "s6_language_delta"])
def test_signal_is_not_constant(signal_columns, name):
    values = signal_columns[name]
    assert len(set(values)) > 1, (
        f"{name} is constant at {values[0]} across {len(values)} conditions -- "
        "this is the exact defect that made the first attribution run's "
        "amortized attributor a one-feature model."
    )


def test_uptake_fires_on_some_conditions_but_not_all():
    """A DRR signal stuck at 1.0 would be as useless as one stuck at 0.0."""
    values = _run_many()["s4_uptake"]
    assert 0.0 < sum(values) / len(values) < 1.0


def test_attr_ratio_spans_both_directions():
    values = _run_many()["s2_attr_ratio"]
    assert max(values) > 0.0
    assert min(values) <= 0.0


def test_no_signal_is_nan():
    """A NaN feature silently poisons a scikit-learn fit rather than
    raising, so it has to be caught here."""
    import math

    for name, values in _run_many().items():
        assert all(math.isfinite(v) for v in values), f"{name} produced a non-finite value"


def test_salience_factor_still_gets_a_live_conflict_signal():
    """S2 and S4 legitimately stay 0.0 for a factor that targets no
    option. If S3 did too, that factor would have no live signal at all
    and Layer 4 could not distinguish its conditions."""
    columns: dict[str, list[float]] = {}
    from PIL import Image

    image = Image.new("RGB", (64, 64), (100, 110, 120))
    factors = [get_operator(fid) for fid in FACTOR_IDS]
    for i in range(15):
        item = _item(i, image)
        effects = describe_factor_effects(
            image, item.question, OPTIONS, CORRECT, FACTOR_IDS,
            seed_key_for=lambda fid, i=i: f"0::{item.item_id}::{item.item_id}::{fid}",
            metadata=item.metadata,
        )
        item.perturbation_effects = effects
        record = run_pipeline_for_item(item, factors, StubRunner(item, effects, seed=i), n_decodes=1)
        for cond_id, vec in record.signal_vectors.items():
            if cond_id.endswith("salience_recomposition"):
                columns.setdefault("s3", []).append(vec.values["s3_nli_conflict"])
    assert any(v != 0.0 for v in columns["s3"])
