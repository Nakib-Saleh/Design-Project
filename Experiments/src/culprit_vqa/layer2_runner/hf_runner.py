"""Real HF-backed ModelRunner (proposal §6.3), for use on a Kaggle GPU
kernel -- NOT importable/runnable without `torch`/`transformers` installed
(lazy-imported inside methods so the rest of the package stays usable in
environments without a GPU stack, e.g. the local hermetic test suite).

Genericized over model family via `transformers.AutoModelForImageTextToText`
(works for Qwen2.5-VL, LLaVA-OneVision, and other chat-template-based
vision-language models) so the same runner class supports the
multi-model comparison in the proposal's Layer 2 design, not just one
specific model.

MEASUREMENT (read this before changing anything here)
-----------------------------------------------------
`gold_prob_mass` is the quantity every Shapley value in this project is
computed from, so it has to actually measure "does the model prefer the
right answer". The original implementation scored the ABSOLUTE
teacher-forced probability of the gold option's literal text, and it did
not: measured on two real runs, its AUC for separating correct from
incorrect answers was 0.46 (Qwen2.5-VL-3B) and 0.53 (LLaVA-OneVision-7B)
-- i.e. indistinguishable from a coin flip. Everything computed on top of
it was therefore noise, which is the real explanation for the amortized
attributor's rho ~= 0.09.

It failed for three reasons, all addressed below by scoring options
RELATIVE TO EACH OTHER instead of in absolute terms:

  1. Not normalized over the alternatives. The meaningful question is
     "does the model prefer gold over the other options", not "what
     absolute probability does this one string get". Absolute string
     probability is dominated by length, token frequency and
     tokenization, none of which track correctness.
  2. Not comparable across items, for the same reason: options with
     different surface texts get systematically different scores
     regardless of whether they are right.
  3. Wrong target for letter-answering models. LLaVA-OneVision emitted a
     bare letter on 100% of items while we scored the probability of the
     full option text, so the two measures were not even about the same
     event.

The replacement is the standard multiple-choice scoring used by
lm-eval-harness / MMLU: score each option, softmax across options, and
read off p(gold). It is bounded, has a meaningful chance baseline (1/k),
is comparable across items and models, and makes correctness
`argmax == gold` -- which retires the string-matching heuristic entirely.

Two scoring modes are provided because they fail in different ways and
the calibration compares them directly:
  * "letter" (default): read the k letter-token logits out of a SINGLE
    forward pass over the prefix. One forward per condition -- cheaper
    than the generation call it replaces -- and free of length bias,
    since every option is one token.
  * "text": teacher-force each option's full text and length-normalize.
    k forward passes, but it does not assume the model can map letters
    to options, so it is the fallback when letter scoring degenerates.

Correctness note on continuation log-probs (unchanged, and still
required): tokenize the prompt prefix ALONE to get its token count (via
the actual multimodal processor, so image-token expansion is accounted
for), then tokenize prefix+continuation together and take the tail --
never re-tokenizing the continuation standalone, which drifts from its
in-context tokenization at the boundary (the bug found in the Phase 0
calibration script).
"""

import gc
import math
from typing import NamedTuple

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer1_intervention.lattice import Condition, make_condition_id
from culprit_vqa.layer1_intervention.real_operators import (
    apply_factors,
    describe_factor_effects,
)
from culprit_vqa.layer2_runner.base import ModelRunner, RunResult

DEFAULT_MAX_IMAGE_SIDE = 768
OPTION_LETTERS = "ABCDEFGH"


class _FactorKey(NamedTuple):
    """Minimal stand-in carrying just the `.id` that `make_condition_id`
    reads, so the runner can rebuild a singleton condition id from a bare
    factor id without importing the taxonomy."""

    id: str


def _cap_resolution(image, max_side: int):
    """Resize so the longer side is at most `max_side`, preserving aspect
    ratio. Real CVQA images vary wildly in native resolution; dynamic-
    resolution vision towers turn large images into proportionally more
    vision tokens, which was the dominant driver of the CUDA OOM failures
    observed in the first full-scale run (49/100 items) -- capping
    resolution up front bounds worst-case activation memory regardless of
    the source image size."""
    from PIL import Image

    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), resample)


def has_degenerate_scores(values) -> bool:
    """True if any raw option score is NaN or infinite.

    Recorded per call so the analysis can EXCLUDE these conditions rather
    than silently trust a repaired distribution. Observed on real
    hardware: 2/173 items in the instrument check produced a NaN logit
    (4-bit quantized inference can overflow in fp16).
    """
    return any(not math.isfinite(v) for v in values)


def softmax(values):
    """Plain-Python softmax over a handful of option scores. Kept off the
    GPU deliberately: these are k<=8 numbers already pulled to the host,
    and computing it here keeps the scoring logic unit-testable without
    torch installed.

    NaN/inf-safe by construction. The naive version returns all-NaN if a
    SINGLE score is NaN, and that NaN propagates straight through
    gold_prob_mass into the Shapley values, making the whole item's
    attribution silently meaningless -- no error, just a record full of
    NaN. A non-finite score carries no information about that option, so
    it is floored below the finite scores instead of being allowed to
    destroy the others. Callers should pair this with
    `has_degenerate_scores` and drop the affected conditions.
    """
    vals = list(values)
    if not vals:
        return []
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return [1.0 / len(vals)] * len(vals)
    floor = min(finite) - 50.0
    clean = [v if math.isfinite(v) else floor for v in vals]
    top = max(clean)
    exps = [math.exp(v - top) for v in clean]
    total = sum(exps)
    if not math.isfinite(total) or total <= 0:
        return [1.0 / len(vals)] * len(vals)
    out = [e / total for e in exps]
    if not all(math.isfinite(x) for x in out):
        return [1.0 / len(vals)] * len(vals)
    return out


class HFVisionLanguageRunner(ModelRunner):
    """Expects each `Item` to carry, in `item.metadata`:
    - "pil_image": PIL.Image.Image (the clean base image)
    - "options": list[str] (multiple-choice options)
    - "correct_idx": int (index of the gold option in "options")
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        seed: int = 0,
        max_new_tokens: int = 48,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
        scoring: str = "letter",
        collect_traces: bool = False,
    ):
        """`scoring`: "letter" (1 forward pass, default) or "text"
        (k forward passes, no letter-mapping assumption). See the module
        docstring for why absolute gold-text probability was abandoned.

        `collect_traces`: run an extra generation pass to capture the
        model's free-text answer. Off by default -- it is the single most
        expensive part of a condition and is needed only by the Layer 3b
        drift signal (S1), not by scoring, attribution, or the
        manipulation check.
        """
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if scoring not in ("letter", "text"):
            raise ValueError(f"scoring must be 'letter' or 'text', got {scoring!r}")

        self.model_id = model_id
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.scoring = scoring
        self.collect_traces = collect_traces
        self._torch = torch
        # Letter token ids depend only on the chat template's generation
        # prompt, which is identical for every item, so they are resolved
        # once and reused. Resolving them per call was costing four full
        # multimodal preprocessing passes (image included) per model call.
        self._letter_id_cache = {}

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=quant_config, device_map="cuda"
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Public ModelRunner interface: apply the condition's factors, answer.
    # ------------------------------------------------------------------

    def run(self, item: Item, condition: Condition, n_decodes: int = 1) -> RunResult:
        """Retries once at half image resolution on CUDA OOM before giving
        up -- a transient/data-dependent OOM (unusually large source image)
        is common enough on a 16GB T4 that a single retry meaningfully
        improves the real success rate without masking a genuine failure
        (a second OOM still propagates)."""
        torch = self._torch
        try:
            return self._run_condition(item, condition, self.max_image_side)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._run_condition(item, condition, max(256, self.max_image_side // 2))

    def _run_condition(self, item: Item, condition: Condition, max_image_side: int) -> RunResult:
        base_image = _cap_resolution(item.metadata["pil_image"], max_image_side)
        options = list(item.metadata["options"])
        correct_idx = item.metadata["correct_idx"]
        factor_ids = [f.id for f in condition.applied_factors]
        outcome = apply_factors(
            base_image, item.question, options, correct_idx, factor_ids,
            seed_key=self._seed_key(item.item_id, condition.condition_id),
            metadata=item.metadata,
        )
        return self._answer(item.item_id, condition.condition_id, outcome.image,
                            outcome.question, options, correct_idx)

    def _seed_key(self, item_id: str, condition_id: str) -> str:
        """Single definition of the operator seed scheme. `describe_effects`
        must reproduce it exactly or the recorded distractor will not be the
        one the model saw."""
        return f"{self.seed}::{item_id}::{condition_id}"

    def describe_effects(self, item: Item, factor_ids) -> dict:
        """`{factor_id: PerturbationEffect}` for this item, matching what
        each factor's SINGLETON condition will actually inject.

        Populating `Item.perturbation_effects` from this is what makes the
        option-level signals (S2 attr-ratio, S4 uptake) live -- without it
        they have no distractor to measure capture by and return a constant
        0.0, which is exactly what happened in the first attribution run.
        """
        base_image = _cap_resolution(item.metadata["pil_image"], self.max_image_side)
        return describe_factor_effects(
            base_image,
            item.question,
            list(item.metadata["options"]),
            item.metadata["correct_idx"],
            list(factor_ids),
            seed_key_for=lambda fid: self._seed_key(
                item.item_id, make_condition_id(item.item_id, frozenset([_FactorKey(fid)]))
            ),
            metadata=item.metadata,
        )

    # ------------------------------------------------------------------
    # Public helper for natural-failure repairs and the language-delta
    # signal: answer a DIRECTLY-GIVEN question against the clean image,
    # bypassing factor perturbation entirely (used when the "condition"
    # isn't a point in the factor lattice -- e.g. a repair hypothesis, or
    # the native-language version of the same clean question).
    # ------------------------------------------------------------------

    def answer_with_question_override(
        self,
        item: Item,
        question_text: str,
        run_label: str,
        image_override=None,
        max_image_side: int | None = None,
    ) -> RunResult:
        """`image_override` (optional) replaces the item's own image --
        used by image-editing repair operators (e.g. the upscale repair
        in the natural-failure audit).

        `max_image_side` (optional) raises this call's resolution cap
        above the runner default. Without it an upscale repair would be
        silently undone: every real CVQA image is larger than the 768px
        default cap, so upscaling then re-capping at 768 returns the exact
        same pixels and the 'visual' repair would measure nothing while
        still costing a GPU call. The cap is never removed entirely --
        uncapped resolution is what caused the first full run to OOM on
        51/100 items.
        """
        torch = self._torch
        source_image = image_override if image_override is not None else item.metadata["pil_image"]
        cap = max_image_side or self.max_image_side
        base_image = _cap_resolution(source_image, cap)
        options = list(item.metadata["options"])
        correct_idx = item.metadata["correct_idx"]
        try:
            return self._answer(item.item_id, run_label, base_image, question_text, options, correct_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            small_image = _cap_resolution(source_image, max(256, cap // 2))
            return self._answer(item.item_id, run_label, small_image, question_text, options, correct_idx)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prefix(self, image, question: str, options: list) -> str:
        letters = OPTION_LETTERS[: len(options)]
        options_str = "\n".join(f"{ltr}) {opt}" for ltr, opt in zip(letters, options))
        if self.scoring == "letter":
            instruction = "Answer with the letter of the correct option only."
        else:
            instruction = "Answer with the option text only."
        prompt_text = f"{question}\nOptions:\n{options_str}\n{instruction}"
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}],
        }]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------
    # Core primitive: score every option, normalize across them.
    # ------------------------------------------------------------------

    def _answer(self, item_id: str, run_label: str, image, question: str, options: list, correct_idx: int) -> RunResult:
        torch = self._torch
        chat_prefix = self.build_prefix(image, question, options)
        prefix_inputs = self.processor(text=[chat_prefix], images=[image], return_tensors="pt").to(self.model.device)
        n_prefix_tokens = prefix_inputs["input_ids"].shape[1]

        if self.scoring == "letter":
            option_scores, used = self._score_letters(
                chat_prefix, prefix_inputs, n_prefix_tokens, len(options), image
            )
        else:
            option_scores = self._score_option_texts(chat_prefix, n_prefix_tokens, options, image)
            used = "text"

        degenerate = has_degenerate_scores(option_scores)
        option_probs = softmax(option_scores)
        gold_prob_mass = option_probs[correct_idx]
        predicted_idx = max(range(len(option_probs)), key=lambda j: option_probs[j])
        predicted_correct = predicted_idx == correct_idx

        # The model's confidence in whatever it actually chose. Replaces
        # the old generation-time top-token proxy: same intent (S5), but
        # defined on the option distribution we now trust, and obtained
        # without paying for a generation pass.
        gen_confidence = option_probs[predicted_idx]

        if self.collect_traces:
            with torch.no_grad():
                gen_out = self.model.generate(
                    **prefix_inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                )
            trace = self.processor.batch_decode(
                gen_out[:, n_prefix_tokens:], skip_special_tokens=True
            )[0]
            del gen_out
        else:
            # Not a real generation -- a readable rendering of the scored
            # choice, so downstream code and logs have something to show
            # without the cost of decoding. Anything that needs the
            # model's actual words must set collect_traces=True.
            trace = f"{OPTION_LETTERS[predicted_idx]}) {options[predicted_idx]}"

        result = RunResult(
            item_id=item_id,
            condition_id=run_label,
            sampled_answers=[trace],
            correct_flags=[predicted_correct],
            cot_traces=[trace],
            gold_prob_mass=gold_prob_mass,
            hidden_states=None,
            extra_signals={
                "gen_confidence": gen_confidence,
                "option_probs": option_probs,
                "predicted_idx": predicted_idx,
                # How far ahead the model's pick was over the gold option;
                # 0.0 exactly when the pick IS gold. A graded "how wrong"
                # that a binary correct flag cannot express.
                "margin": option_probs[predicted_idx] - gold_prob_mass,
                "scoring_used": used,
                # True when a raw score was NaN/inf and softmax had to
                # repair it. The value is usable but not trustworthy --
                # exclude these conditions from headline analysis.
                "degenerate_scores": degenerate,
            },
        )

        # Explicit cleanup: this runner processes many conditions
        # sequentially in one long-lived process, so unreleased
        # intermediate tensors can accumulate/fragment memory across
        # calls -- a contributing factor to the OOM failures in the first
        # full run.
        del prefix_inputs
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _letter_token_ids(self, chat_prefix, n_options):
        """Resolve each option letter's token id at the prompt boundary.

        Resolved IN CONTEXT (tokenizing `prefix + letter` and diffing
        against `prefix`) rather than tokenizing the letter standalone --
        standalone tokenization drifts at the boundary (leading-space
        handling differs by tokenizer), which would silently score the
        wrong token id and produce a plausible-looking but meaningless
        distribution.

        Text-only, and cached. A token id is a property of the text
        tokenizer; image placeholder tokens sit earlier in the sequence
        and cannot change it. Doing this through the full multimodal
        processor -- as the first version did -- re-preprocessed the
        image once per option on every single model call, which is pure
        waste. The cache key is the generation-prompt tail, so a change
        of chat template invalidates it rather than silently reusing
        stale ids.

        Returns None if any letter is not exactly one token, meaning
        letter scoring is unsafe for this tokenizer and the caller must
        fall back to full-text scoring.
        """
        key = (chat_prefix[-32:], n_options)
        if key in self._letter_id_cache:
            return self._letter_id_cache[key]

        tok = self.processor.tokenizer
        base = tok(chat_prefix, add_special_tokens=False)["input_ids"]
        ids = []
        for letter in OPTION_LETTERS[:n_options]:
            full = tok(chat_prefix + letter, add_special_tokens=False)["input_ids"]
            if len(full) != len(base) + 1 or full[: len(base)] != base:
                self._letter_id_cache[key] = None
                return None
            ids.append(full[-1])
        self._letter_id_cache[key] = ids
        return ids

    def _score_letters(self, chat_prefix, prefix_inputs, n_prefix_tokens, n_options, image):
        """Read the k option-letter logits out of ONE forward pass.

        The mode actually used is returned so it can be recorded per call
        -- a silent fallback to text scoring would otherwise make two
        runs incomparable without any visible sign.
        """
        torch = self._torch
        token_ids = self._letter_token_ids(chat_prefix, n_options)
        if token_ids is None:
            letters = list(OPTION_LETTERS[:n_options])
            return (
                self._score_option_texts(chat_prefix, n_prefix_tokens, letters, image),
                "text_fallback",
            )

        with torch.no_grad():
            out = self.model(**prefix_inputs)
        last_logits = out.logits[0, -1].float()
        scores = [last_logits[tid].item() for tid in token_ids]
        del out, last_logits
        return scores, "letter"

    def _score_option_texts(self, chat_prefix, n_prefix_tokens, continuations, image):
        """Teacher-force each continuation; return its MEAN per-token
        log-prob.

        Length-normalized on purpose: a summed log-prob makes longer
        options systematically less likely regardless of correctness,
        which is precisely the length bias that broke the original
        measure. Normalizing before the cross-option softmax keeps the
        comparison about content rather than token count.
        """
        torch = self._torch
        scores = []
        for continuation in continuations:
            full_inputs = self.processor(
                text=[chat_prefix + continuation], images=[image], return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model(**full_inputs)
            logits = out.logits[0]
            cont_ids = full_inputs["input_ids"][0].tolist()[n_prefix_tokens:]
            n_cont = len(cont_ids)
            if n_cont == 0 or logits.shape[0] < n_prefix_tokens + n_cont:
                # Degenerate tokenization -- score it as effectively
                # impossible rather than silently returning 0.0, which
                # softmax would turn into a competitive option.
                scores.append(-1e9)
            else:
                relevant = logits[n_prefix_tokens - 1 : n_prefix_tokens - 1 + n_cont]
                log_probs = torch.log_softmax(relevant.float(), dim=-1)
                scores.append(
                    sum(log_probs[j, cont_ids[j]].item() for j in range(n_cont)) / n_cont
                )
            del out, logits, full_inputs
        return scores
