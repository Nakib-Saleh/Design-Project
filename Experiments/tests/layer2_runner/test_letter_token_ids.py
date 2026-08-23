"""Tests letter-token-id resolution and caching without torch/GPU.

This logic decides WHICH logit is read as "the model's score for option
B". A wrong id here would not raise -- it would silently produce a
plausible-looking distribution over the wrong tokens, and every Shapley
value downstream would be meaningless. Exactly the class of silent
measurement bug that already cost this project months (see the runner's
module docstring), so it gets tested directly.

`HFVisionLanguageRunner.__init__` imports torch and loads a model, so
these build the object with `object.__new__` and attach only the two
attributes the method touches.
"""

from culprit_vqa.layer2_runner.hf_runner import OPTION_LETTERS, HFVisionLanguageRunner

PREFIX = "<|im_start|>user\nWhat is this?<|im_end|>\n<|im_start|>assistant\n"


class FakeTokenizer:
    """Character-level stand-in: every character is its own token, id =
    ord(char). Good enough to exercise the boundary-diff logic, and it
    makes the expected ids checkable by hand."""

    def __init__(self):
        self.calls = 0

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        return {"input_ids": [ord(c) for c in text]}


class MultiTokenLetterTokenizer(FakeTokenizer):
    """Tokenizer where a letter becomes TWO tokens -- the degenerate case
    that must trigger the text-scoring fallback instead of scoring a
    truncated letter."""

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        ids = [ord(c) for c in text]
        if text != PREFIX:
            ids.append(-1)  # letter costs an extra token
        return {"input_ids": ids}


class FakeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _runner(tokenizer):
    r = object.__new__(HFVisionLanguageRunner)
    r.processor = FakeProcessor(tokenizer)
    r._letter_id_cache = {}
    return r


def test_resolves_the_correct_token_id_for_each_letter():
    r = _runner(FakeTokenizer())
    ids = r._letter_token_ids(PREFIX, 4)
    assert ids == [ord("A"), ord("B"), ord("C"), ord("D")]


def test_resolution_is_in_context_not_standalone():
    """Ids must come from diffing `prefix + letter` against `prefix`, so
    that boundary effects (leading-space handling) are captured. With the
    char-level fake, an in-context diff yields exactly the letter's own
    id -- and the tokenizer must have been shown the full prefix."""
    tok = FakeTokenizer()
    r = _runner(tok)
    r._letter_token_ids(PREFIX, 4)
    assert tok.calls == 5  # 1 baseline prefix + 4 prefix+letter


def test_ids_are_cached_across_calls():
    """The cache is the whole point of the optimization: resolving per
    call previously cost four multimodal preprocessing passes per model
    call."""
    tok = FakeTokenizer()
    r = _runner(tok)
    first = r._letter_token_ids(PREFIX, 4)
    calls_after_first = tok.calls
    second = r._letter_token_ids(PREFIX, 4)
    assert second == first
    assert tok.calls == calls_after_first  # no re-tokenization


def test_cache_is_keyed_so_a_different_option_count_recomputes():
    tok = FakeTokenizer()
    r = _runner(tok)
    four = r._letter_token_ids(PREFIX, 4)
    three = r._letter_token_ids(PREFIX, 3)
    assert len(four) == 4 and len(three) == 3
    assert three == four[:3]


def test_cache_invalidates_when_the_chat_template_tail_changes():
    """A different template must not silently reuse ids resolved for the
    old one -- that would score the wrong tokens with no error."""
    tok = FakeTokenizer()
    r = _runner(tok)
    r._letter_token_ids(PREFIX, 4)
    calls_after_first = tok.calls
    r._letter_token_ids(PREFIX + "ASSISTANT_V2:", 4)
    assert tok.calls > calls_after_first


def test_multi_token_letter_returns_none_to_force_text_fallback():
    """Scoring a truncated letter would silently read the wrong logit, so
    the unsafe case must be reported rather than approximated."""
    r = _runner(MultiTokenLetterTokenizer())
    assert r._letter_token_ids(PREFIX, 4) is None


def test_unsafe_result_is_cached_too():
    """The negative result must also be cached, or the expensive probe
    reruns on every single call for a tokenizer that can never work."""
    tok = MultiTokenLetterTokenizer()
    r = _runner(tok)
    assert r._letter_token_ids(PREFIX, 4) is None
    calls_after_first = tok.calls
    assert r._letter_token_ids(PREFIX, 4) is None
    assert tok.calls == calls_after_first


def test_letters_resolved_match_the_scoring_order():
    """Ids must come back in A, B, C, D order -- the scores list is
    indexed positionally against `options`, so any reordering would
    attribute each option's probability to the wrong answer."""
    r = _runner(FakeTokenizer())
    ids = r._letter_token_ids(PREFIX, 4)
    assert ids == [ord(ltr) for ltr in OPTION_LETTERS[:4]]
