"""S1 drift — embedding-style cosine distance + ROUGE-L delta between
clean and perturbed CoT traces (proposal §6.5, I-ScienceQA recipe).

Self-contained (no embedding-model dependency) for the scaffold: uses a
bag-of-words cosine similarity in place of a real sentence embedding, and
an LCS-based ROUGE-L. TODO(thesis): swap `_bow_cosine` for a real sentence
embedding model when Layer 2 gains real model backends.
"""

import re
from collections import Counter

from culprit_vqa.layer1_intervention.items import Item
from culprit_vqa.layer2_runner.base import RunResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bow_cosine(a: str, b: str) -> float:
    ca, cb = Counter(_tokenize(a)), Counter(_tokenize(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb.get(t, 0) for t in ca)
    norm_a = sum(v * v for v in ca.values()) ** 0.5
    norm_b = sum(v * v for v in cb.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l(a: str, b: str) -> float:
    """F-measure ROUGE-L over token sequences a, b; 0.0 for empty input."""
    toks_a, toks_b = _tokenize(a), _tokenize(b)
    if not toks_a or not toks_b:
        return 0.0
    lcs = _lcs_len(toks_a, toks_b)
    if lcs == 0:
        return 0.0
    precision = lcs / len(toks_a)
    recall = lcs / len(toks_b)
    return 2 * precision * recall / (precision + recall)


class DriftSignal:
    name = "s1_drift"

    def compute(self, item: Item, clean: RunResult, perturbed: RunResult) -> float:
        clean_trace = clean.cot_traces[0] if clean.cot_traces else ""
        pert_trace = perturbed.cot_traces[0] if perturbed.cot_traces else ""
        return (1.0 - _bow_cosine(clean_trace, pert_trace)) + (1.0 - _rouge_l(clean_trace, pert_trace))
