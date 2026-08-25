"""Attribution-quality metrics for the amortized attributor (proposal §6.6, §8)."""

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def spearman_correlation(alpha_true: Sequence[float], alpha_pred: Sequence[float]) -> float:
    """Spearman rank correlation between ground-truth and predicted
    attribution. Returns NaN if either sequence is constant or too short
    for a rank correlation to be defined."""
    if len(alpha_true) < 2 or len(alpha_pred) < 2:
        return float("nan")
    result = spearmanr(alpha_true, alpha_pred)
    return float(result.correlation)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> float:
    """Standard binned |confidence - accuracy| ECE for a binary predictor."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return float(ece)
