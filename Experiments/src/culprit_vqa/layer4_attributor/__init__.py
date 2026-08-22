"""Layer 4 — amortized attributor (proposal §6.6).

Learns to predict alpha(x) from a single clean+perturbed signal pass,
trained against Layer 3a's measured Shapley attributions — an
amortization of the expensive 2^k lattice, not an independent source of
truth. Minimal-but-real for this scaffold: plain logistic regression;
GBT/MLP ablations are future thesis work.
"""

from culprit_vqa.layer4_attributor.dataset import AttributorExample, build_examples, to_arrays
from culprit_vqa.layer4_attributor.evaluate import (
    expected_calibration_error,
    spearman_correlation,
)
from culprit_vqa.layer4_attributor.logistic import LogisticAttributor

__all__ = [
    "AttributorExample",
    "build_examples",
    "to_arrays",
    "LogisticAttributor",
    "spearman_correlation",
    "expected_calibration_error",
]
