"""LogisticAttributor — the amortized attributor (proposal §6.6).

Trains on Layer 3a labels (measured phi > 0), not injected-mode labels.
Reuses Layer 3a's `normalize_attribution` for `predict_alpha_for_item` so
ground-truth and amortized attribution share one null-case convention.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from culprit_vqa.layer3a_causal.attribution import normalize_attribution
from culprit_vqa.layer4_attributor.dataset import AttributorExample, to_arrays


class LogisticAttributor:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self._feature_names: list[str] = []
        self._fitted = False

    def fit(self, examples: list[AttributorExample]) -> None:
        X, y, self._feature_names = to_arrays(examples)
        if len(set(y.tolist())) < 2:
            # Degenerate synthetic split (all-harmful or all-not-harmful):
            # sklearn's LogisticRegression requires >= 2 classes. Fall back
            # to a trivial constant-probability model rather than raising,
            # so the smoke test can exercise this edge case cleanly.
            self._fitted = False
            self._constant_label = int(y[0]) if len(y) else 0
            return
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba_harmful(self, examples: list[AttributorExample]) -> np.ndarray:
        X, _, _ = to_arrays(examples, feature_names=self._feature_names)
        if not self._fitted:
            const = getattr(self, "_constant_label", 0)
            return np.full(len(examples), float(const))
        return self.model.predict_proba(X)[:, 1]

    def predict_alpha_for_item(
        self, examples_for_one_item: list[AttributorExample]
    ) -> dict[str, float]:
        """Predicted per-factor pseudo-Shapley score, normalized through
        the same null-case-aware `normalize_attribution` used in Layer 3a."""
        if not examples_for_one_item:
            return {}
        scores = self.predict_proba_harmful(examples_for_one_item) - 0.5
        pseudo_phi = {
            ex.factor_id: float(score)
            for ex, score in zip(examples_for_one_item, scores)
        }
        return normalize_attribution(pseudo_phi)
