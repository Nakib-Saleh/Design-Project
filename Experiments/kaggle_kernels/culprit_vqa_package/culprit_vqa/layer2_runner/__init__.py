"""Layer 2 — model runner (proposal §6.3).

Minimal-but-real for this scaffold: a `ModelRunner` ABC plus a
deterministic `MockModelRunner` for synthetic/small-scale testing.
Real HF/API model backends are future thesis work.
"""

from culprit_vqa.layer2_runner.base import ModelRunner, RunResult
from culprit_vqa.layer2_runner.mock_runner import MockModelRunner

__all__ = ["ModelRunner", "RunResult", "MockModelRunner"]
