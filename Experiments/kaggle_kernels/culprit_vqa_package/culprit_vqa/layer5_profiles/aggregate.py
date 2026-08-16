"""Per-group causal-factor mixture profiles (proposal §6.7)."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from culprit_vqa.pipeline import PipelineRecord


@dataclass
class FailureProfile:
    group_key: str
    factor_mixture: dict[str, float]
    n_items: int


def build_failure_profile(
    records: list[PipelineRecord], group_by: Callable[[PipelineRecord], str]
) -> dict[str, FailureProfile]:
    """Group records by `group_by` (e.g. item language, or a model id in
    metadata) and average each record's normalized alpha per factor id
    within the group."""
    groups: dict[str, list[PipelineRecord]] = defaultdict(list)
    for record in records:
        groups[group_by(record)].append(record)

    profiles: dict[str, FailureProfile] = {}
    for key, group_records in groups.items():
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for record in group_records:
            for factor, value in record.alpha.items():
                sums[factor.id] += value
                counts[factor.id] += 1
        mixture = {factor_id: sums[factor_id] / counts[factor_id] for factor_id in sums}
        profiles[key] = FailureProfile(group_key=key, factor_mixture=mixture, n_items=len(group_records))
    return profiles
