"""Interaction atlas: which factor pairs super-additively break which
models (proposal §6.7)."""

from collections import defaultdict

from culprit_vqa.pipeline import PipelineRecord


def build_interaction_atlas(records: list[PipelineRecord]) -> dict[frozenset[str], float]:
    """Mean pairwise interaction index (keyed by factor-id pair) across
    all records where both factors co-occur in the same item's assigned
    factor set."""
    sums: dict[frozenset[str], float] = defaultdict(float)
    counts: dict[frozenset[str], int] = defaultdict(int)
    for record in records:
        for pair, value in record.interactions.items():
            id_pair = frozenset(f.id for f in pair)
            sums[id_pair] += value
            counts[id_pair] += 1
    return {pair: sums[pair] / counts[pair] for pair in sums}
