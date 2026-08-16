import pytest

from culprit_vqa.fixtures.synthetic_items import build_synthetic_items, build_synthetic_runner
from culprit_vqa.pipeline import run_pipeline_for_item


@pytest.fixture
def mock_runner():
    return build_synthetic_runner(seed=0)


@pytest.fixture
def synthetic_items():
    return build_synthetic_items()


@pytest.fixture(scope="session")
def pipeline_records():
    """Runs the full pipeline once per test session over all 5 synthetic
    items, for reuse across Layer 4/5 tests."""
    runner = build_synthetic_runner(seed=0)
    return [run_pipeline_for_item(item, factors, runner) for item, factors in build_synthetic_items()]
