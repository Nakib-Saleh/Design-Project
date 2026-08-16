"""Kaggle API connectivity check, wrapped as a pytest test.

Marked `integration` (excluded by default via `pytest -m "not integration"`)
since it requires network access and real credentials. Skipped outright if
secrets.env is missing so the hermetic Layer 0-5 suite is never affected.
"""

import pytest

from culprit_vqa.kaggle_connectivity import (
    SECRETS_PATH,
    check_kaggle_connectivity,
    load_kaggle_credentials,
)

pytestmark = pytest.mark.integration


def test_kaggle_connectivity_end_to_end():
    if not SECRETS_PATH.exists():
        pytest.skip(f"secrets.env not found at {SECRETS_PATH}; skipping Kaggle connectivity check")

    try:
        load_kaggle_credentials()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"could not parse secrets.env: {exc}")

    assert check_kaggle_connectivity() == 0
