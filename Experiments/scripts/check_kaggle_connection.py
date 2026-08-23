"""Kaggle API connectivity smoke test (thin CLI wrapper).

Run: python scripts/check_kaggle_connection.py
Requires: pip install -e ".[kaggle]"  and a valid secrets.env in the repo root.

See culprit_vqa.kaggle_connectivity for the actual implementation, shared
with tests/test_kaggle_connection.py.
"""

import sys

from culprit_vqa.kaggle_connectivity import check_kaggle_connectivity

if __name__ == "__main__":
    sys.exit(check_kaggle_connectivity())
