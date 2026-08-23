"""Kaggle API connectivity check — credential loading + the actual check.

Lives in the installed package (not scripts/) so both
`scripts/check_kaggle_connection.py` (a thin CLI wrapper) and
`tests/test_kaggle_connection.py` can import it reliably regardless of how
pytest/python is invoked, rather than depending on `scripts/` being on
sys.path.

Separate from the hermetic, synthetic Layer 0-5 pipeline: this proves the
environment can actually reach a real external service using real
credentials from secrets.env (repo root, gitignored, never committed).

IMPORTANT auth quirk discovered empirically (Sept 2026, kaggle CLI 2.2.4 /
kagglesdk 0.1.37): the legacy `KAGGLE_USERNAME`/`KAGGLE_KEY` env-var pair
authenticates the OLD dataset endpoints (`dataset_list`,
`dataset_download_files`, etc. via `KaggleApi`), but the newer
kagglesdk-backed endpoint groups — kernels, competitions — require the
token to also be exposed as `KAGGLE_API_TOKEN`. The `secrets.env` value
(prefixed `KGAT_...`) is actually this newer-style token. We set BOTH env
vars when activating an account so every code path (old and new) works.
"""

import os
import re
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECRETS_PATH = REPO_ROOT / "secrets.env"

# secrets.env format (as provided): `NakibSaleh == "KGAT_..."`, one line per
# account, i.e. `<username> == "<token>"` — not standard KEY=value .env syntax.
_SECRETS_LINE_RE = re.compile(r'^\s*(?P<username>\S+)\s*==\s*"(?P<key>[^"]+)"\s*$')


def load_all_kaggle_accounts(secrets_path: Path = SECRETS_PATH) -> list[tuple[str, str]]:
    """Parse every `<username> == "<token>"` line in secrets.env.

    Does not print any raw token value.
    """
    if not secrets_path.exists():
        raise FileNotFoundError(f"secrets.env not found at {secrets_path}")
    accounts = []
    for line in secrets_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        match = _SECRETS_LINE_RE.match(line)
        if match:
            accounts.append((match.group("username"), match.group("key")))
    if not accounts:
        raise ValueError(
            f"Could not parse any credential line from {secrets_path}. "
            'Expected one or more lines of the form: <username> == "<token>"'
        )
    return accounts


def load_kaggle_credentials(secrets_path: Path = SECRETS_PATH) -> tuple[str, str]:
    """Backward-compatible: returns the FIRST account's (username, token)."""
    return load_all_kaggle_accounts(secrets_path)[0]


def activate_kaggle_account(username: str, token: str) -> None:
    """Set the env vars needed for BOTH the legacy dataset endpoints and the
    newer kagglesdk-backed endpoints (kernels, competitions) — see module
    docstring for why both are required."""
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = token
    os.environ["KAGGLE_API_TOKEN"] = token


def check_kaggle_connectivity(secrets_path: Path = SECRETS_PATH) -> int:
    """Authenticate to the Kaggle API (first account in secrets.env) and
    download one small real dataset.

    Returns 0 on success, 1 on any failure. Prints progress/failure
    messages but never the raw credential value.
    """
    try:
        username, token = load_kaggle_credentials(secrets_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED to load credentials: {exc}")
        return 1

    activate_kaggle_account(username, token)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "FAILED: the `kaggle` package is not installed. "
            'Run: pip install -e ".[kaggle]"'
        )
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 — surface any auth failure plainly
        print(f"FAILED to authenticate as {username!r}: {exc}")
        return 1

    print(f"Authenticated to Kaggle API as: {username}")

    try:
        datasets = api.dataset_list(search="titanic")
        print(f"Auth check OK - dataset search returned {len(datasets)} result(s).")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED during dataset_list auth check: {exc}")
        return 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Kaggle's canonical tiny public dataset (~60KB) — used purely
            # to confirm the download mechanics work end-to-end.
            api.dataset_download_files("heptapod/titanic", path=tmp_dir, unzip=True, quiet=False)
            downloaded = list(Path(tmp_dir).iterdir())
            print(f"Downloaded {len(downloaded)} file(s) to a temp dir: {[f.name for f in downloaded]}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED to download sample dataset: {exc}")
            return 1

    print("\nKaggle API connectivity check PASSED.")
    return 0
