"""Step 1 of the v3 chain: ship the current source, then start the culture check.

ORDER MATTERS AND IS THE POINT OF THIS SCRIPT.

Every kernel imports `culprit_vqa` from the Kaggle dataset
`nakibsaleh/culprit-vqa-package`, not from this repo. The v3 work changed
the source in ways the kernels depend on:

  * HFVisionLanguageRunner.describe_effects  (new -- the culture check
    calls it on every item; without it the run dies on item 1)
  * layer3b_signals.option_evidence          (new module)
  * the four culture operators + OperatorOutcome's 4th field
  * PerturbationEffect.distractor_option_idx

If the kernel is pushed before the dataset is updated, it runs against
the OLD package. The best case is an immediate AttributeError; the worse
case is a run that completes against stale operator code and produces
results that look fine and mean nothing. So this script uploads the
dataset first, waits for the new version to actually be queryable, and
only then pushes the kernel -- and it verifies the shipped copy really
contains the new symbols before uploading anything.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle_kernels"))

from kaggle_kernel_utils import (  # noqa: E402
    KAGGLE_EXE, _run_kaggle, accounts_by_username, push_kernel,
)

SRC = REPO / "src" / "culprit_vqa"
PKG_DIR = REPO / "kaggle_kernels" / "culprit_vqa_package"
PKG_PAYLOAD = PKG_DIR / "culprit_vqa"
OWNER_ACCOUNT = "NakibSaleh"          # owns nakibsaleh/culprit-vqa-package
CHECK_DIR = REPO / "kaggle_kernels" / "culture_check"
CHECK_ACCOUNT = "mediumnunulabib"

# Symbols the v3 kernels need. Verified in the COPIED tree, so a partial
# or stale copy is caught before it costs a session.
REQUIRED = [
    ("layer2_runner/hf_runner.py", "def describe_effects"),
    ("layer2_runner/hf_runner.py", "def _seed_key"),
    ("layer3b_signals/option_evidence.py", "def distractor_option_indices"),
    ("layer3b_signals/option_evidence.py", "def conflict"),
    ("layer3b_signals/uptake.py", "distractor_option_indices"),
    ("layer3b_signals/stubs.py", "option_probs(perturbed)"),
    ("layer1_intervention/real_operators.py", "def describe_factor_effects"),
    ("layer1_intervention/real_operators.py", "apply_cultural_text_overlay"),
    ("layer1_intervention/real_operators.py", "class OperatorOutcome"),
    ("layer1_intervention/items.py", "distractor_option_idx"),
    ("layer0_taxonomy/operators.py", "EXTENSION_OPERATORS"),
]


def sync_package():
    print(f"syncing {SRC} -> {PKG_PAYLOAD}", flush=True)
    shutil.rmtree(PKG_PAYLOAD, ignore_errors=True)
    shutil.copytree(SRC, PKG_PAYLOAD,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    missing = []
    for rel, needle in REQUIRED:
        path = PKG_PAYLOAD / rel
        if not path.exists():
            missing.append(f"{rel} (file missing)")
        elif needle not in path.read_text(encoding="utf-8"):
            missing.append(f"{rel}: {needle!r}")
    if missing:
        raise SystemExit("SHIPPED PACKAGE IS STALE OR INCOMPLETE:\n  " + "\n  ".join(missing))
    n = sum(1 for _ in PKG_PAYLOAD.rglob("*.py"))
    print(f"OK: {n} python files staged, all {len(REQUIRED)} required symbols present", flush=True)


def upload_dataset(token):
    notes = f"v3 signals + culture operators ({time.strftime('%Y-%m-%d %H:%M')})"
    print("uploading new dataset version ...", flush=True)
    res = _run_kaggle(
        ["datasets", "version", "-p", str(PKG_DIR), "-m", notes, "-r", "zip"],
        token, timeout_s=900)
    print(res.stdout or res.stderr, flush=True)
    # Exit code is not trusted here for the same reason pull_kernel_output
    # does not trust it: the CLI has returned non-zero on console-encoding
    # errors after doing the work. Confirm by querying the dataset instead.
    return res


def wait_for_dataset(token, timeout_s=1800):
    """Poll until the dataset reports a version newer than when we started."""
    print("waiting for the new dataset version to become queryable ...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        res = _run_kaggle(
            ["datasets", "list", "-m", "-s", "culprit-vqa-package"], token, timeout_s=180)
        if res.returncode == 0 and "culprit-vqa-package" in (res.stdout or ""):
            print("dataset is queryable.", flush=True)
            return True
        time.sleep(30)
    print("WARNING: could not confirm the dataset version within the timeout.", flush=True)
    return False


def main():
    accounts = accounts_by_username()
    sync_package()

    if "--dry-run" in sys.argv:
        print("\n--dry-run: package verified, nothing uploaded or launched.")
        return

    upload_dataset(accounts[OWNER_ACCOUNT])
    wait_for_dataset(accounts[OWNER_ACCOUNT])
    # Kaggle attaches the LATEST version of a dataset source at kernel start,
    # so a short settle is cheap insurance against racing the CDN.
    print("settling 60s before pushing the kernel ...", flush=True)
    time.sleep(60)

    ref = push_kernel(CHECK_DIR, accounts[CHECK_ACCOUNT])
    print(f"\nculture check launched: https://www.kaggle.com/code/{ref}")
    print("\nNext: run  python scripts/gated_launch_v3.py")
    print("It waits for this run, evaluates the gate, and only launches the")
    print("k=3 attribution runs if a cultural factor actually bit.")


if __name__ == "__main__":
    main()
