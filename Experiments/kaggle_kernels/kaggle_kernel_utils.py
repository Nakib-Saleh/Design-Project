"""Reusable helpers for pushing/polling/pulling Kaggle script kernels across
the multiple accounts in secrets.env.

Not part of the culprit_vqa package (this drives *local* orchestration of
*remote* Kaggle runs, not anything that runs inside a kernel itself).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from culprit_vqa.kaggle_connectivity import load_all_kaggle_accounts  # noqa: E402

KAGGLE_EXE = REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe"

TERMINAL_STATES = {"complete", "error", "cancelled", "cancelling"}


def accounts_by_username() -> dict[str, str]:
    """{username: token} for every account in secrets.env."""
    return dict(load_all_kaggle_accounts())


def _run_kaggle(args: list[str], token: str, timeout_s: int = 300) -> subprocess.CompletedProcess:
    """`timeout_s` guards against the CLI hanging on a stalled connection.
    Without it a single wedged call silently kills a long poll loop --
    observed once: a status watcher logged one poll and then nothing for
    an hour, so a finished kernel was never reported. A timed-out call is
    returned as a normal failed CompletedProcess so callers (especially
    poll loops) can retry on the next tick instead of crashing."""
    import os

    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = token
    try:
        return subprocess.run(
            [str(KAGGLE_EXE), *args], env=env, capture_output=True, text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args, returncode=124, stdout="", stderr=f"kaggle CLI timed out after {timeout_s}s",
        )


def push_kernel(folder: Path, token: str) -> str:
    """Push a kernel folder (must contain kernel-metadata.json). Returns the
    kernel ref (username/slug) read back from the metadata file."""
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    result = _run_kaggle(["kernels", "push", "-p", str(folder)], token)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"kernel push failed for {folder}")
    # Kaggle may re-slug based on title; parse the printed URL for the real ref.
    for line in result.stdout.splitlines():
        if "kaggle.com/code/" in line:
            ref = line.strip().split("kaggle.com/code/")[-1]
            return ref
    return metadata["id"]  # fallback: as-specified id


def get_status(kernel_ref: str, token: str) -> str:
    result = _run_kaggle(["kernels", "status", kernel_ref], token)
    out = (result.stdout + result.stderr).strip()
    # e.g.: `<ref> has status "KernelWorkerStatus.RUNNING"`
    if '"' in out:
        raw = out.split('"')[1]
        return raw.split(".")[-1].lower()
    return out.lower()


def wait_for_completion(
    kernel_ref: str, token: str, poll_interval_s: int = 30, timeout_s: int = 3600
) -> str:
    """Poll until the kernel reaches a terminal state or timeout. Returns the
    final status string."""
    start = time.time()
    while True:
        status = get_status(kernel_ref, token)
        elapsed = time.time() - start
        print(f"[{elapsed:6.0f}s] {kernel_ref}: {status}")
        if status in TERMINAL_STATES:
            return status
        if elapsed > timeout_s:
            return f"timeout (last status: {status})"
        time.sleep(poll_interval_s)


def pull_output(kernel_ref: str, token: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = _run_kaggle(["kernels", "output", kernel_ref, "-p", str(out_dir)], token)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"kernel output pull failed for {kernel_ref}")
    return out_dir


def pull_log(kernel_ref: str, token: str) -> str:
    """Fetch the kernel's push/output CLI transcript (progress + any error
    text) — not a full stdout log (Kaggle doesn't expose that over the CLI;
    inspect the notebook's own printed/checkpointed JSON output instead)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result = _run_kaggle(["kernels", "output", kernel_ref, "-p", tmp], token)
        return result.stdout + result.stderr
