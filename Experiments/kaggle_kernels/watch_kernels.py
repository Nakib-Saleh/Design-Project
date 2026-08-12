"""Poll one or more Kaggle kernels until all reach a terminal state.

Usage: python kaggle_kernels/watch_kernels.py <account>:<kernel_ref> [...]

Written to survive a poll loop running for hours unattended:
  * a transient API error or CLI timeout is logged and retried on the
    next tick rather than killing the loop (a watcher that dies silently
    is worse than no watcher -- it looks like the run is still going),
  * every line is flushed, so the log file reflects live progress,
  * a status is only believed to be terminal when the CLI actually
    returned one of the known terminal states, so a network error string
    can never be mistaken for "error"/"complete".
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_kernel_utils import TERMINAL_STATES, accounts_by_username, get_status  # noqa: E402

POLL_INTERVAL_S = 60
TIMEOUT_S = 6 * 3600


def main():
    targets = []
    for spec in sys.argv[1:]:
        account, _, ref = spec.partition(":")
        targets.append((account, ref))
    if not targets:
        print("usage: watch_kernels.py <account>:<kernel_ref> [...]")
        return 1

    accounts = accounts_by_username()
    pending = {ref: account for account, ref in targets}
    final = {}
    start = time.time()

    while pending:
        elapsed = time.time() - start
        for ref in list(pending):
            account = pending[ref]
            try:
                status = get_status(ref, accounts[account])
            except Exception as exc:  # noqa: BLE001 -- never let a blip kill the loop
                print(f"[{elapsed:6.0f}s] {ref}: poll failed ({exc}), retrying", flush=True)
                continue
            print(f"[{elapsed:6.0f}s] {ref}: {status}", flush=True)
            if status in TERMINAL_STATES:
                final[ref] = status
                del pending[ref]
        if not pending:
            break
        if elapsed > TIMEOUT_S:
            for ref in pending:
                final[ref] = "TIMEOUT"
            break
        time.sleep(POLL_INTERVAL_S)

    print("\n=== FINAL ===", flush=True)
    for ref, status in final.items():
        print(f"{ref}: {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
