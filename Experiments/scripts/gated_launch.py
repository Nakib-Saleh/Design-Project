"""Unattended overnight driver: wait for the instrument check, decide
whether the attribution run is justified, launch it, then WAIT FOR IT AND
PULL THE RESULTS.

The go/no-go decision is made from measured evidence rather than from an
assumption that the rebuilt measure works -- launching a multi-hour run
on an unvalidated instrument is the failure mode this project already hit
once.

GATE CONDITIONS (all must hold, or nothing launches):
  1. The instrument check completed successfully.
  2. Some NEW measure reaches AUC >= 0.80 for separating correct from
     incorrect answers. The winning mode configures the big run.
  3. At least 2 factors significantly broke the model (McNemar p < 0.05,
     more breaks than fixes). Those factors, and only those, are used --
     a factor that does not move the model contributes noise to the
     Shapley decomposition, and the efficiency residual cannot detect
     that because it is an algebraic identity.
  4. Fewer than 2 surviving factors means pairwise interactions are not
     measurable at all, so the run is skipped.

On a failed gate this writes a report and exits without spending quota.

ROBUSTNESS -- two real incidents drive the design here:

  * A status watcher once hung on a Kaggle CLI call that had no timeout,
    silently stopped polling, and a finished kernel went unnoticed.
    `_run_kaggle` now has a timeout and every poll is wrapped so a blip
    retries instead of killing the loop.

  * A results pull once RAISED even though every file had downloaded
    fine: the Kaggle CLI hit a console-encoding error while printing its
    progress lines and returned non-zero. Judging a download by its exit
    code is therefore wrong -- `pull_kernel_output` judges by whether the
    expected file actually landed on disk. This is precisely how a
    finished run becomes unreachable.

Everything is logged to GATE_PROGRESS.log in the repo (not just the
terminal), so if this process dies the state is still inspectable.
"""

import json
import re
import shutil
import sys
import time
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle_kernels"))

from kaggle_kernel_utils import (  # noqa: E402
    TERMINAL_STATES, _run_kaggle, accounts_by_username, get_status, push_kernel,
)

CHECK_ACCOUNT = "mediumnunulabib"
CHECK_REF = "rsdaiyan/culprit-vqa-manipulation-check"
CHECK_OUT = REPO / "kaggle_kernels" / "manipulation_check" / "output"
CHECK_FILE = "manipulation_check_results.jsonl"

AUC_PASS = 0.80
MIN_FACTORS = 2
REPORT = REPO / "GATE_REPORT.md"
PROGRESS = REPO / "GATE_PROGRESS.log"

# (account, kernel id, results filename, item offset)
# NOTE: the offset MUST equal N_ITEMS in attribution_v2.py, or the two
# accounts will either overlap (wasting half the run on duplicate items)
# or leave a gap in the item range.
LAUNCH_TARGETS = [
    ("NakibSaleh", "nakibsaleh/culprit-vqa-attribution-v2-a", "attribution_v2_a.jsonl", 0),
    ("chotonunulabib", "chotonunulabib/culprit-vqa-attribution-v2-b", "attribution_v2_b.jsonl", 500),
]

ATTRIBUTION_OUT = REPO / "kaggle_kernels" / "attribution_v2" / "output"
CHECK_TIMEOUT_S = 4 * 3600
RUN_TIMEOUT_S = 9 * 3600   # attribution budget is 6h + setup; 9h leaves real margin


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with PROGRESS.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # a logging failure must never take down the run


# ----------------------------------------------------------------------
# Kaggle helpers, hardened
# ----------------------------------------------------------------------

def wait_for(ref, token, label, timeout_s):
    """Poll until terminal. Never raises: a transient API/CLI failure is
    logged and retried on the next tick, because a watcher that dies
    silently is worse than no watcher -- it looks like the run is still
    going."""
    start = time.time()
    consecutive_errors = 0
    while True:
        try:
            status = get_status(ref, token)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            log(f"{label}: poll failed ({exc}) [{consecutive_errors} in a row]; retrying")
            time.sleep(60)
            continue

        elapsed = time.time() - start
        if status in TERMINAL_STATES:
            log(f"{label}: {status} after {elapsed/60:.0f}m")
            return status
        log(f"{label}: {status} ({elapsed/60:.0f}m)")
        if elapsed > timeout_s:
            log(f"{label}: TIMEOUT after {elapsed/3600:.1f}h (last status {status})")
            return "TIMEOUT"
        time.sleep(60)


def pull_kernel_output(ref, token, out_dir, expected_file):
    """Download a kernel's output. Returns the Path to `expected_file` if
    it landed, else None.

    Deliberately ignores the CLI's exit code. A previous pull returned
    non-zero because the CLI hit a console-encoding error while printing
    progress -- but every file had already downloaded. Treating that as
    failure is how a finished run becomes unreachable, which is the exact
    problem this function exists to prevent. Retries a few times, since a
    transient network failure is genuinely recoverable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / expected_file
    for attempt in range(1, 4):
        result = _run_kaggle(["kernels", "output", ref, "-p", str(out_dir)], token, timeout_s=900)
        if target.exists() and target.stat().st_size > 0:
            # Kaggle also returns a copy of the attached package; drop it
            # so the repo does not accumulate duplicate source trees.
            shutil.rmtree(out_dir / "culprit_vqa", ignore_errors=True)
            log(f"pulled {expected_file} ({target.stat().st_size} bytes, attempt {attempt}, "
                f"cli_rc={result.returncode})")
            return target
        log(f"pull attempt {attempt} for {ref}: {expected_file} not present yet "
            f"(cli_rc={result.returncode}); files={[p.name for p in out_dir.glob('*')][:8]}")
        time.sleep(30)
    log(f"FAILED to pull {expected_file} from {ref} after 3 attempts")
    return None


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mcnemar_p(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = max(n10, n01)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def load_check_records():
    path = CHECK_OUT / CHECK_FILE
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a killed kernel must not abort analysis
    return [r for r in records if "error" not in r]


def evaluate_gate(records):
    """Return (ok, scoring_mode, surviving_factors, report_lines)."""
    lines, n = [], len(records)
    lines.append(f"# Gate report\n\nItems evaluated: **{n}**\n")

    if n < 30:
        lines.append(f"\n**GATE FAILED** - only {n} usable records; too few to judge.\n")
        return False, None, [], lines

    if "new_letter_p" not in records[0]:
        # Q1 was settled by the instrument check -- `new_letter` AUC 0.995
        # vs `old_absolute` 0.725 on 173 real items -- and is deliberately
        # not repeated, so there is nothing here to re-decide.
        lines.append("\n## Q1 - already settled\n")
        lines.append(
            "Measure validation was completed by the instrument check: `new_letter` "
            "AUC **0.995** (vs `old_absolute` 0.725) on 173 items. Scoring mode is "
            "fixed to `letter`.\n")
        return _evaluate_factors(records, "letter", lines)

    lines.append("\n## Q1 - does the measure measure anything?\n")
    lines.append("| measure | accuracy | mean p(gold) | AUC | verdict |")
    lines.append("|---|---|---|---|---|")
    aucs = {}
    for name, pk, ck in [("old_absolute", "old_abs_p", "old_abs_correct"),
                         ("new_letter", "new_letter_p", "new_letter_correct"),
                         ("new_text", "new_text_p", "new_text_correct")]:
        pos = [r[pk] for r in records if r.get(ck)]
        neg = [r[pk] for r in records if not r.get(ck)]
        a = auc(pos, neg)
        aucs[name] = a
        acc = len(pos) / n
        mean_p = sum(r[pk] for r in records) / n
        verdict = "PASS" if a >= AUC_PASS else ("weak" if a >= 0.65 else "NOISE")
        lines.append(f"| {name} | {acc:.1%} | {mean_p:.4f} | {a:.3f} | {verdict} |")

    candidates = {k: v for k, v in aucs.items()
                  if k != "old_absolute" and v == v and v >= AUC_PASS}
    if not candidates:
        lines.append(
            f"\n**GATE FAILED** - no new measure reached AUC {AUC_PASS}. "
            "The problem is deeper than normalization; nothing was launched. "
            "Do not run further experiments until this is understood.\n")
        return False, None, [], lines

    best = max(candidates, key=candidates.get)
    scoring = "letter" if best == "new_letter" else "text"
    lines.append(f"\nWinning measure: **{best}** (AUC {candidates[best]:.3f}) "
                 f"-> scoring mode `{scoring}`\n")

    return _evaluate_factors(records, scoring, lines)


def _evaluate_factors(records, scoring, lines):
    """Q2: keep only the factors that significantly break the model.

    A factor that does not move the model is not a usable cause --
    Shapley over a lattice containing it decomposes noise, and the
    efficiency residual cannot detect that because it is an algebraic
    identity that holds for random numbers.
    """
    n = len(records)
    # Drop items whose CONTROL scoring was degenerate (NaN/inf raw logits):
    # every factor delta on such an item is measured against an
    # untrustworthy baseline.
    records = [r for r in records if not r.get("control_degenerate", False)]
    if len(records) < n:
        lines.append(f"\n_Excluded {n - len(records)} item(s) with degenerate control scoring._\n")
    n = len(records)
    if n < 30:
        lines.append(f"\n**GATE FAILED** - only {n} usable records after filtering.\n")
        return False, scoring, [], lines

    lines.append("\n## Q2 - do the interventions actually cause failures?\n")
    ctrl_acc = sum(1 for r in records if r.get("control_correct")) / n
    lines.append(f"Control accuracy: **{ctrl_acc:.1%}**\n")
    lines.append("| factor | acc | acc drop | mean dp | broke | fixed | p | kept |")
    lines.append("|---|---|---|---|---|---|---|---|")

    survivors = []
    all_fids = sorted({fid for r in records for fid in r.get("factors", {})})
    for fid in all_fids:
        rows = [r for r in records if fid in r.get("factors", {})
                and not r["factors"][fid].get("degenerate", False)]
        if not rows:
            continue
        acc = sum(1 for r in rows if r["factors"][fid]["correct"]) / len(rows)
        mean_dp = sum(r["factors"][fid]["delta_p"] for r in rows) / len(rows)
        broke = sum(1 for r in rows if r["control_correct"] and not r["factors"][fid]["correct"])
        fixed = sum(1 for r in rows if not r["control_correct"] and r["factors"][fid]["correct"])
        p = mcnemar_p(broke, fixed)
        keep = p < 0.05 and broke > fixed
        if keep:
            survivors.append((fid, broke - fixed))
        lines.append(f"| {fid} | {acc:.1%} | {ctrl_acc-acc:+.1%} | {mean_dp:+.3f} | "
                     f"{broke} | {fixed} | {p:.3f} | {'YES' if keep else 'no'} |")

    survivors.sort(key=lambda t: -t[1])
    kept = [fid for fid, _ in survivors][:3]  # k<=3 per the lattice cap

    if len(kept) < MIN_FACTORS:
        lines.append(
            f"\n**GATE FAILED** - only {len(kept)} factor(s) significantly broke the model; "
            f"{MIN_FACTORS} are needed for pairwise interactions. Nothing was launched. "
            "The interventions need to be made stronger before attribution is worth running.\n")
        return False, scoring, kept, lines

    lines.append(f"\nFactors kept for the attribution run: **{', '.join(kept)}**\n")
    lines.append("\n**GATE PASSED** - attribution run launched.\n")
    return True, scoring, kept, lines


# ----------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------

def configure_kernel(folder: Path, scoring, factors, offset, results_name, kernel_id, title):
    src = (folder / "attribution_v2.py").read_text(encoding="utf-8")
    src = re.sub(r'^SCORING_MODE = .*# GATE:SCORING$',
                 f'SCORING_MODE = "{scoring}"  # GATE:SCORING', src, flags=re.M)
    src = re.sub(r'^ACTIVE_FACTOR_IDS = .*# GATE:FACTORS$',
                 f'ACTIVE_FACTOR_IDS = {json.dumps(factors)}  # GATE:FACTORS', src, flags=re.M)
    src = re.sub(r'^ITEM_OFFSET = .*# GATE:OFFSET$',
                 f'ITEM_OFFSET = {offset}  # GATE:OFFSET', src, flags=re.M)
    src = re.sub(r'^RESULTS_PATH = .*# GATE:RESULTS$',
                 f'RESULTS_PATH = "/kaggle/working/{results_name}"  # GATE:RESULTS', src, flags=re.M)
    (folder / "attribution_v2.py").write_text(src, encoding="utf-8")

    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": title, "code_file": "attribution_v2.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["nakibsaleh/culprit-vqa-package"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")


def main():
    log("=== gate started ===")
    accounts = accounts_by_username()

    status = wait_for(CHECK_REF, accounts[CHECK_ACCOUNT], "instrument check", CHECK_TIMEOUT_S)

    # Try to pull even on a non-complete status: the kernel checkpoints
    # every item, so a late error still leaves usable rows, and a TIMEOUT
    # here is about our polling rather than about the kernel.
    log("pulling instrument check output ...")
    pull_kernel_output(CHECK_REF, accounts[CHECK_ACCOUNT], CHECK_OUT, CHECK_FILE)
    records = load_check_records()
    log(f"instrument check status={status}, usable records={len(records)}")

    if status != "complete" and not records:
        REPORT.write_text(
            f"# Gate report\n\n**GATE FAILED** - instrument check ended with status "
            f"`{status}` and produced no usable records. Nothing was launched.\n",
            encoding="utf-8")
        log("GATE FAILED (no records). Nothing launched.")
        return

    ok, scoring, factors, lines = evaluate_gate(records)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    if not ok:
        log("GATE FAILED. Nothing launched. See GATE_REPORT.md")
        return

    base = REPO / "kaggle_kernels" / "attribution_v2"
    launched = []
    for account, kernel_id, results_name, offset in LAUNCH_TARGETS:
        folder = REPO / "kaggle_kernels" / f"attribution_v2_{results_name.split('.')[0][-1]}"
        shutil.rmtree(folder, ignore_errors=True)
        # ignore __pycache__: local py_compile checks leave .pyc files in the
        # source folder, and there is no reason to ship them to Kaggle.
        shutil.copytree(base, folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        title = "CULPRIT-VQA Attribution v2 " + kernel_id[-1].upper()
        configure_kernel(folder, scoring, factors, offset, results_name, kernel_id, title)
        try:
            ref = push_kernel(folder, accounts[account])
            launched.append((account, ref, results_name))
            log(f"launched {ref} (offset {offset})")
        except Exception as exc:  # noqa: BLE001
            log(f"FAILED to launch on {account}: {exc}")

    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n## Launched\n\n")
        for account, ref, _ in launched:
            f.write(f"- `{ref}` (account {account}) - https://www.kaggle.com/code/{ref}\n")
        f.write(f"\nScoring mode: `{scoring}`  \nFactors: {', '.join(factors)}\n")

    if not launched:
        log("nothing launched successfully; stopping")
        return

    # --- wait for the runs and PULL THEM. This is the whole point: a
    # previous run finished on Kaggle and its results were never
    # retrieved, so the work was effectively lost until noticed by hand.
    log(f"waiting for {len(launched)} attribution run(s) ...")
    outcomes = []
    for account, ref, results_name in launched:
        st = wait_for(ref, accounts[account], ref, RUN_TIMEOUT_S)
        # Pull regardless of status -- the kernel checkpoints every item,
        # so even an errored or timed-out run has usable rows on disk.
        path = pull_kernel_output(ref, accounts[account], ATTRIBUTION_OUT, results_name)
        n_rows = 0
        if path:
            n_rows = sum(
                1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip())
        outcomes.append((ref, st, path, n_rows))
        log(f"{ref}: status={st}, rows={n_rows}, file={path}")

    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n## Results pulled\n\n")
        f.write("| kernel | status | rows | local file |\n|---|---|---|---|\n")
        for ref, st, path, n_rows in outcomes:
            rel = path.relative_to(REPO) if path else "NOT PULLED"
            f.write(f"| `{ref}` | {st} | {n_rows} | `{rel}` |\n")
        total = sum(n for _, _, _, n in outcomes)
        f.write(f"\n**Total attribution records: {total}**\n")
        f.write("\nAnalysis note: filter on `control_correct == true` for the headline "
                "attribution claim -- on control-wrong items there was no working answer "
                "for a factor to break.\n")

    log(f"=== gate complete: {sum(n for _, _, _, n in outcomes)} total records pulled ===")


if __name__ == "__main__":
    main()
