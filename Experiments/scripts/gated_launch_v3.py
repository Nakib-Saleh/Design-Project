"""Gate: culture check -> (if a third factor bites) k=3 attribution run.

Same shape as scripts/gated_launch.py, which ran the v2 chain end to end
without incident, and it reuses that module's Kaggle plumbing rather than
re-implementing it -- in particular `pull_kernel_output`, which judges a
download by whether the file landed on disk instead of by the CLI's exit
code. That distinction is what stopped a finished run from becoming
unreachable last time.

Three things this gate does that the v2 gate did not:

1. APPLIED SUBSETS. Two culture operators can no-op on an item (no native
   question, no named entity). Averaging an intervention over items it
   never touched drags a real effect toward zero, so each factor is
   judged only on the rows where it actually applied.

2. A POSITIVE CONTROL. `text_overlay_wrong_answer` measured +33.4% at
   n=698. If it does not reproduce here, the harness is broken and the
   gate refuses to launch regardless of what the culture factors did --
   a null result from a broken instrument is worthless.

3. A THIRD FACTOR IS REQUIRED, not just a second. At k=2 Shapley and
   leave-one-out rank factors identically by algebra (verified to
   1.1e-16 over 652 items), so a k=2 rerun cannot answer RQ1. If only the
   two known factors survive, launching would spend a session
   re-measuring something already measured, so the gate stops and says
   so.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle_kernels"))
sys.path.insert(0, str(REPO / "scripts"))

from gated_launch import (  # noqa: E402
    accounts_by_username, mcnemar_p, pull_kernel_output, push_kernel, wait_for,
)

CHECK_ACCOUNT = "mediumnunulabib"
CHECK_REF = "rsdaiyan/culprit-vqa-culture-check"
CHECK_DIR = REPO / "kaggle_kernels" / "culture_check"
CHECK_OUT = CHECK_DIR / "output"
CHECK_FILE = "culture_check_results.jsonl"
CHECK_TIMEOUT_S = 5 * 3600

POSITIVE_CONTROL_ID = "text_overlay_wrong_answer"
KNOWN_FACTORS = {"text_overlay_wrong_answer", "salience_recomposition"}
# Below this the positive control has not reproduced and the run is suspect.
# It measured 33.2% accuracy under the same intervention at n=698.
CONTROL_MAX_ACC = 0.50
MIN_APPLIED = 60          # a factor judged on fewer rows than this is underpowered
MIN_NEW_FACTORS = 1       # at least one CULTURE factor must bite -> k=3

ATTRIB_DIR = REPO / "kaggle_kernels" / "attribution_v3"
ATTRIB_OUT = ATTRIB_DIR / "output"
RUN_TIMEOUT_S = 9 * 3600

# (account, kernel id, results filename, item offset)
# The offset MUST equal N_ITEMS in attribution_v3.py or the accounts
# overlap / leave a gap. Verified offline against the stratified ordering.
LAUNCH_TARGETS = [
    ("NakibSaleh", "nakibsaleh/culprit-vqa-attribution-v3-a", "attribution_v3_a.jsonl", 0),
    ("chotonunulabib", "chotonunulabib/culprit-vqa-attribution-v3-b", "attribution_v3_b.jsonl", 400),
]

REPORT = REPO / "GATE_REPORT_V3.md"
PROGRESS = REPO / "GATE_PROGRESS_V3.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with PROGRESS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_records():
    path = CHECK_OUT / CHECK_FILE
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in rec and "factors" in rec:
            out.append(rec)
    return out


def evaluate(records):
    """Return (ok, kept_factors, report_lines)."""
    lines = []
    n_all = len(records)
    lines.append("# Gate report - culture check -> k=3 attribution\n")
    lines.append(f"Items returned: **{n_all}**\n")

    records = [r for r in records if not r.get("control_degenerate", False)]
    if n_all - len(records):
        lines.append(f"\n_Excluded {n_all - len(records)} item(s) with degenerate control scoring._\n")
    if len(records) < 100:
        lines.append(f"\n**GATE FAILED** - only {len(records)} usable records.\n")
        return False, [], lines

    ctrl_acc = sum(1 for r in records if r.get("control_correct")) / len(records)
    n_native = sum(1 for r in records if r.get("has_native_question"))
    lines.append(f"\nUsable items: **{len(records)}**  ")
    lines.append(f"Control accuracy: **{ctrl_acc:.1%}**  ")
    lines.append(f"Items with a distinct native-language question: **{n_native}** "
                 f"({n_native/len(records):.0%})\n")

    lines.append("\n## Did the interventions cause failures?\n")
    lines.append("Each factor is judged on the rows where it actually applied, and the "
                 "control accuracy is recomputed on that same subset so the drop is "
                 "like-for-like.\n")
    lines.append("| factor | applied | acc | drop | mean dp | broke | fixed | p | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    stats = {}
    for fid in sorted({f for r in records for f in r.get("factors", {})}):
        rows = [r for r in records
                if fid in r.get("factors", {})
                and r["factors"][fid].get("applied", True)
                and not r["factors"][fid].get("degenerate", False)]
        if not rows:
            lines.append(f"| {fid} | 0 | - | - | - | - | - | - | never applied |")
            stats[fid] = None
            continue
        acc = sum(1 for r in rows if r["factors"][fid]["correct"]) / len(rows)
        sub_ctrl = sum(1 for r in rows if r["control_correct"]) / len(rows)
        dp = sum(r["factors"][fid]["delta_p"] for r in rows) / len(rows)
        broke = sum(1 for r in rows if r["control_correct"] and not r["factors"][fid]["correct"])
        fixed = sum(1 for r in rows if not r["control_correct"] and r["factors"][fid]["correct"])
        p = mcnemar_p(broke, fixed)
        sig = p < 0.05 and broke > fixed
        powered = len(rows) >= MIN_APPLIED
        keep = sig and powered
        if fid == POSITIVE_CONTROL_ID:
            verdict = "positive control"
        elif keep:
            verdict = "**KEEP**"
        elif sig and not powered:
            verdict = f"significant but only n={len(rows)}"
        else:
            verdict = "inert"
        stats[fid] = dict(n=len(rows), acc=acc, drop=sub_ctrl - acc, broke=broke,
                          fixed=fixed, p=p, keep=keep)
        lines.append(f"| {fid} | {len(rows)} | {acc:.1%} | {sub_ctrl-acc:+.1%} | {dp:+.3f} | "
                     f"{broke} | {fixed} | {p:.3f} | {verdict} |")

    # --- positive control must reproduce -------------------------------
    pc = stats.get(POSITIVE_CONTROL_ID)
    if not pc:
        lines.append("\n**GATE FAILED** - the positive control produced no usable rows. "
                     "Nothing about this run can be trusted.\n")
        return False, [], lines
    if pc["acc"] > CONTROL_MAX_ACC:
        lines.append(
            f"\n**GATE FAILED** - positive control did not reproduce: accuracy "
            f"{pc['acc']:.1%} under `{POSITIVE_CONTROL_ID}`, expected ~33% "
            f"(it measured 33.2% at n=698). The harness is measuring something "
            f"different from last time, so a null result for the culture factors "
            f"would be uninterpretable. Nothing was launched.\n")
        return False, [], lines
    lines.append(f"\nPositive control reproduced (accuracy {pc['acc']:.1%} vs 33.2% "
                 f"previously) - the harness is measuring what it measured before.\n")

    # --- which culture factors bit? ------------------------------------
    culture_keep = [f for f, s in stats.items()
                    if s and s["keep"] and f != POSITIVE_CONTROL_ID and f not in KNOWN_FACTORS]

    if len(culture_keep) < MIN_NEW_FACTORS:
        lines.append(
            "\n**GATE STOPPED - no culture factor bit.**\n\n"
            "This is a result, not a malfunction. Three structurally different "
            "cultural interventions - a real native-language swap, a wrong-culture "
            "presupposition inside the question, and a wrong-culture caption on the "
            "image - all left the model unmoved, on top of the parenthetical "
            "`wrong_local_entity` that was already inert at n=698.\n\n"
            "The attribution run was NOT launched, deliberately: with only the two "
            "known factors there is no third factor, so the lattice would be k=2 "
            "again, and at k=2 Shapley and leave-one-out rank factors identically "
            "by algebra. The run would cost a session and could not answer RQ1.\n\n"
            "Report this as a negative result on cultural causal attribution. Four "
            "distinct mechanisms failing is a much stronger finding than one badly "
            "built operator failing.\n")
        return False, [], lines

    kept = culture_keep + [f for f in sorted(KNOWN_FACTORS)
                           if stats.get(f) and stats[f]["keep"]]
    # k <= 3 is a hard cap in generate_lattice; take the strongest.
    kept = sorted(kept, key=lambda f: -(stats[f]["broke"] - stats[f]["fixed"]))[:3]

    lines.append(f"\n## GATE PASSED\n\nFactors for the k=3 attribution run: "
                 f"**{', '.join(kept)}**\n")
    lines.append(f"\nAt k={len(kept)} the Shapley-vs-leave-one-out ordering comparison "
                 f"becomes testable for the first time"
                 f"{' (k=3)' if len(kept) == 3 else ''}.\n")
    return True, kept, lines


def configure(folder: Path, factors, offset, results_name, kernel_id, title):
    import re

    src = (folder / "attribution_v3.py").read_text(encoding="utf-8")
    src = re.sub(r'^SCORING_MODE = .*# GATE:SCORING$',
                 'SCORING_MODE = "letter"  # GATE:SCORING', src, flags=re.M)
    src = re.sub(r'^ACTIVE_FACTOR_IDS = .*# GATE:FACTORS$',
                 f'ACTIVE_FACTOR_IDS = {json.dumps(factors)}  # GATE:FACTORS', src, flags=re.M)
    src = re.sub(r'^ITEM_OFFSET = .*# GATE:OFFSET$',
                 f'ITEM_OFFSET = {offset}  # GATE:OFFSET', src, flags=re.M)
    src = re.sub(r'^RESULTS_PATH = .*# GATE:RESULTS$',
                 f'RESULTS_PATH = "/kaggle/working/{results_name}"  # GATE:RESULTS', src, flags=re.M)
    (folder / "attribution_v3.py").write_text(src, encoding="utf-8")

    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": title, "code_file": "attribution_v3.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["nakibsaleh/culprit-vqa-package"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")


def main():
    log("=== v3 gate started ===")
    accounts = accounts_by_username()

    status = wait_for(CHECK_REF, accounts[CHECK_ACCOUNT], "culture check", CHECK_TIMEOUT_S)
    log("pulling culture check output ...")
    pull_kernel_output(CHECK_REF, accounts[CHECK_ACCOUNT], CHECK_OUT, CHECK_FILE)
    records = load_records()
    log(f"culture check status={status}, usable records={len(records)}")

    if not records:
        REPORT.write_text(
            f"# Gate report\n\n**GATE FAILED** - culture check ended `{status}` with no "
            f"usable records. Nothing launched.\n", encoding="utf-8")
        log("GATE FAILED (no records)")
        return

    ok, factors, lines = evaluate(records)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not ok:
        log("GATE STOPPED. Nothing launched. See GATE_REPORT_V3.md")
        return

    launched = []
    for account, kernel_id, results_name, offset in LAUNCH_TARGETS:
        folder = REPO / "kaggle_kernels" / f"attribution_v3_{kernel_id[-1]}"
        shutil.rmtree(folder, ignore_errors=True)
        shutil.copytree(ATTRIB_DIR, folder,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "output"))
        configure(folder, factors, offset, results_name, kernel_id,
                  f"CULPRIT-VQA Attribution v3 {kernel_id[-1].upper()}")
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
        f.write(f"\nFactors: {', '.join(factors)}  \nLattice: k={len(factors)}\n")

    if not launched:
        log("nothing launched; stopping")
        return

    log(f"waiting for {len(launched)} attribution run(s) ...")
    outcomes = []
    for account, ref, results_name in launched:
        st = wait_for(ref, accounts[account], ref, RUN_TIMEOUT_S)
        path = pull_kernel_output(ref, accounts[account], ATTRIB_OUT, results_name)
        rows = 0
        if path:
            rows = sum(1 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
                       if ln.strip())
        outcomes.append((ref, st, path, rows))
        log(f"{ref}: status={st}, rows={rows}, file={path}")

    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n## Results pulled\n\n| kernel | status | rows | local file |\n|---|---|---|---|\n")
        for ref, st, path, rows in outcomes:
            rel = path.relative_to(REPO) if path else "NOT PULLED"
            f.write(f"| `{ref}` | {st} | {rows} | `{rel}` |\n")
        f.write(f"\n**Total records: {sum(r for _, _, _, r in outcomes)}**\n")
        f.write("\nAnalysis notes: filter `control_correct == true` for blame claims; "
                "`option_probs_by_condition` is now stored per condition, so the "
                "option-level signals can be recomputed offline without another run.\n")

    log(f"=== v3 gate complete: {sum(r for _, _, _, r in outcomes)} records ===")


if __name__ == "__main__":
    main()
