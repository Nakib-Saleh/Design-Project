"""CULPRIT-VQA manipulation check v2 -- do the interventions actually
cause failures, at enough scale to tell?

Q1 (does the measure measure anything) is SETTLED and is deliberately not
repeated here. The instrument check answered it on 173 real items:

    old_absolute  AUC 0.725   (accuracy 52.6%)
    new_letter    AUC 0.995   (accuracy 67.1%)   <- winner
    new_text      AUC 0.983   (accuracy 45.1%)

Dropping it matters for more than time. The old measure's reimplementation
was the only code path here WITHOUT the runner's out-of-memory retry, and
it cost 27/200 items to CUDA OOM. Removing it removes that failure mode.

What remains is Q2, which the instrument check could not settle:

    factor                       acc drop   broke/fixed   p
    text_overlay_wrong_answer      +28.9%       51/1     0.000  <- real
    salience_recomposition          +5.2%       20/10    0.099  <- unresolved
    irrelevant_plausible_fact       +1.2%        8/6     0.791  <- inert
    wrong_local_entity              -1.2%        5/7     0.774  <- inert

Only one factor cleared significance, and Shapley needs at least two for
pairwise interactions, so the attribution run was correctly blocked. The
open question is whether `salience_recomposition` is genuinely weak or
merely underpowered: at 173 items its 20-vs-10 split gives p=0.099, and
the same ratio would reach p=0.014 at twice the items.

So: same four factors, letter scoring only, ~4x the items. Either
salience clears the bar and attribution becomes possible with two real
factors, or it does not and the intervention library needs redesigning
before any attribution run is worth the GPU time.

Writes /kaggle/working/manipulation_check_results.jsonl.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("=== Bootstrapping culprit_vqa package from the attached dataset ===", flush=True)
print(f"/kaggle/input: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'MISSING'}", flush=True)

_CANDIDATE_SRCS = [
    "/kaggle/input/culprit-vqa-package",
    "/kaggle/input/culprit-vqa-package/culprit_vqa",
]
_SRC = None
for candidate in _CANDIDATE_SRCS:
    if os.path.exists(candidate) and os.path.exists(os.path.join(candidate, "__init__.py")):
        _SRC = candidate
        break
if _SRC is None and os.path.exists("/kaggle/input"):
    for root_name in os.listdir("/kaggle/input"):
        root_path = os.path.join("/kaggle/input", root_name)
        if not os.path.isdir(root_path):
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            if "__init__.py" in filenames and "layer0_taxonomy" in dirnames:
                _SRC = dirpath
                break
        if _SRC:
            break
if _SRC is None:
    raise FileNotFoundError("Could not locate culprit_vqa package under /kaggle/input")
print(f"Found culprit_vqa package at: {_SRC}", flush=True)

_DST = "/kaggle/working/culprit_vqa"
if not os.path.exists(_DST):
    shutil.copytree(_SRC, _DST)
sys.path.insert(0, "/kaggle/working")

print("=== Installing GPU-stack dependencies ===", flush=True)
# Versions PINNED. Between two otherwise-identical audit runs the number
# of naturally-wrong items moved 81 -> 67 with no code change, and the
# unpinned `-U` install is the most likely cause. Unpinned dependencies
# mean no two runs are comparable, which is fatal for an experiment whose
# whole output is a difference between runs.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.6.0", "bitsandbytes==0.45.5",
    "qwen-vl-utils==0.0.11", "datasets==3.5.0",
])

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from culprit_vqa.layer0_taxonomy.operators import get_operator  # noqa: E402
from culprit_vqa.layer1_intervention.items import Item  # noqa: E402
from culprit_vqa.layer1_intervention.lattice import Condition, make_condition_id  # noqa: E402
from culprit_vqa.layer2_runner.hf_runner import softmax  # noqa: E402
from culprit_vqa.layer2_runner.hf_runner import HFVisionLanguageRunner  # noqa: E402

RESULTS_PATH = "/kaggle/working/manipulation_check_results.jsonl"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_ITEMS = 700
MAX_RUNTIME_S = 5.0 * 3600  # hard stop well inside Kaggle's ~11-12h session kill
FACTOR_IDS = [
    "irrelevant_plausible_fact",
    "wrong_local_entity",
    "text_overlay_wrong_answer",
    "salience_recomposition",
]


# ----------------------------------------------------------------------
# CVQA loading (same defensive helpers as the other kernels -- the real
# schema has no Country/Language columns and options are list fields)
# ----------------------------------------------------------------------

def _get(record, *keys, default=None):
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return default


def _parse_options(raw):
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(o) for o in raw]
    if hasattr(raw, "tolist"):
        return [str(o) for o in raw.tolist()]
    return []


def _parse_answer_idx(raw, n_options):
    if isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        raw = int(raw.strip())
    if isinstance(raw, str):
        raw = {"A": 0, "B": 1, "C": 2, "D": 3}.get(raw.strip().upper(), 0)
    if isinstance(raw, int) and 0 <= raw < n_options:
        return raw
    return 0


def collect_items(n_items):
    ds = load_dataset("afaji/cvqa", split="test", streaming=True)
    pool = []
    for ex in ds:
        options = _parse_options(_get(ex, "Translated Options", "Options"))
        image = _get(ex, "image")
        question = _get(ex, "Translated Question", "Question")
        if not options or image is None or question is None:
            continue
        pool.append(ex)
        if len(pool) >= max(n_items * 2, 300):
            break

    by_subset = {}
    for ex in pool:
        by_subset.setdefault(str(_get(ex, "Subset", default="unknown")), []).append(ex)

    ordered, keys = [], sorted(by_subset)
    idx = {k: 0 for k in keys}
    while len(ordered) < n_items:
        progressed = False
        for k in keys:
            if idx[k] < len(by_subset[k]):
                ordered.append(by_subset[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(ordered) >= n_items:
                    break
        if not progressed:
            break
    return ordered[:n_items]


def build_item(ex, index):
    options = _parse_options(_get(ex, "Translated Options", "Options"))
    correct_idx = _parse_answer_idx(_get(ex, "Label", default=0), len(options))
    subset = str(_get(ex, "Subset", default="unknown"))
    return Item(
        item_id=f"ic_{index:04d}",
        image_ref=f"cvqa::{subset}::{index}",
        question=_get(ex, "Translated Question", "Question"),
        answer=options[correct_idx],
        evidence_region=None,
        language=subset,
        metadata={
            "pil_image": _get(ex, "image"), "options": options, "correct_idx": correct_idx,
            "subset": subset, "category": _get(ex, "Category", default="unknown"),
        },
    )


def main():
    print("=== Collecting CVQA items ===", flush=True)
    t0 = time.time()
    examples = collect_items(N_ITEMS)
    items = [build_item(ex, i) for i, ex in enumerate(examples)]
    del examples
    print(f"Collected {len(items)} items in {time.time()-t0:.1f}s", flush=True)
    if not items:
        raise RuntimeError("No CVQA items collected -- refusing to report an empty result")

    print(f"=== Loading {MODEL_ID} (letter scoring) ===", flush=True)
    t0 = time.time()
    runner = HFVisionLanguageRunner(model_id=MODEL_ID, seed=0, scoring="letter", collect_traces=False)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    factors = [get_operator(fid) for fid in FACTOR_IDS]
    out_f = open(RESULTS_PATH, "a", encoding="utf-8")
    records = []
    t_start = time.time()
    n_failed = 0

    for n, item in enumerate(items):
        elapsed = time.time() - t_start
        if elapsed > MAX_RUNTIME_S:
            print(f"\n=== TIME BUDGET REACHED at item {n}/{len(items)} "
                  f"({elapsed/3600:.2f}h) -- stopping cleanly ===", flush=True)
            break
        try:
            rec = {"item_id": item.item_id, "subset": item.language,
                   "category": item.metadata["category"],
                   "n_options": len(item.metadata["options"])}

            control = Condition(item.item_id, frozenset(), make_condition_id(item.item_id, frozenset()))
            c_res = runner.run(item, control)
            rec["control_p"] = c_res.gold_prob_mass
            rec["control_correct"] = bool(c_res.correct_flags[0])
            # Flags a condition whose raw scores contained NaN/inf. Observed
            # at ~1% on real 4-bit inference; softmax repairs it, but the
            # value is not trustworthy and must be excluded from the counts
            # rather than silently averaged in.
            rec["control_degenerate"] = bool(c_res.extra_signals.get("degenerate_scores", False))

            rec["factors"] = {}
            for f in factors:
                cond = Condition(item.item_id, frozenset([f]), make_condition_id(item.item_id, frozenset([f])))
                fr = runner.run(item, cond)
                rec["factors"][f.id] = {
                    "p": fr.gold_prob_mass,
                    "correct": bool(fr.correct_flags[0]),
                    "delta_p": fr.gold_prob_mass - rec["control_p"],
                    "broke_it": rec["control_correct"] and not bool(fr.correct_flags[0]),
                    "degenerate": bool(fr.extra_signals.get("degenerate_scores", False)),
                }

            records.append(rec)
            out_f.write(json.dumps(rec, default=str) + "\n")
            out_f.flush()
            rate = (time.time() - t_start) / (n + 1)
            broke = [fid for fid, o in rec["factors"].items() if o["broke_it"]]
            fits = int((MAX_RUNTIME_S - elapsed) / max(rate, 1e-6))
            print(f"[{n+1}/{len(items)}] {item.item_id} "
                  f"ctrl={'OK' if rec['control_correct'] else 'x'} p={rec['control_p']:.3f} "
                  f"broke_by={broke or '-'} | {rate:.1f}s/item, "
                  f"~{min(fits, len(items)-n-1)} more fit in budget", flush=True)
        except Exception as exc:  # noqa: BLE001 -- one bad item must not sink the run
            n_failed += 1
            out_f.write(json.dumps({"item_id": item.item_id, "error": str(exc)[:400]}) + "\n")
            out_f.flush()
            print(f"[{n+1}/{len(items)}] {item.item_id} FAILED: {str(exc)[:200]}", flush=True)

    out_f.close()
    print(f"\n=== DONE in {(time.time()-t_start)/3600:.2f}h: {len(records)} ok, "
          f"{n_failed} failed ===", flush=True)
    summarize(records)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def auc(pos, neg):
    """P(a randomly chosen correct-answer score > a randomly chosen
    wrong-answer score), ties counted as half. 0.5 = the measure carries
    no information about correctness. Computed exactly by rank rather
    than sampled."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # average ranks for ties
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks.setdefault(k, avg_rank)
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mcnemar_p(n10, n01):
    from math import comb
    n = n10 + n01
    if n == 0:
        return 1.0
    k = max(n10, n01)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def summarize(records):
    ok = [r for r in records if "error" not in r]
    if not ok:
        print("No usable records.", flush=True)
        return

    print("\n" + "=" * 74, flush=True)
    print("QUESTION 2: DO OUR INTERVENTIONS ACTUALLY CAUSE FAILURES?", flush=True)
    print("=" * 74, flush=True)
    # Exclude items whose CONTROL scoring was degenerate: every factor
    # delta for such an item is measured against an untrustworthy baseline.
    ok = [r for r in ok if not r.get("control_degenerate", False)]
    if not ok:
        print("All records had degenerate control scoring -- nothing usable.", flush=True)
        return
    ctrl_acc = sum(1 for r in ok if r["control_correct"]) / len(ok)
    ctrl_p = sum(r["control_p"] for r in ok) / len(ok)
    print(f"control: accuracy {ctrl_acc:.1%}, mean p(gold) {ctrl_p:.3f}\n", flush=True)
    print(f"{'factor':<28}{'acc':>8}{'acc drop':>10}{'mean dp':>10}{'broke':>7}{'fixed':>7}{'p':>8}", flush=True)
    for fid in FACTOR_IDS:
        rows = [r for r in ok if fid in r.get("factors", {})
                and not r["factors"][fid].get("degenerate", False)]
        if not rows:
            continue
        acc = sum(1 for r in rows if r["factors"][fid]["correct"]) / len(rows)
        mean_dp = sum(r["factors"][fid]["delta_p"] for r in rows) / len(rows)
        # Paired: among items where control and factor disagree, which way?
        broke = sum(1 for r in rows if r["control_correct"] and not r["factors"][fid]["correct"])
        fixed = sum(1 for r in rows if not r["control_correct"] and r["factors"][fid]["correct"])
        p = mcnemar_p(broke, fixed)
        flag = "" if p < 0.05 else "   <- not significant"
        print(f"{fid:<28}{acc:>7.1%}{ctrl_acc-acc:>+10.1%}{mean_dp:>+10.3f}"
              f"{broke:>7}{fixed:>7}{p:>8.3f}{flag}", flush=True)
    print("\n  'broke' = control correct -> factor wrong (the intended effect).", flush=True)
    print("  'fixed' = the reverse, i.e. noise. A factor whose broke/fixed", flush=True)
    print("  split is not significant is not a usable cause: Shapley over a", flush=True)
    print("  lattice of such factors is decomposing noise, and the efficiency", flush=True)
    print("  residual cannot detect that (it is an algebraic identity).", flush=True)


if __name__ == "__main__":
    main()
