"""CULPRIT-VQA culture check -- do any CULTURALLY-LOADED interventions bite?

Why this run exists
-------------------
The first manipulation check killed the thesis's cultural arm. Of four
factors, only two survived, and both are visual/cross-modal:

    text_overlay_wrong_answer   +33.4%  p=0.000   KEEP
    salience_recomposition       +5.3%  p=0.001   KEEP
    irrelevant_plausible_fact    +1.7%  p=0.155   inert
    wrong_local_entity           +1.1%  p=0.200   inert   <- the cultural one

So the attribution run had no culturally-loaded cause to attribute
anything to, and "multicultural failure attribution" was unsupported by
its own instrument. `wrong_local_entity` appends a parenthetical aside
("(Note: some people confuse this with Diwali.)") that the model can skip
without affecting the answer -- a structural defect, not a dosage one.

This run tests four cultural interventions that each remove that defect a
different way, so the result identifies WHICH property makes a cultural
cue bite rather than only whether one does:

    biased_prior_phrasing        ask the ORIGINAL native-language question
                                 (real CVQA text, no template at all)
    western_default_substitution assert a wrong culture as a presupposition
                                 INSIDE the question stem
    cultural_text_overlay        assert a wrong culture VISUALLY, using the
                                 same banner mechanism as the +33.4% factor
                                 but naming no answer option

A fourth candidate, entity_swap_in_question, was built and tested but
dropped before launch: a text-only pre-flight showed it fires on only 24%
of real CVQA questions. See EXCLUDED_FROM_THIS_RUN below.

The three that remain span the modality axis deliberately -- two textual,
one visual -- so a null result cannot be explained away as "the cue was
in the channel this model ignores".

`text_overlay_wrong_answer` rides along as a positive control: if it does
not reproduce roughly +33%, the harness is wrong and nothing else in the
run should be believed.

All three fired on 100% of 400 pre-flight rows, and none ever asserted
the item's own culture. Applied-ness is still recorded per item and the
statistics computed on the APPLIED SUBSET: averaging an intervention over
items it never touched dilutes a real effect toward zero, which is its
own way of producing a false negative.
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

RESULTS_PATH = "/kaggle/working/culture_check_results.jsonl"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_ITEMS = 580           # 6 model calls per item (control + 3 culture + 2 known).
                        # 580*6 = 3480 calls, matching the 700*5 = 3500 of the
                        # manipulation check that finished inside this budget.
MAX_RUNTIME_S = 5.0 * 3600  # hard stop well inside Kaggle's ~11-12h session kill
CULTURE_FACTOR_IDS = [
    "biased_prior_phrasing",         # native-language swap   (textual, real data)
    "western_default_substitution",  # wrong-culture premise  (textual, template)
    "cultural_text_overlay",         # wrong-culture caption  (visual)
]
# `entity_swap_in_question` is implemented, registered and tested, but is
# NOT run here. A text-only pre-flight over 400 real CVQA rows found it
# applies to just 24% of questions (most have no named entity), and some
# of those swaps are nonsense rather than culturally misleading -- it
# replaced a person's name in "What is Mona Jimenez doing with his hands?"
# with a country. At n~170 applied, an inert verdict would be
# indistinguishable from no power, which is exactly the ambiguity this
# run exists to avoid. Dropping it also buys back the items needed to
# keep 700 at the same wall clock.
EXCLUDED_FROM_THIS_RUN = ["entity_swap_in_question"]
# The two factors that already survived at n=698 (+33.4% and +5.3%). Both
# are re-measured here rather than carried over on trust, for two reasons:
# the k=3 lattice needs THREE factors and the gate can only promote what
# this run actually measured, and text_overlay doubles as a positive
# control -- if it does not reproduce ~33%, the harness has changed and a
# null result for the culture factors would be uninterpretable.
POSITIVE_CONTROL_ID = "text_overlay_wrong_answer"
KNOWN_FACTOR_IDS = [POSITIVE_CONTROL_ID, "salience_recomposition"]
FACTOR_IDS = CULTURE_FACTOR_IDS + KNOWN_FACTOR_IDS


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
    raw_subset = _get(ex, "Subset", default="unknown")
    subset = str(raw_subset)
    question = _get(ex, "Translated Question", "Question")
    # The ORIGINAL native-language question. biased_prior_phrasing swaps it
    # in, so without this the operator silently no-ops on every item and
    # would be recorded as inert for a plumbing reason rather than a
    # scientific one.
    native_question = _get(ex, "Question")
    if not native_question or native_question == question:
        native_question = None
    return Item(
        item_id=f"cc_{index:04d}",
        image_ref=f"cvqa::{subset}::{index}",
        question=question,
        answer=options[correct_idx],
        evidence_region=None,
        language=subset,
        metadata={
            "pil_image": _get(ex, "image"), "options": options, "correct_idx": correct_idx,
            # raw_subset keeps the ('Language','Country') pair intact so the
            # culture operators can pick a culture that is NOT the item's own.
            "subset": raw_subset, "subset_str": subset,
            "category": _get(ex, "Category", default="unknown"),
            "native_question": native_question,
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
                   "n_options": len(item.metadata["options"]),
                   "has_native_question": bool(item.metadata.get("native_question"))}

            # What each factor will actually inject for THIS item, using the
            # runner's own seed scheme. Two purposes: it records whether the
            # operator applied at all (no-ops return no keywords), and it
            # populates item.perturbation_effects, which is what makes the
            # option-level signals live in the attribution run.
            effects = runner.describe_effects(item, FACTOR_IDS)
            item.perturbation_effects = effects

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
                    # False when the operator found nothing to change on this
                    # item (no native question, no named entity). Such rows
                    # are identical to control and must be excluded, not
                    # averaged in as evidence of no effect.
                    "applied": bool(effects[f.id].distractor_keywords),
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

    print("\n" + "=" * 78, flush=True)
    print("DO CULTURALLY-LOADED INTERVENTIONS ACTUALLY CAUSE FAILURES?", flush=True)
    print("=" * 78, flush=True)
    ok = [r for r in ok if not r.get("control_degenerate", False)]
    if not ok:
        print("All records had degenerate control scoring -- nothing usable.", flush=True)
        return

    n_native = sum(1 for r in ok if r.get("has_native_question"))
    print(f"items: {len(ok)}   with a distinct native-language question: "
          f"{n_native} ({n_native/len(ok):.0%})", flush=True)

    ctrl_acc = sum(1 for r in ok if r["control_correct"]) / len(ok)
    ctrl_p = sum(r["control_p"] for r in ok) / len(ok)
    print(f"control: accuracy {ctrl_acc:.1%}, mean p(gold) {ctrl_p:.3f}", flush=True)
    print("\nStatistics are computed on each factor's APPLIED SUBSET, and the", flush=True)
    print("control accuracy is recomputed on that same subset so the drop is", flush=True)
    print("a like-for-like comparison.\n", flush=True)

    header = (f"{'factor':<30}{'applied':>9}{'acc':>8}{'drop':>9}"
              f"{'mean dp':>10}{'broke':>7}{'fixed':>7}{'p':>8}  verdict")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    survivors = []
    for fid in FACTOR_IDS:
        rows = [r for r in ok
                if fid in r.get("factors", {})
                and r["factors"][fid].get("applied", True)
                and not r["factors"][fid].get("degenerate", False)]
        if not rows:
            print(f"{fid:<30}{0:>9}{'--':>8}   never applied on any item", flush=True)
            continue
        acc = sum(1 for r in rows if r["factors"][fid]["correct"]) / len(rows)
        sub_ctrl = sum(1 for r in rows if r["control_correct"]) / len(rows)
        mean_dp = sum(r["factors"][fid]["delta_p"] for r in rows) / len(rows)
        broke = sum(1 for r in rows if r["control_correct"] and not r["factors"][fid]["correct"])
        fixed = sum(1 for r in rows if not r["control_correct"] and r["factors"][fid]["correct"])
        p = mcnemar_p(broke, fixed)
        keep = p < 0.05 and broke > fixed
        if keep and fid != POSITIVE_CONTROL_ID:
            survivors.append(fid)
        tag = "KEEP" if keep else "inert"
        if fid == POSITIVE_CONTROL_ID:
            tag = "positive control" if keep else "CONTROL FAILED - DISTRUST RUN"
        print(f"{fid:<30}{len(rows):>9}{acc:>7.1%}{sub_ctrl-acc:>+9.1%}"
              f"{mean_dp:>+10.3f}{broke:>7}{fixed:>7}{p:>8.3f}  {tag}", flush=True)

    print("\n" + "-" * 78, flush=True)
    pc = [r for r in ok if POSITIVE_CONTROL_ID in r.get("factors", {})]
    if pc:
        pc_broke = sum(1 for r in pc if r["control_correct"]
                       and not r["factors"][POSITIVE_CONTROL_ID]["correct"])
        pc_acc = sum(1 for r in pc if r["factors"][POSITIVE_CONTROL_ID]["correct"]) / len(pc)
        print(f"Positive control reproduced: accuracy {pc_acc:.1%} "
              f"(expected ~33%), broke {pc_broke}. "
              f"{'OK' if pc_acc < 0.50 else 'MISMATCH -- investigate before using this run'}",
              flush=True)

    print(f"\nCultural factors that bite: "
          f"{', '.join(survivors) if survivors else 'NONE'}", flush=True)
    if survivors:
        print("\n-> A third live factor makes a k=3 lattice possible, which is what", flush=True)
        print("   turns the Shapley-vs-LOO ordering question into a testable one:", flush=True)
        print("   at k=2 the two methods rank factors identically by algebra.", flush=True)
    else:
        print("\n-> Report as a negative result. Four structurally different", flush=True)
        print("   cultural interventions leaving the model unmoved is a finding", flush=True)
        print("   about the model, and a much stronger one than a single", flush=True)
        print("   badly-built operator failing.", flush=True)


if __name__ == "__main__":
    main()
