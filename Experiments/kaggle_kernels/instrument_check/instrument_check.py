"""CULPRIT-VQA instrument validation + manipulation check.

This is the run that decides whether the project has a foundation. It
asks two questions that were never asked, in order, because the second
is meaningless if the first fails.

QUESTION 1 -- does our measure measure anything?
  Every Shapley value in this project is computed from `gold_prob_mass`.
  The original implementation scored the ABSOLUTE teacher-forced
  probability of the gold option's literal text. Measured after the fact
  on two completed runs, its AUC for separating correct from incorrect
  answers was 0.46 (Qwen-3B) and 0.53 (LLaVA-7B) -- a coin flip. That
  single fact explains the amortized attributor's rho ~= 0.09, the
  near-zero repair deltas, and the cross-model non-replication: all of it
  was computed on noise.

  Here we score the same clean items three ways and compare AUCs head to
  head on identical data:
    old_absolute : the original measure (the suspected-broken baseline)
    new_letter   : softmax over the k option-letter logits, 1 forward
    new_text     : softmax over length-normalized option-text log-probs
  PASS = a new measure reaches AUC >= 0.80. If neither does, stop: the
  problem is deeper than normalization and no downstream experiment is
  worth running.

QUESTION 2 -- do our interventions actually cause failures?
  The 2^k factorial lattice assumes the injected factors change model
  behaviour. Nobody ever checked. If a factor does not move accuracy,
  Shapley over that lattice is decomposing noise -- and note that the
  efficiency residual we previously treated as validation CANNOT catch
  this: it is an algebraic identity that holds exactly for random
  numbers.

  Here each factor is applied ALONE against the same item's control and
  we report the accuracy drop and the p(gold) drop it causes, with a
  paired test. A factor that does not move the model is not a usable
  cause and should be cut or strengthened.

Writes /kaggle/working/instrument_check_results.jsonl (one record per
item) plus a printed summary.
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

RESULTS_PATH = "/kaggle/working/instrument_check_results.jsonl"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_ITEMS = 200
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


# ----------------------------------------------------------------------
# The OLD measure, reimplemented verbatim so the comparison is honest.
# ----------------------------------------------------------------------

def old_absolute_measure(runner, image, question, options, correct_idx):
    """Exactly what `gold_prob_mass` used to be: the geometric-mean
    per-token probability of the gold option's literal text, in absolute
    terms, with correctness read off a greedy generation via substring /
    leading-letter matching.

    Reimplemented here rather than imported because the runner no longer
    contains it -- keeping it inline is what makes the head-to-head AUC
    comparison on identical inputs possible.
    """
    gold_answer = options[correct_idx]
    letters = "ABCDEFGH"[: len(options)]
    options_str = " ".join(f"{l}) {o}" for l, o in zip(letters, options))
    prompt_text = f"{question}\nOptions: {options_str}\nAnswer with the option text only."
    messages = [{"role": "user",
                 "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    chat_prefix = runner.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prefix_inputs = runner.processor(text=[chat_prefix], images=[image], return_tensors="pt").to(runner.model.device)
    n_prefix = prefix_inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen = runner.model.generate(**prefix_inputs, max_new_tokens=48, do_sample=False)
    trace = runner.processor.batch_decode(gen[:, n_prefix:], skip_special_tokens=True)[0]

    full_inputs = runner.processor(
        text=[chat_prefix + gold_answer], images=[image], return_tensors="pt"
    ).to(runner.model.device)
    with torch.no_grad():
        out = runner.model(**full_inputs)
    logits = out.logits[0]
    gold_ids = full_inputs["input_ids"][0].tolist()[n_prefix:]
    n_gold = len(gold_ids)
    if n_gold > 0 and logits.shape[0] >= n_prefix + n_gold:
        rel = logits[n_prefix - 1: n_prefix - 1 + n_gold]
        lp = torch.log_softmax(rel.float(), dim=-1)
        avg = sum(lp[j, gold_ids[j]].item() for j in range(n_gold)) / n_gold
        p = math.exp(avg) if avg > -50 else 0.0
    else:
        p = 0.0

    # old correctness heuristic
    t = trace.strip()
    gl = letters[correct_idx]
    if gold_answer.strip().lower() in t.lower():
        correct = True
    elif not t or t[0].upper() != gl.upper():
        correct = False
    else:
        rest = t[1:]
        correct = not rest or not rest[0].isalnum()

    del out, logits, full_inputs, prefix_inputs, gen
    torch.cuda.empty_cache()
    return p, correct, trace


def score_clean(runner, item, mode):
    """Score the item's CLEAN image/question in the given mode."""
    runner.scoring = mode
    r = runner.answer_with_question_override(item, item.question, run_label=f"{item.item_id}::clean::{mode}")
    return r.gold_prob_mass, bool(r.correct_flags[0])


def main():
    print("=== Collecting CVQA items ===", flush=True)
    t0 = time.time()
    examples = collect_items(N_ITEMS)
    items = [build_item(ex, i) for i, ex in enumerate(examples)]
    print(f"Collected {len(items)} items in {time.time()-t0:.1f}s", flush=True)

    print(f"=== Loading {MODEL_ID} ===", flush=True)
    t0 = time.time()
    runner = HFVisionLanguageRunner(model_id=MODEL_ID, seed=0, scoring="letter", collect_traces=False)
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    factors = [get_operator(fid) for fid in FACTOR_IDS]
    out_f = open(RESULTS_PATH, "a", encoding="utf-8")
    records = []
    t_start = time.time()

    for n, item in enumerate(items):
        try:
            rec = {"item_id": item.item_id, "subset": item.language,
                   "category": item.metadata["category"],
                   "n_options": len(item.metadata["options"])}

            # --- Q1: three measures on the same clean input ---
            rec["new_letter_p"], rec["new_letter_correct"] = score_clean(runner, item, "letter")
            rec["new_text_p"], rec["new_text_correct"] = score_clean(runner, item, "text")
            old_p, old_correct, old_trace = old_absolute_measure(
                runner, item.metadata["pil_image"], item.question,
                item.metadata["options"], item.metadata["correct_idx"],
            )
            rec["old_abs_p"], rec["old_abs_correct"] = old_p, old_correct
            rec["old_trace"] = old_trace[:80]

            # --- Q2: each factor alone, against this item's own control ---
            runner.scoring = "letter"
            control = Condition(item.item_id, frozenset(), make_condition_id(item.item_id, frozenset()))
            c_res = runner.run(item, control)
            rec["control_p"] = c_res.gold_prob_mass
            rec["control_correct"] = bool(c_res.correct_flags[0])

            rec["factors"] = {}
            for f in factors:
                cond = Condition(item.item_id, frozenset([f]), make_condition_id(item.item_id, frozenset([f])))
                fr = runner.run(item, cond)
                rec["factors"][f.id] = {
                    "p": fr.gold_prob_mass,
                    "correct": bool(fr.correct_flags[0]),
                    "delta_p": fr.gold_prob_mass - rec["control_p"],
                    "broke_it": rec["control_correct"] and not bool(fr.correct_flags[0]),
                }

            records.append(rec)
            out_f.write(json.dumps(rec, default=str) + "\n")
            out_f.flush()
            rate = (time.time() - t_start) / (n + 1)
            print(f"[{n+1}/{len(items)}] {item.item_id} "
                  f"letter_p={rec['new_letter_p']:.3f}({'OK' if rec['new_letter_correct'] else 'x'}) "
                  f"text_p={rec['new_text_p']:.3f}({'OK' if rec['new_text_correct'] else 'x'}) "
                  f"old_p={old_p:.4f}({'OK' if old_correct else 'x'}) "
                  f"| {rate:.1f}s/item, eta {rate*(len(items)-n-1)/60:.0f}m", flush=True)
        except Exception as exc:  # noqa: BLE001 -- one bad item must not sink the run
            out_f.write(json.dumps({"item_id": item.item_id, "error": str(exc)[:400]}) + "\n")
            out_f.flush()
            print(f"[{n+1}/{len(items)}] {item.item_id} FAILED: {str(exc)[:200]}", flush=True)

    out_f.close()
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
    print(f"QUESTION 1: DOES THE MEASURE MEASURE ANYTHING?   (n={len(ok)})", flush=True)
    print("=" * 74, flush=True)
    print(f"{'measure':<16}{'accuracy':>10}{'mean p(gold)':>14}{'AUC':>8}   verdict", flush=True)
    verdicts = {}
    for name, pkey, ckey in [("old_absolute", "old_abs_p", "old_abs_correct"),
                             ("new_letter", "new_letter_p", "new_letter_correct"),
                             ("new_text", "new_text_p", "new_text_correct")]:
        pos = [r[pkey] for r in ok if r[ckey]]
        neg = [r[pkey] for r in ok if not r[ckey]]
        a = auc(pos, neg)
        acc = len(pos) / len(ok)
        mean_p = sum(r[pkey] for r in ok) / len(ok)
        verdict = "PASS" if a >= 0.80 else ("weak" if a >= 0.65 else "NOISE")
        verdicts[name] = a
        print(f"{name:<16}{acc:>9.1%}{mean_p:>14.4f}{a:>8.3f}   {verdict}", flush=True)
    print("\n  AUC 0.5 = the measure cannot tell right from wrong at all.", flush=True)
    print("  PASS threshold is 0.80. If no new measure passes, STOP -- the", flush=True)
    print("  problem is deeper than normalization.", flush=True)

    print("\n" + "=" * 74, flush=True)
    print("QUESTION 2: DO OUR INTERVENTIONS ACTUALLY CAUSE FAILURES?", flush=True)
    print("=" * 74, flush=True)
    ctrl_acc = sum(1 for r in ok if r["control_correct"]) / len(ok)
    ctrl_p = sum(r["control_p"] for r in ok) / len(ok)
    print(f"control: accuracy {ctrl_acc:.1%}, mean p(gold) {ctrl_p:.3f}\n", flush=True)
    print(f"{'factor':<28}{'acc':>8}{'acc drop':>10}{'mean dp':>10}{'broke':>7}{'fixed':>7}{'p':>8}", flush=True)
    for fid in FACTOR_IDS:
        rows = [r for r in ok if fid in r.get("factors", {})]
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
