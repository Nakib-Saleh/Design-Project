"""CULPRIT-VQA attribution run v3 -- k=3 lattice with live behavioral signals.

What v2 established, and what it could not
------------------------------------------
v2 (1000 items, 0 errors, letter scoring) produced the project's first
externally-validated attribution: mean blame ordering matched an
independently measured solo-strength ordering. It also hit two ceilings
that this run is built to clear.

1. ONLY TWO FACTORS SURVIVED THE MANIPULATION CHECK, so the lattice was
   k=2. At k=2, phi_1 - phi_2 == LOO_1 - LOO_2 exactly (verified at
   1.1e-16 over 652 items), so Shapley and leave-one-out CANNOT rank the
   factors differently. RQ1's central claim -- that Shapley identifies a
   better primary culprit -- was untestable, not unsupported. Three
   factors make it testable.

2. THREE OF SIX BEHAVIORAL SIGNALS WERE CONSTANT ZERO on every one of the
   1000 rows, so the amortized attributor (Layer 4) was effectively a
   one-feature model and reached rho=0.49 against a 0.70 target. Two
   independent causes, both fixed:
     - Item.perturbation_effects was never populated, so S2/S4 had no
       record of what each factor injected. This run calls
       runner.describe_effects() per item.
     - Under letter scoring the model emits a SINGLE token, so every
       trace-text signal was structurally dead. S2/S3/S4 are now computed
       from the per-option probability distribution the scoring pass
       already produces, at no extra cost.

Cost note
---------
k=3 means 8 lattice conditions per item instead of 4. That is paid for by
dropping trace generation: v2 generated up to 24 tokens per condition to
feed S1 drift, but under letter scoring the model returned just "A", so
the generation bought a single character. With collect_traces=False the
trace is a rendered "A) Torii", which gives S1 strictly more to compare
while costing nothing, and S5 never needed generation at all.

CONFIG BELOW IS REWRITTEN BY THE GATE LAUNCHER -- see the GATE: markers.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# --- GATE-CONTROLLED CONFIG (rewritten by scripts/gated_launch.py) ---
SCORING_MODE = "letter"  # GATE:SCORING
ACTIVE_FACTOR_IDS = ["text_overlay_wrong_answer", "salience_recomposition", "cultural_text_overlay"]  # GATE:FACTORS
ITEM_OFFSET = 0  # GATE:OFFSET
RESULTS_PATH = "/kaggle/working/attribution_v3_results.jsonl"  # GATE:RESULTS
# ---------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SEED = 0
N_ITEMS = 400           # per account -> 800 items total across the two runs.
                        # Lower than v2's 500 because k=3 doubles the lattice
                        # (8 conditions vs 4). Sized by RAM as well as time:
                        # collect_real_items holds every pooled example and HF
                        # decodes the image on access, so the offset account pools
                        # ~2*N_ITEMS rows of decoded PIL images before the model
                        # loads. Offsets in the launcher MUST equal this value so
                        # the two accounts cover disjoint ranges.
TOTAL_ITEMS = 800       # across ALL accounts = sum of every account's N_ITEMS.
                        # The stratified ordering is built over exactly this many
                        # items REGARDLESS of which slice this account wants, which
                        # is what makes the slices disjoint -- see collect_real_items.
MAX_RUNTIME_S = 6.0 * 3600   # item-loop budget. Total kernel time is this plus
                             # ~10 min of pip install + CVQA streaming + model load,
                             # so the session lands around 6.2h -- inside the target
                             # window and far inside Kaggle's ~11-12h kill.
COLLECT_TRACES = False       # v2 set this True so S1 drift would have real generated
                             # text. Inspecting the v2 output showed the model returned a
                             # SINGLE letter ("A") for every condition, so the generation
                             # pass bought one character and S1 was already just
                             # "did the chosen option change". With this False the trace
                             # is a rendered "A) Torii" -- strictly more for S1 to
                             # compare -- and a full generate() per condition is saved,
                             # which is what pays for the k=3 lattice. S5 reads the
                             # option distribution and never needed generation.
TRACE_MAX_TOKENS = 24        # unused while COLLECT_TRACES is False; kept so flipping
                             # the flag back does not silently generate unbounded text.

print("=== Bootstrapping culprit_vqa package from the attached dataset ===", flush=True)
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

print("=== Installing GPU-stack dependencies (PINNED) ===", flush=True)
# Pinned: two otherwise-identical audit runs disagreed 81 vs 67 on which
# items the model got wrong, with no code change. Unpinned `-U` installs
# are the likeliest cause, and an experiment whose entire output is a
# difference between runs cannot tolerate a moving dependency floor.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.51.3", "accelerate==1.6.0", "bitsandbytes==0.45.5",
    "qwen-vl-utils==0.0.11", "datasets==3.5.0",
])

from datasets import load_dataset  # noqa: E402

from culprit_vqa.layer0_taxonomy.operators import get_operator  # noqa: E402
from culprit_vqa.layer1_intervention.items import Item  # noqa: E402
from culprit_vqa.layer2_runner.hf_runner import HFVisionLanguageRunner  # noqa: E402
from culprit_vqa.layer3a_causal.attribution import efficiency_residual  # noqa: E402
from culprit_vqa.layer3a_causal.probability import build_p_function  # noqa: E402
from culprit_vqa.pipeline import run_pipeline_for_item  # noqa: E402


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


def collect_real_items(n_items: int, offset: int):
    """Build the FULL stratified ordering of TOTAL_ITEMS, then return this
    account's slice of it.

    The ordering is deliberately computed over TOTAL_ITEMS regardless of
    which slice is wanted. An earlier version pooled `n_items + offset +
    buffer` rows, which made the pool size -- and therefore the subset
    buckets, and therefore the round-robin ordering -- depend on the
    offset. Because CVQA streams grouped by subset, the two accounts then
    saw different subset key sets and built DIFFERENT orderings, so their
    supposedly-disjoint index ranges overlapped in actual items:
    simulated on a realistic stream, 214 of 500 items were duplicated
    across the two accounts, giving 786 unique items where the pooled
    files would claim 1000.

    That is a silent validity failure -- no error, just double-counted
    items inflating apparent sample size -- so the ordering must be
    offset-independent. Both accounts now pool identically and slice the
    same list.
    """
    if offset + n_items > TOTAL_ITEMS:
        raise ValueError(
            f"slice [{offset}:{offset + n_items}] exceeds TOTAL_ITEMS={TOTAL_ITEMS}; "
            "raise TOTAL_ITEMS or the slices will not cover what the analysis assumes"
        )

    ds = load_dataset("afaji/cvqa", split="test", streaming=True)
    pool = []
    for ex in ds:
        options = _parse_options(_get(ex, "Translated Options", "Options"))
        if not options or _get(ex, "image") is None:
            continue
        if _get(ex, "Translated Question", "Question") is None:
            continue
        pool.append(ex)
        # Fixed target, independent of this account's offset. The +150 is
        # slack so the round-robin can fill TOTAL_ITEMS even when subsets
        # are unevenly sized.
        if len(pool) >= TOTAL_ITEMS + 150:
            break

    by_subset = {}
    for ex in pool:
        by_subset.setdefault(str(_get(ex, "Subset", default="unknown")), []).append(ex)

    ordered, keys = [], sorted(by_subset)
    idx = {k: 0 for k in keys}
    while len(ordered) < TOTAL_ITEMS:
        progressed = False
        for k in keys:
            if idx[k] < len(by_subset[k]):
                ordered.append(by_subset[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(ordered) >= TOTAL_ITEMS:
                    break
        if not progressed:
            break
    return ordered[offset:offset + n_items]


def build_item(ex, index):
    options = _parse_options(_get(ex, "Translated Options", "Options"))
    correct_idx = _parse_answer_idx(_get(ex, "Label", default=0), len(options))
    question = _get(ex, "Translated Question", "Question")
    raw_subset = _get(ex, "Subset", default="unknown")
    subset = str(raw_subset)
    native_question = _get(ex, "Question")
    if not native_question or native_question == question:
        native_question = None
    return Item(
        item_id=f"av2_{index:05d}",
        image_ref=f"cvqa::{subset}::{index}",
        question=question,
        answer=options[correct_idx],
        evidence_region=None,
        language=subset,
        perturbation_effects={},
        metadata={
            "pil_image": _get(ex, "image"), "options": options, "correct_idx": correct_idx,
            # raw_subset keeps the ('Language','Country') pair intact: the
            # culture operators need it to pick a culture that is NOT the
            # item's own. subset_str stays for logging and grouping.
            "subset": raw_subset, "subset_str": subset,
            "category": _get(ex, "Category", default="unknown"),
            "native_question": native_question,
        },
    )


def assign_factor_plan(n_items, factor_ids, seed=SEED):
    """Every item gets a k=3 lattice when three or more factors survived
    the gate; falls back to k=2 pairs when only two did.

    Falls back gracefully: with exactly 2 surviving factors there are no
    triples, so everything is k=2 and pairwise interactions are still
    measurable -- the run degrades to a v2 repeat rather than failing.
    """
    import itertools

    pairs = list(itertools.combinations(factor_ids, 2))
    triples = list(itertools.combinations(factor_ids, 3))
    if not pairs:
        raise ValueError(f"need >= 2 factors to measure interactions, got {factor_ids}")

    # v2 gave only 20% of items a k=3 lattice. That was right when the
    # third factor was speculative, but the whole purpose of this run is
    # the k=3 comparison -- at k=2 Shapley and leave-one-out rank factors
    # identically by algebra, so k=2 items contribute nothing to RQ1.
    # Every item therefore gets a full triple when three factors survived.
    if triples:
        return [list(triples[i % len(triples)]) for i in range(n_items)]
    return [list(pairs[i % len(pairs)]) for i in range(n_items)]


def compute_language_delta(item, runner):
    """S6: English p(gold) minus native-language p(gold) on the CLEAN
    question. Now on the normalized scale, so unlike the v1 version this
    is a difference between two comparable quantities rather than between
    two absolute string probabilities."""
    native = item.metadata.get("native_question")
    if not native or native == item.question:
        return 0.0
    en = runner.answer_with_question_override(item, item.question, run_label=f"{item.item_id}::lang_en")
    nat = runner.answer_with_question_override(item, native, run_label=f"{item.item_id}::lang_native")
    return en.gold_prob_mass - nat.gold_prob_mass


def serialize_record(record):
    p_fn = build_p_function(record.run_results)
    control = record.run_results[frozenset()]
    return {
        "item_id": record.item.item_id,
        "subset": record.item.language,
        "category": record.item.metadata.get("category"),
        "scoring_mode": SCORING_MODE,
        "factors": [f.id for f in record.factors],
        "phi": {f.id: v for f, v in record.phi.items()},
        "alpha": {f.id: v for f, v in record.alpha.items()},
        "loo": {f.id: v for f, v in record.loo.items()},
        "interactions": {"|".join(sorted(f.id for f in pair)): v for pair, v in record.interactions.items()},
        "efficiency_residual": efficiency_residual(record.phi, p_fn, record.factors),
        "p_values": {
            "|".join(sorted(f.id for f in subset)) or "control": p_fn(subset)
            for subset in record.run_results
        },
        # Per-condition correctness: with the fixed measure this is
        # argmax==gold, so it is directly comparable to p(gold) rather
        # than coming from a separate string-matching heuristic.
        "correct_by_condition": {
            "|".join(sorted(f.id for f in subset)) or "control": bool(rr.correct_flags[0])
            for subset, rr in record.run_results.items()
        },
        # Promoted to top level because the headline analysis MUST filter on
        # it: attribution answers "what broke this?", which presupposes the
        # model got it right without any factor applied. On control-wrong
        # items phi measures movement around an already-failing baseline and
        # should not be pooled with the rest.
        "control_correct": bool(control.correct_flags[0]),
        "control_p_gold": control.gold_prob_mass,
        "control_option_probs": control.extra_signals.get("option_probs"),
        # Per-condition option distributions. v2 stored only the control's,
        # which made it impossible to recompute any option-level signal
        # after the fact -- the reason the S2/S3/S4 fix could not be
        # validated on v2 data and needed a fresh run.
        "option_probs_by_condition": {
            "|".join(sorted(f.id for f in subset)) or "control":
                rr.extra_signals.get("option_probs")
            for subset, rr in record.run_results.items()
        },
        "predicted_idx_by_condition": {
            "|".join(sorted(f.id for f in subset)) or "control":
                rr.extra_signals.get("predicted_idx")
            for subset, rr in record.run_results.items()
        },
        "degenerate_by_condition": {
            "|".join(sorted(f.id for f in subset)) or "control":
                bool(rr.extra_signals.get("degenerate_scores", False))
            for subset, rr in record.run_results.items()
        },
        # What each factor injected, so the option-level signals can be
        # recomputed or audited offline.
        "distractor_option_idx": {
            fid: eff.distractor_option_idx
            for fid, eff in record.item.perturbation_effects.items()
        },
        "factor_applied": {
            fid: bool(eff.distractor_keywords)
            for fid, eff in record.item.perturbation_effects.items()
        },
        "control_degenerate": bool(control.extra_signals.get("degenerate_scores", False)),
        # Which scoring path actually ran. A silent fallback from letter to
        # text scoring would make two runs incomparable with no visible sign.
        "scoring_used": control.extra_signals.get("scoring_used"),
        "signal_vectors": {cid: sv.values for cid, sv in record.signal_vectors.items()},
        "language_delta": record.item.metadata.get("language_delta", 0.0),
        "sample_trace": control.cot_traces[0][:150],
    }


def main():
    print(f"=== CONFIG: scoring={SCORING_MODE} factors={ACTIVE_FACTOR_IDS} "
          f"offset={ITEM_OFFSET} budget={MAX_RUNTIME_S/3600:.1f}h ===", flush=True)

    t0 = time.time()
    examples = collect_real_items(N_ITEMS, ITEM_OFFSET)
    print(f"Collected {len(examples)} items (offset {ITEM_OFFSET}) in {time.time()-t0:.1f}s", flush=True)
    if not examples:
        raise RuntimeError(
            f"No items returned for offset {ITEM_OFFSET}. The stream did not reach the\n"
            f"requested slice -- do NOT let this run silently produce nothing."
        )
    if len(examples) < N_ITEMS:
        # Not fatal (the time budget would likely have cut it short anyway),
        # but it means the two accounts' slices are not the sizes assumed,
        # which matters when the two result files are pooled for analysis.
        print(f"WARNING: got {len(examples)}/{N_ITEMS} items at offset {ITEM_OFFSET}", flush=True)

    plan = assign_factor_plan(len(examples), ACTIVE_FACTOR_IDS)

    # Build every Item up front, then drop the raw pool: the pooled rows and
    # the Items hold the same images, and keeping both alive across the model
    # load doubles peak host RAM for no reason.
    items = [build_item(ex, i + ITEM_OFFSET) for i, ex in enumerate(examples)]
    del examples
    gc.collect()

    print(f"=== Loading {MODEL_ID} (scoring={SCORING_MODE}) ===", flush=True)
    t0 = time.time()
    runner = HFVisionLanguageRunner(
        model_id=MODEL_ID, seed=SEED, scoring=SCORING_MODE,
        collect_traces=COLLECT_TRACES, max_new_tokens=TRACE_MAX_TOKENS,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    t_start = time.time()
    n_ok = n_failed = 0
    stopped_early = False

    with open(RESULTS_PATH, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(items):
            elapsed = time.time() - t_start
            if elapsed > MAX_RUNTIME_S:
                stopped_early = True
                print(f"\n=== TIME BUDGET REACHED at item {i}/{len(items)} "
                      f"({elapsed/3600:.2f}h) -- stopping cleanly ===", flush=True)
                break

            factors = [get_operator(fid) for fid in plan[i]]
            t_item = time.time()
            try:
                # THE FIX for the three dead signals. Records what each
                # factor will actually inject into THIS item, using the
                # runner's own seed scheme so the recorded distractor is
                # the option the model is really shown. Without this,
                # Item.perturbation_effects stays empty and S2/S4 return a
                # constant 0.0 -- silently, with no error, exactly as in v2.
                item.perturbation_effects = runner.describe_effects(
                    item, [f.id for f in factors]
                )
                item.metadata["language_delta"] = compute_language_delta(item, runner)
                record = run_pipeline_for_item(item, factors, runner, n_decodes=1)
                out_f.write(json.dumps(serialize_record(record), default=str) + "\n")
                out_f.flush()
                n_ok += 1
                per_item = (time.time() - t_start) / (i + 1)
                remaining_by_budget = int((MAX_RUNTIME_S - elapsed) / max(per_item, 1e-6))
                print(f"[{i+1}/{len(items)}] OK {item.item_id} k={len(factors)} "
                      f"{time.time()-t_item:.1f}s | {per_item:.1f}s/item, "
                      f"~{min(remaining_by_budget, len(items)-i-1)} more fit in budget", flush=True)
            except Exception:
                n_failed += 1
                err = traceback.format_exc()
                out_f.write(json.dumps({"item_id": item.item_id, "error": err[-400:]}) + "\n")
                out_f.flush()
                print(f"[{i+1}/{len(items)}] FAILED {item.item_id}: {err[-250:]}", flush=True)

    total_h = (time.time() - t_start) / 3600
    print(f"\n=== DONE in {total_h:.2f}h: {n_ok} ok, {n_failed} failed, "
          f"stopped_early={stopped_early} ===", flush=True)


if __name__ == "__main__":
    main()
