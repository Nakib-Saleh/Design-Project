"""CULPRIT-VQA natural-failure audit (proposal §7, the RQ4 headline experiment).

Unlike the main pipeline (which measures causes of failures WE inject),
this tests: can we diagnose a mistake the model makes ON ITS OWN, with
the image and question completely untouched?

For each of a pool of real CVQA items:
  1. Ask the clean, unmodified question. If the model gets it right,
     skip (nothing to diagnose).
  2. If wrong, try FIVE repair hypotheses -- one per taxonomy axis -- and
     record for each whether it flipped the answer to correct AND how far
     it moved the model's probability on the gold answer:
       - "add_category_hint"        (knowledge): lacked topical context
       - "supply_cultural_context"  (culturality): lacked the cultural setting
       - "ask_in_native_language"   (text modality): the translation lost something
       - "chain_of_thought"         (reasoning): took a snap answer
       - "enhance_visual_detail"    (visual modality): evidence too small to see
  3. Also run TWO placebo repairs that add comparable text carrying no
     usable information. Greedy decoding is sensitive to any prompt
     change, so some "fixes" are noise; the placebo flip rate estimates
     that floor and a real repair only counts if it beats it.

Writes /kaggle/working/natural_failure_results.jsonl -- one JSON record
per NATURALLY-WRONG item (correct items are not written, just counted).
"""

import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("=== Bootstrapping culprit_vqa package from the attached dataset ===", flush=True)
print(f"/kaggle/input contents: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'MISSING'}", flush=True)

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
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.51.0", "accelerate>=0.34.0", "bitsandbytes>=0.43.0",
    "qwen-vl-utils", "datasets",
])

from datasets import load_dataset  # noqa: E402

from dataclasses import asdict  # noqa: E402

from culprit_vqa.layer1_intervention.items import Item  # noqa: E402
from culprit_vqa.layer2_runner.hf_runner import HFVisionLanguageRunner  # noqa: E402
from culprit_vqa.natural_failure import run_natural_failure_audit  # noqa: E402

RESULTS_PATH = "/kaggle/working/natural_failure_results.jsonl"
N_ITEMS = 220  # pool of clean items to test; only naturally-wrong ones get a full record


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


def collect_real_items(n_items: int):
    ds = load_dataset("afaji/cvqa", split="test", streaming=True)
    all_examples = []
    for ex in ds:
        options = _parse_options(_get(ex, "Translated Options", "Options"))
        image = _get(ex, "image")
        question = _get(ex, "Translated Question", "Question")
        if not options or image is None or question is None:
            continue
        all_examples.append(ex)
        if len(all_examples) >= max(n_items * 2, 300):
            break

    by_subset = {}
    for ex in all_examples:
        subset = str(_get(ex, "Subset", default="unknown"))
        by_subset.setdefault(subset, []).append(ex)

    ordered = []
    subset_keys = sorted(by_subset.keys())
    idx_per_subset = {k: 0 for k in subset_keys}
    while len(ordered) < n_items and subset_keys:
        progressed = False
        for k in subset_keys:
            i = idx_per_subset[k]
            if i < len(by_subset[k]):
                ordered.append(by_subset[k][i])
                idx_per_subset[k] += 1
                progressed = True
                if len(ordered) >= n_items:
                    break
        if not progressed:
            break
    return ordered[:n_items]


def build_item(ex, index: int) -> Item:
    options = _parse_options(_get(ex, "Translated Options", "Options"))
    correct_idx = _parse_answer_idx(_get(ex, "Label", default=0), len(options))
    question = _get(ex, "Translated Question", "Question")
    image = _get(ex, "image")
    subset = str(_get(ex, "Subset", default="unknown"))
    category = _get(ex, "Category", default="unknown")

    # The ORIGINAL, untranslated question -- what `ask_in_native_language`
    # substitutes in. Stored as None when it is missing or identical to
    # the English text (true for the English-native subsets), so the
    # audit marks that repair unapplied rather than counting it as a
    # tested-and-failed hypothesis on items where it changes nothing.
    native_question = _get(ex, "Question")
    if not native_question or native_question == question:
        native_question = None

    return Item(
        item_id=f"nf_{index:04d}",
        image_ref=f"cvqa::{subset}::{index}",
        question=question,
        answer=options[correct_idx],
        evidence_region=None,
        language=subset,
        metadata={
            "pil_image": image, "options": options, "correct_idx": correct_idx,
            "subset": subset, "category": category, "native_question": native_question,
        },
    )


def main():
    print("=== Collecting real CVQA items for the natural-failure pool ===", flush=True)
    t0 = time.time()
    examples = collect_real_items(N_ITEMS)
    print(f"Collected {len(examples)} items in {time.time()-t0:.1f}s", flush=True)

    print("=== Loading quantized model ===", flush=True)
    t0 = time.time()
    runner = HFVisionLanguageRunner(model_id="Qwen/Qwen2.5-VL-3B-Instruct", seed=0)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    items = [build_item(ex, i) for i, ex in enumerate(examples)]
    item_by_id = {item.item_id: item for item in items}

    counts = {"correct": 0, "wrong": 0}
    out_f = open(RESULTS_PATH, "a", encoding="utf-8")

    def on_record(record):
        counts["wrong"] += 1
        subset = item_by_id[record.item_id].language
        category = item_by_id[record.item_id].metadata["category"]
        payload = asdict(record)
        payload["subset"] = subset
        payload["category"] = category
        out_f.write(json.dumps(payload, default=str) + "\n")
        out_f.flush()
        flipped_by = [rid for rid, o in record.repairs.items() if o["flipped"]]
        # Best non-placebo probability gain, printed even with no flip:
        # the point of recording deltas is that partial movement toward
        # the gold answer is still evidence about the cause.
        gains = [(o["delta_gold_prob"], rid) for rid, o in record.repairs.items()
                 if o["applied"] and not o["is_placebo"] and o["delta_gold_prob"] == o["delta_gold_prob"]]
        best = max(gains) if gains else None
        best_str = f"{best[1]} {best[0]:+.3f}" if best else "n/a"
        print(f"[{counts['correct']+counts['wrong']}/{len(items)}] {record.item_id}: "
              f"WRONG on clean (p_gold={record.clean_gold_prob_mass:.3f}), "
              f"flipped_by={flipped_by or 'none'}, best_gain={best_str}", flush=True)

    def on_skip_correct(item):
        counts["correct"] += 1
        print(f"[{counts['correct']+counts['wrong']}/{len(items)}] {item.item_id}: "
              f"correct on clean question, skipped", flush=True)

    def on_error(item, exc):
        counts["errored"] = counts.get("errored", 0) + 1
        out_f.write(json.dumps({"item_id": item.item_id, "error": str(exc)}) + "\n")
        out_f.flush()
        print(f"[{counts['correct']+counts['wrong']+counts['errored']}/{len(items)}] "
              f"{item.item_id}: FAILED: {exc}", flush=True)

    t0 = time.time()
    try:
        records = run_natural_failure_audit(
            items, runner, on_record=on_record, on_skip_correct=on_skip_correct, on_error=on_error
        )
    finally:
        out_f.close()

    print(f"\n=== DONE in {time.time()-t0:.0f}s: {counts['correct']} correct-on-clean, "
          f"{counts['wrong']} naturally wrong, {counts.get('errored', 0)} errored, "
          f"total {len(items)} ===", flush=True)
    print_repair_summary(records)


def print_repair_summary(records):
    """In-log summary so the result is readable from the kernel log alone,
    without pulling the jsonl. Reports each repair against the placebo
    floor -- a raw flip rate means nothing on its own, since any prompt
    edit perturbs greedy decoding. Full analysis lives in
    scripts/analyze_natural_failure_audit.py."""
    if not records:
        print("No naturally-wrong items -- nothing to summarize.", flush=True)
        return

    tested, flipped, deltas, is_placebo, axis = {}, {}, {}, {}, {}
    for r in records:
        for rid, o in r.repairs.items():
            is_placebo[rid] = o["is_placebo"]
            axis[rid] = o["axis"]
            if not o["applied"]:
                continue
            tested[rid] = tested.get(rid, 0) + 1
            flipped[rid] = flipped.get(rid, 0) + int(o["flipped"])
            d = o["delta_gold_prob"]
            if d == d:  # not NaN
                deltas.setdefault(rid, []).append(d)

    def rate(rid):
        return flipped.get(rid, 0) / tested[rid] if tested.get(rid) else 0.0

    floor = max((rate(rid) for rid in tested if is_placebo[rid]), default=0.0)
    print(f"\nPlacebo flip floor: {floor:.1%}", flush=True)
    print(f"{'repair':<34}{'axis':<18}{'tested':>7}{'flips':>7}{'rate':>8}{'vs floor':>10}{'mean dp':>9}", flush=True)
    for rid in sorted(tested, key=lambda k: -rate(k)):
        ds = deltas.get(rid, [])
        mean_d = sum(ds) / len(ds) if ds else float("nan")
        label = "PLACEBO (control)" if is_placebo[rid] else axis[rid]
        print(f"{rid:<34}{label:<18}{tested[rid]:>7}{flipped.get(rid, 0):>7}"
              f"{rate(rid):>7.1%}{rate(rid) - floor:>+9.1%}{mean_d:>9.3f}", flush=True)

    any_real = sum(1 for r in records
                   if any(o["flipped"] and not o["is_placebo"] for o in r.repairs.values()))
    print(f"\nFixed by at least one real repair: {any_real}/{len(records)} "
          f"({any_real/len(records):.0%})", flush=True)


if __name__ == "__main__":
    main()
