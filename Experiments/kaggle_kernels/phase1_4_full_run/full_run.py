"""CULPRIT-VQA Phase 1-4 -- full scaled smoke test on real CVQA images.

Runs ~100 real CVQA items through the actual (tested locally) Layers
0/1/2/3a/3b pipeline: real factorial lattices, a real quantized VLM
(Qwen2.5-VL-3B-Instruct, 4-bit), real Shapley causal ground truth, real
(if minimal) behavioral signals. Layers 4/5 (amortized attributor,
failure profiles) run locally afterward on the checkpointed output using
the same tested code -- no GPU needed for those.

Writes /kaggle/working/phase1_4_results.jsonl -- one JSON record per item,
appended after every item so a quota cutoff loses at most one item.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback

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
    # Last resort: search one level deep for a directory containing __init__.py
    # plus a layer0_taxonomy subfolder (i.e. the actual package root).
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
    raise FileNotFoundError(
        f"Could not locate the culprit_vqa package under /kaggle/input. "
        f"Contents: {os.listdir('/kaggle/input') if os.path.exists('/kaggle/input') else 'MISSING'}"
    )
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

from culprit_vqa.layer0_taxonomy.operators import get_operator  # noqa: E402
from culprit_vqa.layer1_intervention.items import Item  # noqa: E402
from culprit_vqa.layer1_intervention.real_operators import REAL_OPERATORS  # noqa: E402
from culprit_vqa.layer2_runner.hf_runner import HFVisionLanguageRunner  # noqa: E402
from culprit_vqa.layer3a_causal.attribution import efficiency_residual  # noqa: E402
from culprit_vqa.layer3a_causal.probability import build_p_function  # noqa: E402
from culprit_vqa.pipeline import run_pipeline_for_item  # noqa: E402

RESULTS_PATH = "/kaggle/working/phase1_4_results.jsonl"
N_ITEMS = 150
OPERATOR_IDS = list(REAL_OPERATORS.keys())  # 4 operators with real apply() functions
SEED = 0


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
    """Deterministically build n_items real (base image/question/options)
    records from CVQA, cycling through subsets for stratification, then
    cycling further to fill the requested count."""
    ds = load_dataset("afaji/cvqa", split="test", streaming=True)
    all_examples = []
    for ex in ds:
        options = _parse_options(_get(ex, "Translated Options", "Options"))
        image = _get(ex, "image")
        question = _get(ex, "Translated Question", "Question")
        if not options or image is None or question is None:
            continue
        all_examples.append(ex)
        if len(all_examples) >= max(n_items * 3, 300):
            break  # enough raw pool to sample from without exhausting the stream

    # Stratify: one pass over distinct subsets first, then cycle for the rest.
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


def assign_factor_plan(n_items: int, seed: int = SEED):
    """~80% of items get k=2 (cycling through all 6 pairs of the 4 real
    operators), ~20% get k=3 (cycling through all 4 triples) -- so the
    interaction-index and fuller-lattice code paths both get real
    coverage, not just the common case."""
    import itertools

    pairs = list(itertools.combinations(OPERATOR_IDS, 2))
    triples = list(itertools.combinations(OPERATOR_IDS, 3))
    n_k3 = max(1, round(n_items * 0.2))
    n_k2 = n_items - n_k3

    plan = []
    for i in range(n_k2):
        plan.append(list(pairs[i % len(pairs)]))
    for i in range(n_k3):
        plan.append(list(triples[i % len(triples)]))
    random.Random(seed).shuffle(plan)
    return plan


def build_item(ex, index: int) -> Item:
    options = _parse_options(_get(ex, "Translated Options", "Options"))
    correct_idx = _parse_answer_idx(_get(ex, "Label", default=0), len(options))
    question = _get(ex, "Translated Question", "Question")
    native_question = _get(ex, "Question")
    image = _get(ex, "image")
    subset = str(_get(ex, "Subset", default="unknown"))
    category = _get(ex, "Category", default="unknown")

    return Item(
        item_id=f"cvqa_{index:04d}",
        image_ref=f"cvqa::{subset}::{index}",
        question=question,
        answer=options[correct_idx],
        evidence_region=None,
        language=subset,
        perturbation_effects={},
        metadata={
            "pil_image": image,
            "options": options,
            "correct_idx": correct_idx,
            "subset": subset,
            "category": category,
            "native_question": native_question,
        },
    )


def compute_language_delta(item: Item, runner) -> float:
    """S6, computed once per item (a property of the item/model pair, not
    of a specific perturbation): English gold-prob-mass minus native-
    language gold-prob-mass on the CLEAN question. Positive means the
    model does better in English than in the question's original
    language -- a real language/cultural-context signal, using data CVQA
    already provides (its English translation field) rather than a
    synthetic parallel item."""
    native_question = item.metadata.get("native_question")
    if not native_question or native_question == item.question:
        return 0.0
    english_result = runner.answer_with_question_override(item, item.question, run_label=f"{item.item_id}::lang_en")
    native_result = runner.answer_with_question_override(item, native_question, run_label=f"{item.item_id}::lang_native")
    return english_result.gold_prob_mass - native_result.gold_prob_mass


def serialize_record(record) -> dict:
    p_fn = build_p_function(record.run_results)
    residual = efficiency_residual(record.phi, p_fn, record.factors)
    return {
        "item_id": record.item.item_id,
        "subset": record.item.language,
        "category": record.item.metadata.get("category"),
        "factors": [f.id for f in record.factors],
        "phi": {f.id: v for f, v in record.phi.items()},
        "alpha": {f.id: v for f, v in record.alpha.items()},
        "loo": {f.id: v for f, v in record.loo.items()},
        "interactions": {"|".join(sorted(f.id for f in pair)): v for pair, v in record.interactions.items()},
        "efficiency_residual": residual,
        "p_values": {
            "|".join(sorted(f.id for f in subset)) or "control": p_fn(subset)
            for subset in record.run_results
        },
        "signal_vectors": {
            cond_id: sv.values for cond_id, sv in record.signal_vectors.items()
        },
        "language_delta": record.item.metadata.get("language_delta", 0.0),
        "sample_trace": record.run_results[frozenset()].cot_traces[0][:150],
    }


def main():
    print("=== Collecting real CVQA items ===", flush=True)
    t0 = time.time()
    examples = collect_real_items(N_ITEMS)
    print(f"Collected {len(examples)} items in {time.time()-t0:.1f}s", flush=True)
    if len(examples) < N_ITEMS:
        print(f"WARNING: only got {len(examples)}/{N_ITEMS} valid items", flush=True)

    plan = assign_factor_plan(len(examples))

    print("=== Loading quantized model ===", flush=True)
    t0 = time.time()
    runner = HFVisionLanguageRunner(model_id="Qwen/Qwen2.5-VL-3B-Instruct", seed=SEED)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    n_ok, n_failed = 0, 0
    with open(RESULTS_PATH, "a", encoding="utf-8") as out_f:
        for i, ex in enumerate(examples):
            factor_ids = plan[i]
            factors = [get_operator(fid) for fid in factor_ids]
            item = build_item(ex, i)
            t_item = time.time()
            try:
                item.metadata["language_delta"] = compute_language_delta(item, runner)
                record = run_pipeline_for_item(item, factors, runner, n_decodes=1)
                serialized = serialize_record(record)
                out_f.write(json.dumps(serialized, default=str) + "\n")
                out_f.flush()
                n_ok += 1
                print(f"[{i+1}/{len(examples)}] OK item={item.item_id} k={len(factors)} "
                      f"time={time.time()-t_item:.1f}s residual={serialized['efficiency_residual']:.2e}",
                      flush=True)
            except Exception:
                n_failed += 1
                err = traceback.format_exc()
                out_f.write(json.dumps({
                    "item_id": item.item_id, "error": err, "factors": factor_ids
                }) + "\n")
                out_f.flush()
                print(f"[{i+1}/{len(examples)}] FAILED item={item.item_id}: {err[-300:]}", flush=True)

    print(f"\n=== DONE: {n_ok} ok, {n_failed} failed, total {len(examples)} ===", flush=True)


if __name__ == "__main__":
    main()
