"""Mini calibration for the genericized HFVisionLanguageRunner:
  (1) regression-check that switching to AutoModelForImageTextToText
      still works correctly for Qwen2.5-VL-3B (the model everything so
      far has used),
  (2) first-contact check for a second, different model family
      (LLaVA-OneVision-0.5B) for the planned model-comparison experiment.

Small scale (10 items per model) on purpose -- this is a "does it even
work" check before committing to a full run, not a real experiment.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

print("=== Installing GPU-stack dependencies ===", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.51.0", "accelerate>=0.34.0", "bitsandbytes>=0.43.0",
    "qwen-vl-utils", "datasets",
])

from datasets import load_dataset  # noqa: E402

from culprit_vqa.layer0_taxonomy.operators import get_operator  # noqa: E402
from culprit_vqa.layer1_intervention.items import Item  # noqa: E402
from culprit_vqa.layer1_intervention.lattice import generate_lattice  # noqa: E402
from culprit_vqa.layer2_runner.hf_runner import HFVisionLanguageRunner  # noqa: E402

RESULTS_PATH = "/kaggle/working/model_calibration_results.json"
N_ITEMS = 10
MODELS_TO_TEST = [
    # Feasibility probe for upgrading the second model from 0.5B to 7B.
    # The 0.5B is a poor comparison subject for the natural-failure
    # audit specifically: nearly all of its mistakes are "too weak to do
    # the task", which no hint can repair, so the audit would return
    # ~everything unexplained and teach us nothing about whether the
    # diagnosis method works. 7B is competent enough for its failures to
    # be repairable-in-principle, which is what the audit needs.
    #
    # Two things must hold before committing to the 220-item run, and
    # both have bitten this project before: it must fit in a T4's 16GB
    # (the first full run OOM'd on 51/100 items), and per-call latency
    # must be low enough that ~700 calls fit the session limit -- 4-bit
    # inference on a T4 is dequantization-bound and can be far slower
    # than the parameter count suggests.
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
]


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
    items = []
    for i, ex in enumerate(ds):
        options = _parse_options(_get(ex, "Translated Options", "Options"))
        image = _get(ex, "image")
        question = _get(ex, "Translated Question", "Question")
        if not options or image is None or question is None:
            continue
        correct_idx = _parse_answer_idx(_get(ex, "Label", default=0), len(options))
        items.append(Item(
            item_id=f"calib_{len(items):02d}",
            image_ref=f"cvqa::{len(items)}",
            question=question, answer=options[correct_idx],
            language=str(_get(ex, "Subset", default="unknown")),
            metadata={"pil_image": image, "options": options, "correct_idx": correct_idx,
                      "category": _get(ex, "Category", default="unknown")},
        ))
        if len(items) >= n_items:
            break
    return items


def test_model(model_id, items, results):
    entry = {"model_id": model_id, "calls": [], "errors": []}
    print(f"\n=== Testing model: {model_id} ===", flush=True)
    t0 = time.time()
    try:
        runner = HFVisionLanguageRunner(model_id=model_id, seed=0)
    except Exception:
        entry["load_error"] = traceback.format_exc()
        print(f"FAILED TO LOAD {model_id}: {entry['load_error'][-500:]}", flush=True)
        results.append(entry)
        return
    entry["load_seconds"] = time.time() - t0
    print(f"Loaded in {entry['load_seconds']:.1f}s", flush=True)

    factor = get_operator("salience_recomposition")
    for item in items:
        lattice = generate_lattice(item, [factor])
        control = next(c for c in lattice if not c.applied_factors)
        t_call = time.time()
        try:
            r = runner.run(item, control, n_decodes=1)
            call_time = time.time() - t_call
            entry["calls"].append({
                "item_id": item.item_id, "seconds": call_time,
                "correct": bool(r.correct_flags[0]), "trace": r.cot_traces[0][:80],
                "gen_confidence": r.extra_signals.get("gen_confidence"),
            })
            print(f"  [{item.item_id}] {call_time:.2f}s correct={r.correct_flags[0]} "
                  f"trace={r.cot_traces[0][:60]!r}", flush=True)
        except Exception:
            err = traceback.format_exc()
            entry["errors"].append({"item_id": item.item_id, "error": err[-500:]})
            print(f"  [{item.item_id}] FAILED: {err[-300:]}", flush=True)

    import torch

    # Peak memory and per-call latency are the two numbers that decide
    # whether the full 220-item audit is safe to launch, so record them
    # explicitly rather than inferring them from whether this small
    # probe happened to survive.
    entry["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    entry["total_gpu_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    times = [c["seconds"] for c in entry["calls"]]
    if times:
        # Ignore the first call: it pays one-off CUDA kernel autotuning
        # and allocator warm-up that the remaining ~700 calls will not.
        steady = times[1:] or times
        entry["median_call_seconds"] = sorted(steady)[len(steady) // 2]
        entry["max_call_seconds"] = max(steady)
        # The audit makes ~1 clean call per pooled item plus ~6 repair
        # calls per naturally-wrong item; at the observed ~63% accuracy
        # on 220 items that is roughly 220 + 80*6 = 700 calls.
        entry["projected_audit_hours"] = entry["median_call_seconds"] * 700 / 3600
        print(f"  median call {entry['median_call_seconds']:.2f}s, "
              f"peak GPU {entry['peak_gpu_gb']:.1f}/{entry['total_gpu_gb']:.1f} GB, "
              f"projected 700-call audit {entry['projected_audit_hours']:.1f}h", flush=True)

    del runner
    torch.cuda.empty_cache()
    results.append(entry)


def main():
    items = collect_items(N_ITEMS)
    print(f"Collected {len(items)} calibration items", flush=True)

    results = []
    for model_id in MODELS_TO_TEST:
        test_model(model_id, items, results)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n=== SUMMARY ===", flush=True)
    for entry in results:
        n_ok = len(entry.get("calls", []))
        n_err = len(entry.get("errors", []))
        n_correct = sum(1 for c in entry.get("calls", []) if c["correct"])
        print(f"{entry['model_id']}: load_error={'load_error' in entry}, ok={n_ok}, errors={n_err}")
        if n_ok:
            # Accuracy on 10 items is far too small to report as a result,
            # but it does answer the one question that matters here: does
            # this model produce failures worth auditing? At 0/10 it is
            # too weak (nothing repairable); at 10/10 it yields no
            # failures to diagnose at all.
            print(f"  correct {n_correct}/{n_ok}, peak GPU "
                  f"{entry.get('peak_gpu_gb', float('nan')):.1f}/"
                  f"{entry.get('total_gpu_gb', float('nan')):.1f} GB, "
                  f"median call {entry.get('median_call_seconds', float('nan')):.2f}s, "
                  f"projected audit {entry.get('projected_audit_hours', float('nan')):.1f}h")


if __name__ == "__main__":
    main()
