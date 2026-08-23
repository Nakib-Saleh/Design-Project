"""CULPRIT-VQA Phase 0 -- Kaggle GPU calibration run.

Purpose: measure REAL per-call latency and GPU memory for one quantization-free
small VLM (Qwen2.5-VL-3B-Instruct) on a handful of real CVQA images, before
committing compute budget to the full 100-200 image smoke test. Self-contained
(no dependency on the culprit_vqa package) so it can be pushed as a single
Kaggle script kernel.

Writes /kaggle/working/phase0_calibration_results.json with per-call timing,
memory, and correctness signal, plus a printed summary.
"""

import json
import subprocess
import sys
import time
import traceback

print("=== Phase 0: installing dependencies ===", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-U",
    "transformers>=4.51.0",
    "accelerate>=0.34.0",
    "qwen-vl-utils",
    "datasets",
])

import torch  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from datasets import load_dataset  # noqa: E402

RESULTS_PATH = "/kaggle/working/phase0_calibration_results.json"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
N_ITEMS = 20


def _get(record, *candidate_keys, default=None):
    """Return the first present key among several candidate spellings —
    used because the exact CVQA column naming is not verified locally
    against the live HF dataset schema."""
    for key in candidate_keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _parse_options(raw) -> list:
    """CVQA's options come as a list-like field (confirmed via the first
    calibration pass); handle a plain list, a numpy array, or (fallback) a
    string-encoded list, defensively."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(o) for o in raw]
    if hasattr(raw, "tolist"):  # numpy array
        return [str(o) for o in raw.tolist()]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            import ast

            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(o) for o in parsed]
            except (ValueError, SyntaxError):
                pass
        # last resort: split on common delimiters
        for delim in ["|", ";"]:
            if delim in text:
                return [p.strip() for p in text.split(delim)]
    return []


def gpu_mem_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main():
    results = {
        "model_id": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "calls": [],
        "errors": [],
    }

    print("=== Loading a small stratified CVQA sample ===", flush=True)
    t0 = time.time()
    ds = load_dataset("afaji/cvqa", split="test", streaming=True)
    items = []
    seen_pairs = set()
    for ex in ds:
        # CVQA's real schema (confirmed empirically, first calibration pass):
        # no separate Country/Language columns -- "Subset" encodes the
        # country-language pair directly (e.g. "Bangladesh_Bengali").
        subset = _get(ex, "Subset", "subset", default="")
        if subset in seen_pairs:
            continue
        seen_pairs.add(subset)
        items.append(ex)
        if len(items) >= N_ITEMS:
            break
    print(f"Collected {len(items)} items across {len(seen_pairs)} subsets "
          f"in {time.time() - t0:.1f}s", flush=True)
    results["n_items_collected"] = len(items)
    results["subsets"] = sorted(seen_pairs)
    if items:
        print(f"Example record keys: {sorted(items[0].keys())}", flush=True)
        results["example_keys"] = sorted(items[0].keys())

    print("=== Loading model (this includes first-time download) ===", flush=True)
    t0 = time.time()
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
        )
    except Exception:
        results["errors"].append({"stage": "model_load", "traceback": traceback.format_exc()})
        _dump(results)
        raise
    load_time = time.time() - t0
    results["model_load_seconds"] = load_time
    print(f"Model loaded in {load_time:.1f}s. Peak GPU mem so far: {gpu_mem_mb():.0f} MB", flush=True)

    for i, ex in enumerate(items):
        record = {
            "index": i,
            "subset": _get(ex, "Subset", "subset"),
            "category": _get(ex, "Category", "category"),
        }
        try:
            image = _get(ex, "image", "Image")
            question = _get(ex, "Translated Question", "Question", "translated_question", "question")
            options = _parse_options(_get(ex, "Translated Options", "Options", "translated_options", "options"))
            answer_idx = _get(ex, "Label", "label", "Answer", "answer", default=0)
            if i == 0:
                # One-time diagnostic dump of the raw (unparsed) fields so a
                # parsing mismatch is visible in results JSON even on failure.
                record["_raw_options_field"] = str(
                    _get(ex, "Translated Options", "Options", "translated_options", "options")
                )
                record["_raw_label_field"] = str(answer_idx)

            if isinstance(answer_idx, str) and not answer_idx.strip().lstrip("-").isdigit():
                answer_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer_idx.strip().upper())
            if isinstance(answer_idx, str) and answer_idx.strip().lstrip("-").isdigit():
                answer_idx = int(answer_idx.strip())

            if options and isinstance(answer_idx, int) and 0 <= answer_idx < len(options) and options[answer_idx]:
                gold_answer = options[answer_idx]
            elif isinstance(answer_idx, str) and options and answer_idx.strip() in options:
                gold_answer = answer_idx.strip()
            elif options:
                gold_answer = options[0]
            else:
                gold_answer = None

            if image is None or question is None or not options or gold_answer is None:
                raise ValueError(
                    "Could not extract required fields from example. "
                    f"image={image is not None} question={question!r} options={options!r} "
                    f"answer_idx={answer_idx!r}; available keys: {sorted(ex.keys())}"
                )

            option_letters = "ABCDEFGH"[: len(options)]
            options_str = " ".join(f"{letter}) {opt}" for letter, opt in zip(option_letters, options))
            prompt_text = f"{question}\nOptions: {options_str}\nAnswer with the option text only."
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}],
            }]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(model.device)

            # --- teacher-forced gold-answer log-prob (cheap p(S) estimate) ---
            # NOTE: approximate for this calibration script (tokenizing gold_answer
            # standalone can drift slightly from its in-context tokenization). Good
            # enough for timing/memory measurement; Phase 2's real ModelRunner will
            # align this properly (e.g. via offset mapping on the full sequence).
            t_fwd = time.time()
            full_text = chat_text + gold_answer
            forced_inputs = processor(text=[full_text], images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**forced_inputs)
            logits = out.logits[0]
            gold_ids = processor.tokenizer(gold_answer, add_special_tokens=False)["input_ids"]
            n_gold = len(gold_ids)
            if n_gold > 0 and logits.shape[0] > n_gold:
                relevant_logits = logits[-(n_gold + 1):-1]
                log_probs = torch.log_softmax(relevant_logits.float(), dim=-1)
                gold_logprob = sum(
                    log_probs[j, gold_ids[j]].item() for j in range(n_gold)
                ) / n_gold
            else:
                gold_logprob = None
            forward_time = time.time() - t_fwd

            # --- short generation for the CoT trace (used by behavioral signals) ---
            t_gen = time.time()
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            trace = processor.batch_decode(
                gen_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            gen_time = time.time() - t_gen

            record.update({
                "forward_seconds": forward_time,
                "generate_seconds": gen_time,
                "gold_answer": gold_answer,
                "predicted_trace": trace,
                "gold_avg_logprob": gold_logprob,
                "peak_gpu_mem_mb": gpu_mem_mb(),
            })
            print(f"[{i+1}/{len(items)}] fwd={forward_time:.2f}s gen={gen_time:.2f}s "
                  f"mem={record['peak_gpu_mem_mb']:.0f}MB pred={trace[:60]!r}", flush=True)
        except Exception:
            record["error"] = traceback.format_exc()
            print(f"[{i+1}/{len(items)}] FAILED: {record['error'][-300:]}", flush=True)
        results["calls"].append(record)
        _dump(results)  # checkpoint after every item

    ok_calls = [c for c in results["calls"] if "error" not in c]
    if ok_calls:
        avg_fwd = sum(c["forward_seconds"] for c in ok_calls) / len(ok_calls)
        avg_gen = sum(c["generate_seconds"] for c in ok_calls) / len(ok_calls)
        results["summary"] = {
            "n_ok": len(ok_calls),
            "n_failed": len(results["calls"]) - len(ok_calls),
            "avg_forward_seconds": avg_fwd,
            "avg_generate_seconds": avg_gen,
            "avg_total_seconds_per_condition": avg_fwd + avg_gen,
            "peak_gpu_mem_mb": max(c["peak_gpu_mem_mb"] for c in ok_calls),
        }
        print("\n=== SUMMARY ===")
        print(json.dumps(results["summary"], indent=2))
    else:
        print("\n=== ALL CALLS FAILED -- see results JSON for tracebacks ===")

    _dump(results)


def _dump(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
