"""Collect 150 small VQA items for the Calibrated Repair Diagnosis smoke test.

Source: CVQA (afaji/cvqa, HuggingFace) -- an open, permissively licensed
multilingual/multicultural VQA benchmark.

WHY CVQA AND NOT VQAv2 / OK-VQA / A-OKVQA
-----------------------------------------
Part A needs ONE item pool on which ALL five damage types can be applied.
Most VQA sets cannot support two of them at all:

  * the LANGUAGE damage needs a real human-written question in a
    non-English language AND its English translation for the same item.
    CVQA ships both per row; the English-only sets ship neither.
  * the CULTURE damage needs a culturally grounded image, so that
    asserting a wrong origin is actually false rather than meaningless.
    CVQA is built from 39 language-country communities.

It is also multiple-choice, which the overlay damage needs (there must be
a specific wrong option to write onto the image) and which makes scoring
well-posed.

ASSIGNMENT: ROUND-ROBIN, NOT BY SUITABILITY
-------------------------------------------
The tempting thing is to hand-pick culturally-loaded images for the
culture damage and close-ups for the crop damage. That would wreck the
experiment: the translation table's rows would then differ in item
properties as well as in damage type, so any diagonal could be explained
by "cultural items are just harder" rather than by targeted repair.

Every item here is screened to support all five damages, and damage is
assigned by round-robin over a subset-stratified ordering. Language,
country, category and image size spread evenly across all five rows by
construction.

MEMORY AND DISK
---------------
Written for constrained hardware. Images are downscaled and written to
disk AS THEY STREAM; only metadata is held in RAM, so peak memory stays
flat regardless of pool size. Storing at MAX_SIDE_OUT (below the runner's
own 768px cap) means the file on disk is exactly what the model will see
-- no second resize at inference, and the smoke test is reproducible from
the saved images alone.
"""

import io
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "smoke150"
IMG_DIR = OUT / "images"
STAGE_DIR = OUT / "_staging"
MANIFEST = OUT / "manifest.jsonl"

N_TOTAL = 150
MAX_SIDE_OUT = 512      # downscale only; never upscales a smaller original
JPEG_QUALITY = 85       # ~40-70KB per image -> ~8MB for the whole set
MIN_SIDE_IN = 250       # crop damage needs room to remove evidence
MIN_OPTIONS = 3         # overlay damage needs a wrong option to name
CANDIDATE_POOL = 360    # ~2.4x N_TOTAL: enough to stratify, cheap to stage
MAX_SCAN = 2500
SEED = 0

# Five damage types, one per axis emphasis. Every kept item satisfies all
# five preconditions, which is what makes round-robin assignment valid.
DAMAGES = [
    ("overlay_wrong_answer", "cross-modal",
     "write a wrong answer onto the image", "look_closer"),
    ("crop_away_evidence", "visual",
     "crop/rescale so the deciding evidence is gone", "look_closer"),
    ("wrong_cultural_origin", "culturality",
     "assert the photo comes from another culture", "supply_cultural_context"),
    ("swap_to_native_language", "modality_text",
     "ask the original non-English question instead", "ask_in_native_language"),
    ("bury_in_irrelevant_text", "relevance",
     "prepend plausible but irrelevant filler", "think_step_by_step"),
]


def norm_options(raw):
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [p.strip() for p in raw.split("|")]
    return [str(o).strip() for o in raw if str(o).strip()]


def shrink(img):
    """Downscale so the longer side is at most MAX_SIDE_OUT. Never upscales."""
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_SIDE_OUT:
        scale = MAX_SIDE_OUT / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return img.convert("RGB")


def main():
    from datasets import load_dataset

    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    shutil.rmtree(IMG_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("streaming CVQA test split ...", flush=True)
    ds = load_dataset("afaji/cvqa", split="test", streaming=True)

    by_subset = defaultdict(list)
    scanned = kept = 0
    reject = Counter()

    for ex in ds:
        scanned += 1
        if scanned > MAX_SCAN or kept >= CANDIDATE_POOL:
            break

        img = ex.get("image")
        q_en = ex.get("Translated Question")
        q_native = ex.get("Question")
        opts_en = norm_options(ex.get("Translated Options"))
        opts_native = norm_options(ex.get("Options"))
        label = ex.get("Label")
        subset = ex.get("Subset")
        category = ex.get("Category")

        if img is None:
            reject["no image"] += 1
            continue
        if min(img.size) < MIN_SIDE_IN:
            reject["image too small"] += 1
            continue
        if not q_en or not opts_en or len(opts_en) < MIN_OPTIONS:
            reject["missing question/options"] += 1
            continue
        if not isinstance(label, int) or not (0 <= label < len(opts_en)):
            reject["bad label"] += 1
            continue
        # Language damage precondition: a REAL native question distinct from
        # the English one, or the damage silently no-ops.
        if not q_native or q_native == q_en:
            reject["no distinct native question"] += 1
            continue
        # Culture damage precondition: a known (Language, Country) so the
        # asserted origin is actually wrong.
        if not (isinstance(subset, (list, tuple)) and len(subset) >= 2):
            reject["no subset pair"] += 1
            continue
        if not category:
            reject["no category"] += 1
            continue

        # Shrink and write NOW; keep only the path in memory.
        stage_name = f"c{kept:04d}.jpg"
        small = shrink(img)
        small.save(STAGE_DIR / stage_name, "JPEG", quality=JPEG_QUALITY, optimize=True)
        w, h = small.size
        del img, small

        by_subset[f"{subset[0]}::{subset[1]}"].append({
            "stage": stage_name,
            "question_en": q_en.strip(),
            "question_native": q_native.strip(),
            "options_en": opts_en,
            "options_native": opts_native or opts_en,
            "correct_idx": int(label),
            "answer_en": opts_en[int(label)],
            "language": str(subset[0]),
            "country": str(subset[1]),
            "category": str(category),
            "width": w,
            "height": h,
        })
        kept += 1
        if kept % 60 == 0:
            print(f"  scanned {scanned}, staged {kept}, subsets {len(by_subset)}", flush=True)

    print(f"\nscanned {scanned} rows; {kept} eligible across {len(by_subset)} subsets")
    print("rejections:", dict(reject.most_common()))
    if kept < N_TOTAL:
        raise SystemExit(f"only {kept} eligible items; need {N_TOTAL}")

    # --- stratify: round-robin across subsets so no culture dominates ---
    rng = random.Random(SEED)
    for k in by_subset:
        rng.shuffle(by_subset[k])
    keys = sorted(by_subset)
    ordered, idx = [], {k: 0 for k in keys}
    while len(ordered) < N_TOTAL:
        progressed = False
        for k in keys:
            if idx[k] < len(by_subset[k]):
                ordered.append(by_subset[k][idx[k]])
                idx[k] += 1
                progressed = True
                if len(ordered) >= N_TOTAL:
                    break
        if not progressed:
            break

    # --- assign damage round-robin over the stratified order ---
    records = []
    for i, item in enumerate(ordered):
        d_id, axis, desc, repair = DAMAGES[i % len(DAMAGES)]
        item_id = f"s150_{i:03d}"
        stage = item.pop("stage")
        shutil.move(str(STAGE_DIR / stage), str(IMG_DIR / f"{item_id}.jpg"))
        records.append({
            "item_id": item_id,
            "image_path": f"images/{item_id}.jpg",
            **item,
            "assigned_damage": d_id,
            "damage_axis": axis,
            "damage_description": desc,
            "matched_repair": repair,
        })

    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----------------------------------------------------------- report
    total_bytes = sum(p.stat().st_size for p in IMG_DIR.glob("*.jpg"))
    sides = [max(r["width"], r["height"]) for r in records]
    print(f"\nwrote {len(records)} items -> {MANIFEST}")
    print(f"images  -> {IMG_DIR}")
    print(f"disk    -> {total_bytes/1e6:.1f} MB total, "
          f"{total_bytes/len(records)/1024:.0f} KB avg")
    print(f"longest side: min {min(sides)}, max {max(sides)} px (cap {MAX_SIDE_OUT})")

    print(f"\n{'damage type':<26}{'axis':<14}{'n':>4}  matched repair")
    print("-" * 76)
    dc = Counter(r["assigned_damage"] for r in records)
    for d_id, axis, _desc, repair in DAMAGES:
        print(f"{d_id:<26}{axis:<14}{dc[d_id]:>4}  {repair}")

    print(f"\nlanguages: {len(set(r['language'] for r in records))}   "
          f"countries: {len(set(r['country'] for r in records))}   "
          f"categories: {len(set(r['category'] for r in records))}")

    # Balance check: each damage row must span many cultures/categories or
    # the translation table is confounded.
    print(f"\n{'damage type':<26}{'langs':>6}{'cats':>6}{'avg px':>8}{'opts':>6}")
    print("-" * 76)
    for d_id, *_ in DAMAGES:
        rows = [r for r in records if r["assigned_damage"] == d_id]
        print(f"{d_id:<26}"
              f"{len({r['language'] for r in rows}):>6}"
              f"{len({r['category'] for r in rows}):>6}"
              f"{sum(max(r['width'], r['height']) for r in rows)/len(rows):>8.0f}"
              f"{sum(len(r['options_en']) for r in rows)/len(rows):>6.1f}")

    print(f"\ntop categories: {Counter(r['category'] for r in records).most_common(5)}")
    print("\nsample items:")
    for r in records[:4]:
        print(f"  [{r['item_id']}] {r['language']}/{r['country']} | {r['category']} "
              f"| {r['width']}x{r['height']}")
        print(f"     EN : {r['question_en'][:68]}")
        print(f"     NAT: {r['question_native'][:68]}")
        print(f"     opts: {r['options_en']} -> {r['answer_en']}")
        print(f"     damage: {r['assigned_damage']}")


if __name__ == "__main__":
    main()
