"""Summarizes a natural_failure_results.jsonl (from kaggle_kernels/natural_failure_audit/)
-- no GPU needed, pure local analysis.

Reports each repair's flip rate AGAINST the placebo floor rather than on
its own. A raw flip rate is uninterpretable: greedy decoding is sensitive
to any prompt change at all, so some fraction of "fixes" are just
perturbation noise. The placebo repairs add comparable text carrying no
usable information, so their flip rate estimates that floor, and a real
repair is only evidence for its axis if it clears it.

Also reports mean change in the model's probability on the gold answer,
which captures repairs that moved the model toward the right answer
without crossing the decision boundary -- invisible to a flip count.

Run: python scripts/analyze_natural_failure_audit.py <path-to-natural_failure_results.jsonl>
"""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict


def _finite(values):
    return [v for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]


def _mean(values):
    vals = _finite(values)
    return statistics.mean(vals) if vals else float("nan")


def _fmt(x, places=3):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{places}f}"


def load_records(path):
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    ok = [r for r in records if "error" not in r and "fatal_error" not in r and "repairs" in r]
    errors = [r for r in records if r not in ok]
    return ok, errors


def summarize_repairs(ok):
    """Per-repair: how many items it was actually tested on, how many it
    flipped, and how much it moved gold probability on average."""
    stats = defaultdict(lambda: {"tested": 0, "flipped": 0, "deltas": [], "axis": "?", "placebo": False})
    for r in ok:
        for rid, o in r["repairs"].items():
            s = stats[rid]
            s["axis"] = o.get("axis", "?")
            s["placebo"] = bool(o.get("is_placebo", False))
            # An unapplied repair (e.g. native-language on an item with no
            # native text) was never tested -- counting it as a failure
            # would understate that repair's true rate.
            if not o.get("applied", True):
                continue
            s["tested"] += 1
            if o["flipped"]:
                s["flipped"] += 1
            s["deltas"].append(o.get("delta_gold_prob"))
    return stats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "kaggle_kernels/natural_failure_audit/output/natural_failure_results.jsonl"
    ok, errors = load_records(path)
    if not ok:
        print(f"No usable records in {path} ({len(errors)} error rows).")
        return

    print(f"Naturally-wrong items recorded: {len(ok)}  (error rows: {len(errors)})")
    print(f"Mean p(gold) on the clean question: {_fmt(_mean([r.get('clean_gold_prob_mass') for r in ok]))}\n")

    stats = summarize_repairs(ok)
    placebos = {rid: s for rid, s in stats.items() if s["placebo"]}
    reals = {rid: s for rid, s in stats.items() if not s["placebo"]}

    # The noise floor: the best-performing placebo, not the average.
    # Using the average would let a real repair "beat the placebo" while
    # still doing no better than the stronger of two information-free
    # prompt edits, which is not evidence of anything.
    placebo_rates = [s["flipped"] / s["tested"] for s in placebos.values() if s["tested"]]
    floor = max(placebo_rates) if placebo_rates else 0.0
    placebo_deltas = [d for s in placebos.values() for d in _finite(s["deltas"])]
    delta_floor = max(
        (_mean(s["deltas"]) for s in placebos.values() if _finite(s["deltas"])), default=0.0
    )

    print("=== PLACEBO CONTROLS (the noise floor) ===")
    for rid, s in sorted(placebos.items()):
        rate = s["flipped"] / s["tested"] if s["tested"] else float("nan")
        print(f"  {rid}: {s['flipped']}/{s['tested']} flipped ({_fmt(rate * 100, 1)}%), "
              f"mean delta p(gold) = {_fmt(_mean(s['deltas']))}")
    print(f"  -> flip floor: {_fmt(floor * 100, 1)}%   delta floor: {_fmt(delta_floor)}")
    print(f"  (n placebo observations: {len(placebo_deltas)})\n")

    print("=== REAL REPAIRS (vs floor) ===")
    header = f"  {'repair':<26} {'axis':<16} {'tested':>7} {'flips':>6} {'rate':>8} {'vs floor':>9} {'mean d p(gold)':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    verdicts = []
    for rid, s in sorted(reals.items(), key=lambda kv: -(kv[1]["flipped"] / kv[1]["tested"] if kv[1]["tested"] else 0)):
        rate = s["flipped"] / s["tested"] if s["tested"] else float("nan")
        lift = rate - floor if math.isfinite(rate) else float("nan")
        mean_delta = _mean(s["deltas"])
        print(f"  {rid:<26} {s['axis']:<16} {s['tested']:>7} {s['flipped']:>6} "
              f"{_fmt(rate * 100, 1):>7}% {_fmt(lift * 100, 1):>8}% {_fmt(mean_delta):>15}")
        verdicts.append((rid, lift, mean_delta - delta_floor))

    print("\n  Clears the placebo floor on flips:")
    cleared = [rid for rid, lift, _ in verdicts if math.isfinite(lift) and lift > 0]
    print(f"    {', '.join(cleared) if cleared else 'NONE -- no repair beat an information-free prompt edit'}")
    print("  Clears the placebo floor on p(gold) movement:")
    cleared_d = [rid for rid, _, dlift in verdicts if math.isfinite(dlift) and dlift > 0]
    print(f"    {', '.join(cleared_d) if cleared_d else 'NONE'}")

    # --- coverage: how many failures ANY real repair explains ---
    def real_flips(r):
        return [rid for rid, o in r["repairs"].items()
                if o.get("applied", True) and not o.get("is_placebo") and o["flipped"]]

    any_real = sum(1 for r in ok if real_flips(r))
    multi = sum(1 for r in ok if len(real_flips(r)) > 1)
    print(f"\n=== COVERAGE ===")
    print(f"  Fixed by at least one real repair: {any_real}/{len(ok)} ({any_real/len(ok):.0%})")
    print(f"  Fixed by more than one (multi-causal, a single label would hide this): "
          f"{multi}/{len(ok)} ({multi/len(ok):.0%})")
    print(f"  Unexplained by every real repair: {len(ok)-any_real}/{len(ok)} ({(len(ok)-any_real)/len(ok):.0%})")

    # --- which axis wins where: the diagnosis, established causally ---
    axis_by_subset = defaultdict(Counter)
    axis_totals = Counter()
    for r in ok:
        for rid in real_flips(r):
            axis = stats[rid]["axis"]
            axis_totals[axis] += 1
            axis_by_subset[r.get("subset", "unknown")][axis] += 1
    if axis_totals:
        print("\n=== DIAGNOSED CAUSE (axis of whichever repair worked) ===")
        for axis, n in axis_totals.most_common():
            print(f"  {axis}: {n}")

    by_category = Counter(r.get("category", "unknown") for r in ok)
    print("\n=== Naturally-wrong items by category ===")
    for cat, n in by_category.most_common(10):
        print(f"  {cat}: {n}")

    print("\n=== Sample records ===")
    for r in ok[:5]:
        print(f"{r['item_id']} ({r.get('subset')}, {r.get('category')}): "
              f"fixed_by={real_flips(r) or 'none'}")
        print(f"  clean trace: {r['clean_trace'][:100]!r}")


if __name__ == "__main__":
    main()
