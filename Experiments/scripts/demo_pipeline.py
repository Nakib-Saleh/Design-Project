"""End-to-end demo: runs the 5 synthetic fixture items through the full
CULPRIT-VQA pipeline (Layers 1->2->3a->3b, then 4, then 5) and prints a
summary. No network, no GPU, no real model weights — everything routes
through MockModelRunner and the stub validity functions.

Run: python scripts/demo_pipeline.py
"""

from culprit_vqa.fixtures.synthetic_items import build_synthetic_items, build_synthetic_runner
from culprit_vqa.layer3a_causal.attribution import efficiency_residual
from culprit_vqa.layer3a_causal.probability import build_p_function
from culprit_vqa.layer4_attributor.dataset import build_examples
from culprit_vqa.layer4_attributor.evaluate import spearman_correlation
from culprit_vqa.layer4_attributor.logistic import LogisticAttributor
from culprit_vqa.layer5_profiles.aggregate import build_failure_profile
from culprit_vqa.layer5_profiles.interaction_atlas import build_interaction_atlas
from culprit_vqa.pipeline import run_pipeline_for_item


def main() -> None:
    items = build_synthetic_items()
    runner = build_synthetic_runner(seed=0)

    print(f"=== Running {len(items)} synthetic items through the pipeline ===\n")
    records = []
    for item, factors in items:
        record = run_pipeline_for_item(item, factors, runner)
        records.append(record)

        p_fn = build_p_function(record.run_results)
        residual = efficiency_residual(record.phi, p_fn, record.factors)

        print(f"[{item.item_id}] k={len(factors)}, lattice size={len(record.run_results)}")
        print(f"  phi:   { {f.id: round(v, 4) for f, v in record.phi.items()} }")
        print(f"  alpha: { {f.id: round(v, 4) for f, v in record.alpha.items()} }")
        print(f"  leave-one-out: { {f.id: round(v, 4) for f, v in record.loo.items()} }")
        if record.interactions:
            print(
                "  interactions:  "
                f"{ {tuple(sorted(f.id for f in pair)): round(v, 4) for pair, v in record.interactions.items()} }"
            )
        print(f"  efficiency residual: {residual:.2e}")
        print()

    print("=== Layer 4: amortized attributor (train on all-but-last, predict held-out) ===\n")
    train_records, held_out_record = records[:-1], records[-1]
    train_examples = build_examples(train_records)
    held_out_examples = build_examples([held_out_record])

    attributor = LogisticAttributor()
    attributor.fit(train_examples)
    predicted_alpha = attributor.predict_alpha_for_item(held_out_examples)
    true_alpha = {ex.factor_id: held_out_record.alpha.get(f, 0.0)
                  for ex in held_out_examples
                  for f in held_out_record.factors if f.id == ex.factor_id}

    print(f"  held-out item: {held_out_record.item.item_id}")
    print(f"  true alpha:      {  {k: round(v, 4) for k, v in true_alpha.items()} }")
    print(f"  predicted alpha: {  {k: round(v, 4) for k, v in predicted_alpha.items()} }")
    if len(true_alpha) >= 2:
        rho = spearman_correlation(list(true_alpha.values()), list(predicted_alpha.values()))
        print(f"  Spearman rho (n={len(true_alpha)}, too small to be conclusive): {rho}")
    else:
        print("  (fewer than 2 factors on the held-out item — Spearman rho not defined)")
    print()

    print("=== Layer 5: failure profiles ===\n")
    profiles = build_failure_profile(records, group_by=lambda r: r.item.language)
    for key, profile in profiles.items():
        print(f"  language={key}: n_items={profile.n_items}, "
              f"mixture={ {k: round(v, 4) for k, v in profile.factor_mixture.items()} }")

    atlas = build_interaction_atlas(records)
    print(f"\n  interaction atlas: { {tuple(sorted(p)): round(v, 4) for p, v in atlas.items()} }")


if __name__ == "__main__":
    main()
