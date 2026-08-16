# CULPRIT-VQA

Causal failure-attribution framework for multimodal VQA. See
[PROPOSAL_CULPRIT_VQA.md](PROPOSAL_CULPRIT_VQA.md) for the full research
design.

This is an early scaffold. Layers 0/1/3a (factor taxonomy, factorial
intervention engine, Shapley causal ground truth — the paper's
centerpiece) are implemented for real. Layers 2/3b/4/5 (model runner,
behavioral signals, amortized attributor, failure profiles) are
minimal-but-working implementations, expanded in later thesis work.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      |  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

(If you have `uv`, `uv venv && uv pip install -e ".[dev]"` works the same way.)

## Running the tests

```bash
pytest -m "not integration"     # hermetic Layer 0-5 suite — no network/GPU needed
python scripts/demo_pipeline.py # end-to-end run over 5 synthetic fixture items, prints a summary
```

Every test in `tests/` asserts a real, checkable condition — lattice
sizes, the Shapley efficiency/symmetry/null-player axioms, validity-check
accept/reject behavior, the attribution null case, and a full pipeline
smoke test. No network, GPU, or real model weights are required.

## Kaggle connectivity check (optional, real network)

A separate, additive check that the environment can authenticate to a
real external data source (`secrets.env`, gitignored, holds the Kaggle
credential — never commit it):

```bash
pip install -e ".[kaggle]"
python scripts/check_kaggle_connection.py
# or: pytest -m integration
```

Note: CVQA itself is distributed via HuggingFace / cvqa-benchmark.org,
not Kaggle — this check downloads a small generic public Kaggle dataset
purely to validate the auth+download mechanics for later real-substrate
work.

## Project layout

```
src/culprit_vqa/
  layer0_taxonomy/     # 5-axis factor space, operator library, pillar mapping
  layer1_intervention/ # Item model, factorial lattice generation, 3-way validity checks
  layer2_runner/       # ModelRunner interface + deterministic MockModelRunner
  layer3a_causal/      # Shapley ground truth, leave-one-out, interaction indices
  layer3b_signals/     # Behavioral signals (non-causal, amortization features only)
  layer4_attributor/   # Amortized attributor (logistic regression)
  layer5_profiles/     # Per-group failure profiles + interaction atlas
  pipeline.py          # Wires Layers 1->2->3a->3b for one item
  fixtures/            # Hand-built synthetic items shared by tests and the demo script
scripts/
  demo_pipeline.py            # end-to-end synthetic demo
  check_kaggle_connection.py  # real-network Kaggle connectivity check
tests/                 # mirrors src/, one test module per layer + end-to-end smoke test
```
