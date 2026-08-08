# CULPRIT-VQA — Handoff / Progress Context

**Read this file first, in full, before touching anything else.** It exists so a new
Claude session (or a new developer) can resume this project without re-deriving
anything already learned the hard way. Everything stated here as fact has been
verified by actually running it — none of this is aspirational.

Companion documents: [PROPOSAL_CULPRIT_VQA.md](PROPOSAL_CULPRIT_VQA.md) (the research
design/thesis proposal — read this second, for *why*) and [README.md](README.md) (short
project-structure overview).

---

## 1. One-paragraph status

**READ THIS BEFORE TRUSTING ANY EARLIER RESULT IN THIS FILE.** CULPRIT-VQA is a
causal failure-attribution framework for multimodal VQA (see the proposal for the
research framing). The 6-layer pipeline is fully built, locally tested (**115
passing hermetic tests**, zero GPU needed), and has been run for real on Kaggle's
free GPU tier across nine experiments with real CVQA images and real quantized
vision-language models.

**However — in Sept 2026 the project's central measurement was found to be
broken, which invalidates most of the numeric findings produced before that
point.** `gold_prob_mass`, the quantity every Shapley value is computed from, was
the *absolute* teacher-forced probability of the gold option's literal text.
Measured after the fact on two completed runs, its AUC for separating correct from
incorrect answers was **0.46 and 0.53 — a coin flip**. It was not measuring
correctness at all. Everything downstream was therefore computed on noise. The
measure has since been rebuilt (normalized option scoring), and a gated
re-validation is what the project is currently waiting on. See §9 for exactly which
findings survive and which do not, and §8 for the correction itself.

**Nothing has been committed to git — the working tree has zero commits and no
remote configured.** Before anyone else can "pull" this repo it must be committed
and pushed. That's still the first operational thing to resolve.

---

## 2. Repo map

```
PROPOSAL_CULPRIT_VQA.md     Research proposal — read for the "why" and formal design
README.md                   Short setup + structure overview
HANDOFF.md                  This file
pyproject.toml              Python project config (deps: numpy/scipy/scikit-learn/pillow + dev/kaggle extras)
secrets.env                 GITIGNORED — Kaggle credentials, does NOT exist after a git pull (see §4)

src/culprit_vqa/            The installable package — see §3 for the layer-by-layer map
tests/                      Mirrors src/, 115 passing tests, zero GPU/network needed (pytest -m "not integration")
scripts/                    Local, no-GPU utility/analysis scripts (see §3)

GATE_REPORT.md              WRITTEN BY scripts/gated_launch.py — the verdict on whether the
                            rebuilt measure works. READ THIS FIRST if it exists (see §8/§10).

kaggle_kernels/              Everything related to real Kaggle GPU runs — see §5/§6
  culprit_vqa_package/        Staging copy of src/culprit_vqa, zipped+uploaded as a public Kaggle Dataset
  kaggle_kernel_utils.py       Reusable local push/poll/pull helpers (not uploaded to Kaggle)
  watch_kernels.py             Robust multi-kernel status poller (survives API blips; use this,
                               not an ad-hoc loop — see §6)
  phase0_calibration/          Kernel 1: GPU/model calibration (see §7 experiment log)
  model_calibration/           Kernel 2: multi-model sanity check before the LLaVA run
  phase1_4_full_run/           Kernel 3: main 150-item attribution run (Qwen2.5-VL-3B)
  phase1_4_llava_run/          Kernel 4: same, second model (LLaVA-OneVision-0.5B)
  natural_failure_audit/       Kernel 5+7: real-mistake diagnosis. output/ = v1 (2 repairs),
                               output_v2/ = v2 (5 repairs + 2 placebos) — v2 supersedes v1
  model_calibration_7b/        T4 feasibility probe for the 7B model (memory/latency/GO-NO-GO)
  natural_failure_audit_7b/    Kernel 8: the v2 audit on LLaVA-OneVision-7B
  instrument_check/            Kernel 9: THE decisive run — old-vs-new measure AUC + manipulation
                               check on all 4 factors (see §8)
  attribution_v2/              The big attribution run. Its config lines are REWRITTEN by
                               scripts/gated_launch.py from the instrument check's verdict —
                               do not hand-edit the lines marked `# GATE:`
  each kernel dir has kernel-metadata.json + the .py script + output/ (downloaded real results)
```

---

## 3. The package (`src/culprit_vqa/`) — what's real vs. minimal

| Layer | Status | Key files |
|---|---|---|
| **0 — taxonomy** | Real, complete | `layer0_taxonomy/{axes,factors,operators,pillar_mapping}.py` — 5-axis factor space, all 9 proposal operators registered (only 4 have a *real* perturbation-application function yet — see §10) |
| **1 — intervention engine** | Real, complete | `layer1_intervention/{items,lattice}.py` (factorial lattice generation), `validity/` (3-way check), `real_operators.py` (4 real image/text perturbation functions), `repair_operators.py` (**5 hypothesis-mapped repairs + 2 placebo controls**, each a `Repair` dataclass carrying the taxonomy axis it probes; may edit question text, image, or both) |
| **2 — model runner** | Real (HF) + mock | `layer2_runner/mock_runner.py` (synthetic, deterministic), `layer2_runner/hf_runner.py` (**the real one** — `HFVisionLanguageRunner`, any `AutoModelForImageTextToText` model, 4-bit, GPU-only, lazy-imports torch/transformers). **Rewritten Sept 2026 for normalized option scoring — read its module docstring and §8 before changing anything about how `gold_prob_mass` is computed.** |
| **3a — causal ground truth** | Real, complete | `layer3a_causal/{probability,shapley,attribution}.py` — exact Shapley values, leave-one-out, pairwise interaction indices, null-safe normalized attribution |
| **3b — behavioral signals** | 4 real, 1 stub | `layer3b_signals/drift.py` (S1, real), `uptake.py` (S4, real), `stubs.py` (S2 real-minimal, **S3 still a constant-0.0 stub**, S5 now reads the option-distribution confidence rather than a generation proxy, S6 real via native-vs-English CVQA text) |
| **4 — amortized attributor** | Real-minimal | `layer4_attributor/` — plain logistic regression. Its rho ~= 0.09 is now explained: it was trained against Shapley values computed from a broken measure (§8). Untested on the fixed measure. |
| **5 — failure profiles** | Real-minimal | `layer5_profiles/{aggregate,interaction_atlas}.py` |
| **orchestration** | Real | `pipeline.py` (`run_pipeline_for_item` — wires layers 1→2→3a→3b for one item), `natural_failure.py` (`run_natural_failure_audit` — the RQ4-style loop; applies **every** repair to every failure and lets whichever one works define the label, rather than classifying first; records continuous p(gold) deltas alongside binary flips, and marks no-op repairs `applied=False` so they are not counted as failures) |
| **fixtures** | Real | `fixtures/synthetic_items.py` — 5 hand-built synthetic items used by the hermetic tests + `scripts/demo_pipeline.py` |
| **kaggle_connectivity.py** | Real | Credential loading + the auth-quirk fix (see §6) |

**⚠️ A REAL BUG WAS FOUND AND FIXED here — do not "fix" it back:** the efficiency
axiom for this φ sign convention (φ_m = p(S) − p(S∪{m})) is
**`Σφ_m = p(∅) − p(M)`**, NOT `p(M) − p(∅)`. This was derived by hand and verified;
every real run has `efficiency_residual` ≈ 1e-17. See
`layer3a_causal/attribution.py`'s `efficiency_residual` docstring for the full
derivation if this needs re-deriving for confidence.

**⚠️ But do NOT treat that near-zero residual as validation of anything
scientific.** Earlier versions of this file called it "the single most important
validation result". That was wrong, and it cost the project months of building on
sand. `Σφ = p(∅) − p(M)` is an **algebraic identity** of the Shapley formula: it
holds exactly for *any* `p` function, including one that returns random numbers.
It confirms the arithmetic is implemented correctly and nothing leaks — a
necessary bookkeeping check on the tool — and it says **nothing whatsoever** about
whether `p` measures anything real. The measure being garbage (see §8) is
precisely the failure mode a zero residual cannot detect. Validate `p` itself
(§8), separately and always.

### Local scripts (no GPU/network needed)
- `scripts/demo_pipeline.py` — full pipeline on synthetic fixtures (sanity check the codebase runs at all)
- `scripts/check_kaggle_connection.py` — Kaggle *dataset*-endpoint connectivity check (needs `secrets.env` + network)
- `scripts/analyze_kaggle_run.py <path-to-phase1_4-results.jsonl>` — reconstructs records from a real-run's JSON output and runs Layer 4/5 locally
- `scripts/analyze_natural_failure_audit.py <path-to-natural_failure-results.jsonl>` — summarizes a natural-failure-audit run, reporting every repair **against the placebo floor** (a raw flip rate is uninterpretable on its own) plus paired significance
- `scripts/gated_launch.py` — waits for the instrument check and launches the big attribution run **only** if its verdict justifies it; writes `GATE_REPORT.md` either way

Run the hermetic suite with: `pytest -m "not integration"` (115 tests, ~2s, no network/GPU).

---

## 4. Environment setup (do this first)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -m "not integration"        # should show 89 passed
```

### Kaggle credentials — `secrets.env` does NOT survive a git pull (gitignored)

Whoever resumes this needs their **own** `secrets.env` at the repo root, one line
per Kaggle account:
```
<kaggle-username> == "<kaggle-api-token>"
```
(Yes, `==` not `=` — that's the format this project settled on; see
`culprit_vqa/kaggle_connectivity.py`'s `_SECRETS_LINE_RE`.)

The three accounts used in the work so far (owned by the original project owner,
not shared in this repo for obvious reasons) each had: free tier, 2×T4 GPUs, ~30
GPU-hours/week quota. To continue the *exact same* experiments, get fresh tokens
from those accounts or substitute your own — the code doesn't care whose account it is.

**⚠️ IMPORTANT, non-obvious auth gotcha:** Kaggle's CLI (as of the version pinned in
`pyproject.toml`, `kaggle>=1.6` — actual version tested was 2.2.4/kagglesdk 0.1.37)
has **two different auth mechanisms** depending on the endpoint:
- Dataset endpoints (`dataset_list`, `dataset_download_files`, etc.) work with the
  legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars.
- **Kernel and competition endpoints require `KAGGLE_API_TOKEN`** — the legacy pair
  fails with a `401 Unauthorized` on those specifically, with a confusing generic
  error message that doesn't mention this.

`culprit_vqa.kaggle_connectivity.activate_kaggle_account(username, token)` sets
**all three env vars** at once for exactly this reason — always use it (or
`kaggle_kernels/kaggle_kernel_utils.py`'s helpers, which call it) rather than
setting env vars by hand.

**⚠️ Another gotcha: the secrets.env "username" is not necessarily the Kaggle account's
actual username/slug.** Two of the three accounts used had a *different* real Kaggle
username than the login label in secrets.env:
| secrets.env label | actual Kaggle username (used in kernel refs / URLs) |
|---|---|
| `NakibSaleh` | `nakibsaleh` (matches) |
| `chotonunulabib` | `gmnoorlabib` (**different!**) |
| `mediumnunulabib` | `rsdaiyan` (**different!**) |

Before constructing a `kernel-metadata.json` `"id"` field for a given credential,
**authenticate and run `kaggle kernels list --mine` (or check any existing dataset
ownership) to discover the real username** — don't assume it matches the secrets.env label.

---

## 5. How the Kaggle experiments actually work (read before running anything)

Every real experiment is a **Kaggle "script" kernel**, pushed via the CLI, that:
1. Bootstraps the `culprit_vqa` package from an **attached Kaggle Dataset**
   (`nakibsaleh/culprit-vqa-package`, currently **public**, so any of the three
   accounts — or anyone — can attach it), because a kernel push only uploads the
   one `code_file`, not the whole local repo.
2. `pip install`s the GPU-stack deps (`transformers`, `accelerate`, `bitsandbytes`,
   `qwen-vl-utils`, `datasets`) — NOT pre-baked into the dataset, installed fresh each run.
3. Pulls real images/questions live from the `afaji/cvqa` HuggingFace dataset
   (needs `enable_internet: true` in kernel-metadata, which is set).
4. Runs the real experiment, **checkpointing every item to a `.jsonl` file** in
   `/kaggle/working/` as it goes (so a mid-run crash loses at most one item).
5. Is pulled back locally via `kaggle kernels output <ref> -p <dir> -o`.

### The package-as-dataset workflow (do this every time `src/culprit_vqa/` changes)
```bash
# 1. Re-sync the staging copy from the real source of truth:
rm -rf kaggle_kernels/culprit_vqa_package/culprit_vqa
mkdir -p kaggle_kernels/culprit_vqa_package/culprit_vqa
cp -r src/culprit_vqa/* kaggle_kernels/culprit_vqa_package/culprit_vqa/
find kaggle_kernels/culprit_vqa_package -name "__pycache__" -exec rm -rf {} +

# 2. Push a new dataset version (creates a new version of the SAME dataset — don't `create` again):
cd kaggle_kernels/culprit_vqa_package
KAGGLE_API_TOKEN="<token>" kaggle datasets version -p . -r zip -m "<what changed>"
# then wait for `kaggle datasets status nakibsaleh/culprit-vqa-package` to say "ready"
```
**Known quirk:** `kaggle datasets create/version -r zip` on a directory zips its
*contents*, not the directory itself — so the mounted dataset at
`/kaggle/input/culprit-vqa-package/` does NOT contain a `culprit_vqa/` subfolder, it
contains `culprit_vqa`'s contents directly at its root (an `__init__.py`,
`layer0_taxonomy/`, etc., with no wrapper). Every kernel script therefore starts
with a bootstrap block that copies the mounted dataset into
`/kaggle/working/culprit_vqa` (recreating the proper package name) before adding
`/kaggle/working` to `sys.path` and importing. **Copy that exact bootstrap block
from any existing kernel script (e.g. the top ~50 lines of
`kaggle_kernels/phase1_4_full_run/full_run.py`) rather than re-deriving it** — it
also has a defensive fallback that searches `/kaggle/input` for the package if the
expected path is wrong, which saved real debugging time once already.

### Pushing/monitoring/pulling a kernel
```bash
cd kaggle_kernels/<some_experiment>
KAGGLE_API_TOKEN="<token>" kaggle kernels push -p .
# poll (NEVER foreground-sleep in a loop -- use run_in_background or a Monitor-style tool):
KAGGLE_API_TOKEN="<token>" kaggle kernels status <owner>/<kernel-slug>
# when status says COMPLETE:
KAGGLE_API_TOKEN="<token>" kaggle kernels output <owner>/<kernel-slug> -p output --file-pattern ".*\.jsonl" -o
```
**Gotchas:**
- The kernel's actual URL slug is derived from the **title**, which can silently
  differ from the `"id"` you wrote in `kernel-metadata.json` on first push (Kaggle
  warns but still creates it under the title-derived slug) — re-pushing with the
  original mismatched id then fails with a `409 Conflict`. Fix: after the first
  push, read the real URL from the push output and update `"id"` in
  `kernel-metadata.json` to match exactly before pushing again.
- **`"machine_shape": "NvidiaTeslaT4"` must be set explicitly** in
  `kernel-metadata.json`. Without it, Kaggle sometimes allocates a **P100** instead,
  which crashes real model inference with
  `CUDA error: no kernel image is available for execution on the device` — the
  installed PyTorch build doesn't include Pascal (P100) kernels. This is a
  known-real failure mode, already hit and fixed once.
- Kaggle CLI output pull sometimes throws a cosmetic
  `'charmap' codec can't encode characters` error on Windows (a console-encoding
  issue printing a non-ASCII character from the kernel log) — **the file still
  downloads correctly**, this is not a real failure, don't chase it.
- Two *independent* experiments should be run **in parallel on two different
  accounts** rather than sequentially on one — same total compute-hours, half the
  wall-clock wait. (This was a real, useful course-correction mid-project — see §7.)

---

## 6. Standing project conventions (please follow these)

- **The project owner wants a short, plain-English (no jargon), pre-flight
  explanation of what a run will test and what changed, given right before actually
  launching any real Kaggle GPU run** — not after. This applies every time, not just
  once. (Saved as a Claude memory on the owner's machine; restating it here since
  memory doesn't transfer with the repo.)
- **Never commit without being asked.** Everything in this project so far was built
  and run without a single git commit — the owner explicitly wants to review before
  committing.
- **Never put real Kaggle tokens in a committed file.** `secrets.env` is gitignored;
  keep it that way.
- Prefer reusing the tested `src/culprit_vqa` package inside kernel scripts (via the
  dataset-attachment trick) over re-deriving logic inline in a kernel script — this
  has paid off directly (e.g. the Shapley math has never needed re-fixing inside a
  kernel because it's the same tested code every time).
- When a real GPU run fails partway, **fix forward and re-run rather than trying to
  rescue partial output** — checkpointing means you only lose the last incomplete
  item, and every kernel script already writes incrementally for this reason.
- **Validate an instrument before building on it.** Zero errors, a clean
  end-to-end run and a perfect efficiency residual are all compatible with a
  measure that is pure noise (§8). For any new measure, check that it separates
  the cases it claims to distinguish (AUC vs correctness) *first*.
- **Every effect needs a control.** The project's "15% of failures fixed" result
  evaporated the moment placebos were added. If an intervention has no placebo,
  its effect size is not interpretable.
- **Pin dependency versions in every kernel.** Unpinned `pip install -U` is the
  likeliest cause of two identical runs disagreeing 81 vs 67 (§8).
- **Budget every run against the ~11-12h Kaggle session limit** before launching:
  `n_items * calls_per_item * seconds_per_call`, measured with a small feasibility
  probe, kept under ~10h. Better still, give the kernel a wall-clock budget and
  let it stop cleanly (see `attribution_v2/`), so an underestimate cannot lose the
  whole run.
- **Poll Kaggle with a timeout.** A status watcher once hung on a CLI call with no
  timeout and silently stopped polling, so a finished kernel went unnoticed for an
  hour. `kaggle_kernel_utils._run_kaggle` now has a 300s timeout; use
  `kaggle_kernels/watch_kernels.py`, which survives blips.

---

## 7. Experiment log — what's been run, and where the results live

All raw results are downloaded into the repo under `kaggle_kernels/*/output/` —
**you do not need Kaggle access to analyze existing results**, only to run new ones.

Runs 1-6 predate the measurement fix (§8). Their *engineering* outcomes are still
valid; their *numeric findings* are not. Runs 7-9 are the correction.

| # | Kernel dir | Purpose | Scale | Outcome |
|---|---|---|---|---|
| 1 | `phase0_calibration/` | First real-GPU contact; discover `afaji/cvqa` schema | 20 items | Fixed: P100-vs-T4 crash; CVQA has no `Country`/`Language` cols (it's `Subset`), options are a list field. ~8.6s/condition on a T4. |
| 2 | `phase1_4_full_run/` v1-v2 | First full-scale attribution run, Qwen2.5-VL-3B 4-bit | 100 items | 51/100 succeeded, 49 CUDA OOM (uncapped image resolution). Fixed: 768px cap, cache clearing, retry-at-half-res. |
| 3 | `phase1_4_full_run/` v3-v5 | Same at target scale | 150 items | 150/150, 0 errors. **Numbers invalidated by §8.** `output/phase1_4_results.jsonl` |
| 4 | `natural_failure_audit/` v1 | Diagnose the model's own natural mistakes | 220-item pool | 81 mistakes (37%), 0 errors. **Superseded by #7.** `output/natural_failure_results.jsonl` |
| 5 | `model_calibration/` | Regression-check the genericized runner + first contact with a 2nd model | 10 items x 2 models | Both loaded, 0 errors. Found LLaVA answers with a bare letter ("D"); at the time this was patched with a string-matching heuristic (`_matches_gold`, now **deleted** — see §8). |
| 6 | `phase1_4_llava_run/` | Run #3 on LLaVA-OneVision-0.5B, in parallel on a 2nd account | 150 items | 150/150, 0 errors. **Numbers invalidated by §8.** `output/phase1_4_llava_results.jsonl` |
| 7 | `natural_failure_audit/` v2 | Natural-failure audit rebuilt: 5 hypothesis-mapped repairs + **2 placebo controls** + continuous confidence deltas | 220-item pool | 67 mistakes, 0 errors. **Produced a real, still-valid methodological finding** — see §9 finding 2. `output_v2/natural_failure_results.jsonl` |
| 8 | `natural_failure_audit_7b/` | Run #7 on LLaVA-OneVision-**7B** (upgraded from 0.5B) | 220-item pool | 77 mistakes, 0 errors. Cross-model transfer test. See §9 finding 3. `output/natural_failure_results_7b.jsonl` |
| 9 | `instrument_check/` | **The run that found the broken measure and re-validates the fix.** Head-to-head AUC of old vs new measures + a manipulation check on all 4 factors | 200 items | Launched; see `GATE_REPORT.md` for its verdict. |
| — | `model_calibration_7b/` | T4 feasibility probe before committing to the 7B run | 10 items | 10.3/15.6 GB peak, 10.4s median call, 10/10 clean → GO. `output/model_calibration_results.json` |

**Pending / gated:** `attribution_v2/` + `scripts/gated_launch.py`. The launcher
waits for run #9 and launches the big attribution run on two accounts **only if** a
new measure clears AUC >= 0.80 **and** >= 2 factors significantly break the model.
On failure it launches nothing and writes `GATE_REPORT.md`. **Read `GATE_REPORT.md`
before doing anything else** — it says whether the foundation held.

---

## 8. THE MEASUREMENT BUG — read this before trusting any number below

**What was wrong.** `RunResult.gold_prob_mass` was the geometric-mean per-token
probability of the gold option's *literal text*, in absolute terms. Three fatal
defects:

1. **Not normalized against the alternatives.** The meaningful question is "does
   the model prefer gold over the other options"; we asked "what absolute
   probability does this one string get", which is dominated by string length,
   token frequency and tokenization.
2. **Not comparable across items** — options with different surface texts get
   systematically different scores regardless of correctness.
3. **Wrong target entirely for letter-answering models.** LLaVA-OneVision emitted a
   bare letter on **100%** of items while we scored the probability of the full
   option text. The two measures were not even about the same event.

**How it was caught.** Ask of any measure: does it separate the cases it claims to
distinguish? Computed on existing run data, comparing `gold_prob_mass` when the
answer was right vs wrong:

| Model | mean p(gold) when correct | when wrong | **AUC** |
|---|---|---|---|
| Qwen2.5-VL-3B | 0.0859 | 0.0686 | **0.459** |
| LLaVA-OneVision-7B | 0.0420 | 0.0366 | **0.528** |

AUC 0.5 is a coin flip. The instrument could not tell a right answer from a wrong
one. **This single fact explains** the attributor's rho ~= 0.09, the +/-0.002 repair
deltas, and the cross-model non-replication — all of it was computed on noise.

**The fix (in `layer2_runner/hf_runner.py`).** Standard lm-eval/MMLU scoring: score
every option, softmax across them, read off p(gold). Bounded, comparable across
items and models, with a meaningful chance baseline (1/k). Two modes:
`scoring="letter"` (default; reads the k letter-token logits from **one** forward
pass — no length bias, cheaper than the generation it replaces) and
`scoring="text"` (teacher-forces each option's full text, length-normalized; k
forward passes, no letter-mapping assumption). Correctness is now `argmax == gold`,
which is why **`_matches_gold` was deleted** — the string-matching heuristic only
ever existed because correctness was read off generated text.

**Lesson to carry forward:** validate the instrument before building on it. A
plausible-looking pipeline that runs cleanly end-to-end, with 0 errors and a perfect
efficiency residual, told us nothing — because none of those checks touch whether
`p` measures the thing it claims to.

---

## 9. Findings — which survive the measurement fix, and which don't

### INVALIDATED (computed from the broken measure — do not cite)

1. **"The causal math is validated on real data (residual ~1e-17)."** The residual
   is an algebraic identity that holds for random numbers. It validates the
   arithmetic, not the science. See the warning box in §3.
2. **"The two models show opposite vulnerability patterns"** (the alpha table: Qwen
   fragile to visual corruption, LLaVA-0.5B fragile to irrelevant text). Those
   values are normalized Shapley values computed from a measure with AUC ~= 0.5.
   The pattern may or may not be real; **as of now it is unsupported.**
   Re-derivable from `attribution_v2` output once the gate passes.
3. **"Chain-of-thought fixes 12% of natural failures, topic hint 5%."** Killed by
   its own control — see finding 2 below.

### STILL VALID

1. **The engineering works.** 150/150 and 220/220 real-GPU runs with zero errors,
   full 39/39 CVQA subset coverage at n=150, multi-account parallel execution,
   package-as-dataset deployment, checkpointed recovery. None of this depends on
   the measure.

2. **Placebo controls killed our own headline result — the most valuable finding
   the project has produced.** Run #7 added two *placebo* repairs: extra text of
   comparable length carrying no usable information. On Qwen-3B:

   | Repair | Fixed | vs. placebo floor | Real? |
   |---|---|---|---|
   | Supply cultural context | 16.4% | **+10.4 pts** | **Yes** (p=0.021, p=0.039 vs both placebos) |
   | Ask in native language | 10.4% | +4.5 pts | No (p=0.22) |
   | Bigger/sharper image | 10.4% | +4.5 pts | No (p=0.22) |
   | Topic hint | 6.0% | **+0.0 pts** | **No** (p=1.00) |
   | Chain-of-thought | 6.0% | **+0.0 pts** | **No** (p=1.00) |

   Meaningless text flipped **6%** of answers by itself. The two repairs the
   earlier "15% fixed" claim rested on sit *exactly* on that floor. Placebo-
   corrected coverage: 30% "fixed by something" → **16% fixed by something that
   beat noise**. This finding is about *flip rates*, which come from argmax-style
   correctness rather than from `gold_prob_mass` — which is why it survives §8
   while the continuous deltas from the same run do not.

3. **Diagnoses did NOT transfer across models — the decisive negative result.**
   Run #8 repeated #7 on LLaVA-7B. Of the **44 items both models got wrong**, the
   cultural-context repair fixed: **both models 2**, Qwen only 3, LLaVA only 5,
   neither 34. If "this failure is a culture problem" were a property of the
   *item*, the same items would respond in both models. They don't. **Right now
   failure causes look model-specific, not item-specific.** That is publishable,
   but it is a much weaker claim than the project was aiming for, and it should
   change how the thesis frames the metric.

### KNOWN METHODOLOGICAL FLAWS (fix before re-running the audit)

- **The two placebos are not equivalent.** On LLaVA-7B the prefix placebo flipped
  14.3% and the suffix placebo 2.6% — a 5x gap between two supposedly
  information-free edits. The prefix one says *"Exactly one option is intended to
  be correct"*, which is a genuine hint about task structure. Rewrite both to be
  truly neutral and add a third, so the floor is *estimated* rather than taken from
  one text.
- **There is no visual placebo.** Both placebos are text-only, so nothing controls
  for "the image changed at all". The image repair's 7 flips came with a *negative*
  mean probability change (-0.021), which is what noise looks like. Needed:
  upscale-then-downscale, so pixel count changes but no detail is added.
- **We are underpowered by ~5x.** 220 items → ~70 failures → ~10 discordant pairs.
  Detecting a 10-point difference needs ~50. CVQA has ~10k items and the quota
  allows thousands — 220 was arbitrary and never justified.
- **Unpinned dependencies broke reproducibility.** Two identical audit runs
  disagreed on which items the model got wrong (**81 vs 67**, the 67 a strict
  subset). The likeliest cause is `pip install -U transformers` picking up a new
  version mid-project. All new kernels now pin exact versions; older ones do not.
  Kernel logs also came back 0 bytes, so this could not be confirmed from the logs
  — worth fixing.

---

## 10. What's NOT done yet — prioritized

**0. Read `GATE_REPORT.md`.** It contains the verdict on whether the rebuilt
measure works and whether the injected factors actually break the model. If the
gate failed, *stop and understand why* rather than running more experiments — that
is the mistake this project already made once.

1. **If the gate passed:** analyze the `attribution_v2` output. The real question,
   finally askable: with a working measure, does the Shapley attribution recover
   causes we *planted*? If it can't do that, the method is dead and it is better to
   know now.
2. **If the gate failed on AUC:** the problem is deeper than normalization.
   Investigate before spending more GPU time.
3. **If the gate failed on factors:** the interventions are too weak to break the
   model. Make them stronger (replace the image outright, contradict the answer
   explicitly) — a factor that does not move the model cannot be a cause.
4. **Fix the placebo design and add a visual placebo**, then re-run the
   natural-failure audit at ~1000 items rather than 220.
5. **Re-derive the cross-model alpha comparison** (invalidated finding 2) on the
   fixed measure — it was the project's most interesting claim and deserves an
   honest re-test.
6. **Consider the reframe.** Given finding 3 (causes are model-specific), the
   stronger and more defensible thesis may be *causally decomposing what "cultural
   difficulty" in a multicultural benchmark actually consists of* — how much is
   perception, how much cultural knowledge, how much language — per model. CVQA
   *asserts* cultural specificity; nobody has causally tested it. That contribution
   does not require diagnoses to transfer, and the existing interventions suit it.
7. **Build the remaining 5 of 9 proposal operators** as real perturbations. Only
   `irrelevant_plausible_fact`, `wrong_local_entity`, `text_overlay_wrong_answer`,
   `salience_recomposition` exist. Missing: `diffusion_irrelevant_object`,
   `contradictory_caption`, `western_default_substitution`, `biased_prior_phrasing`,
   `long_tail_entity_swap`.
8. **Wire up real S3 (NLI conflict)** — still a constant-0.0 stub.
9. **Get this into git with a remote.**

---

## 11. If you get confused about "what's the deliverable here"

This is not a normal "model accuracy went up" ML project. The intended deliverable
is a **new way of assigning mathematically-exact blame percentages for a specific
failure among several candidate causes**, plus evidence that it reveals patterns a
single-model accuracy study would miss. See the one-paragraph pitch at the top of
`PROPOSAL_CULPRIT_VQA.md`, and don't reduce it back to "did accuracy improve" when
explaining it to anyone.

**But be honest about where that stands.** As of the last session:

- The blame-splitting *arithmetic* is correct and tested. Whether it is blaming
  anything real is **unproven** — the measure it consumed was noise (§8), and the
  re-validation is what `GATE_REPORT.md` decides.
- The most solid empirical result the project currently owns is a **negative** one:
  causes of failure appear model-specific rather than item-specific (§9 finding 3),
  and two of the repairs previously reported as working are indistinguishable from
  meaningless text (§9 finding 2).
- Those are real, reportable contributions — "we built controls and they killed our
  own headline" is good science and should be written up as such. Just do not let
  the write-up claim the metric has been validated against ground truth until an
  experiment actually shows that.

If you are tempted to run another experiment before reading `GATE_REPORT.md`:
don't. The project already spent several rounds tuning repairs, placebos and a
second model on top of a broken ruler.
