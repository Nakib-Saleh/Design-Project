# CULPRIT-VQA: Causal Attribution of Multimodal Failure Modes in Multilingual Cultural VQA

**Proposal v2 — 12 September 2026**
*(v1 revised after CVPR-style review; changelog in §14)*

---

## 1. One-paragraph pitch

Vision–language models (VLMs) fail on multicultural, multilingual VQA — but current evaluation only reports *that* they fail, not *why*. **CULPRIT-VQA** is a **causal failure-attribution framework**: it (1) describes failure sources in a five-axis factor space (modality × relevance × conflict × culturality × knowledge), (2) applies **factorial controlled interventions** on a culturally diverse multilingual VQA substrate (CVQA-style, 26+ languages), (3) computes **intervention-based causal attributions** (Shapley values over the perturbation lattice) as ground truth for *which factor caused each failure*, (4) trains an **amortized attributor** that approximates these causal attributions from single-pass behavioral signals, and (5) validates the whole pipeline on **natural (non-injected) failures** via constructive repair interventions and human diagnosis. Cultural bias is the studied *phenomenon*; **causal "why" attribution is the contribution**. The headline output: per-sample, per-language, per-model failure profiles with a defined causal semantics — "removing the cultural-prior factor restores correctness; causal contribution 0.61."

---

## 2. Problem statement

1. **Accuracy-only evaluation is uninformative.** CVQA, CultureVerse, and MMAC show large accuracy gaps on non-Western content, but a gap conflates ≥5 causes: language handling, visual recognition, missing cultural knowledge, Western-prior override, cross-modal conflict.
2. **Distraction/robustness research is fragmented and correlational.** Visual distractors (Distract-Bench, Idis), textual distraction (GSM-IC, DIM-Bench, DistractionIF), cross-modal conflict (CLASH, CrossCheck-Bench, VisualTextTrap), and cultural bias (MMAC, CulturalGround) each measure *accuracy drop under one perturbation family*. None attributes individual failures to causes, and none handles **interacting** failure sources.
3. **Attribution must be causal, not correlational.** A classifier trained on injected-perturbation labels risks learning *what perturbations look like* rather than *what caused the failure*. CULPRIT-VQA's central design decision: **interventions define ground truth; learned models only amortize it.**

**Scope exclusion:** input-uncertainty failures (blur, low quality, ambiguity, OOD corruption) are explicitly out of scope — they are covered by the corruption-robustness literature (VLM-RobustBench et al.) and involve degraded perception rather than mis-selection of evidence.

---

## 3. Research questions and hypotheses

| # | Research question | Hypothesis / success criterion |
|---|---|---|
| RQ1 | Can intervention-based causal attribution disentangle failure factors, **including interacting ones**? | Shapley attributions over the factorial lattice separate single-factor causes; interaction indices detect compositional failures (e.g., cultural prior × text overlay) that leave-one-out analysis misattributes. |
| RQ2 | Do causal failure profiles differ systematically across languages/cultures? | Low-resource pairs fail predominantly via culturality/knowledge factors; high-resource pairs via relevance/conflict factors. |
| RQ3 | Can a single-pass amortized attributor approximate intervention ground truth? | Attributor's α̂ correlates with Shapley ground truth (target Spearman ρ ≥ 0.7 held-out; report honestly if lower — the gap itself is a finding about behavioral-signal sufficiency). |
| RQ4 | Does attribution transfer to *natural* failures? | On unperturbed CVQA errors, constructive repair interventions + human diagnosis agree with CULPRIT attributions above chance and above an LLM-judge baseline (triple-agreement protocol, §8.3). |

RQ1 (causal disentanglement under interaction) and RQ4 (external validity) are the two load-bearing experiments.

---

## 4. Failure-factor taxonomy (Layer 0)

### 4.1 Five-axis factor space (the formal object)

A perturbation or natural failure hypothesis is a point in:

| Axis | Values | Meaning |
|---|---|---|
| **Modality** | visual / textual / cross-modal | where the interfering signal lives |
| **Relevance** | relevant / irrelevant | whether the signal bears on the answer |
| **Conflict** | consistent / contradictory | relation to the decisive visual evidence |
| **Culturality** | neutral / culturally loaded | whether cultural priors are engaged |
| **Knowledge** | perceptual / knowledge-required | whether long-tail knowledge is needed |

An observed failure additionally gets an **effect label** from trace analysis: *ignored / adopted / dominated* (distractor uptake level, following DRR/HFR).

This factorization resolves the v1 taxonomy criticisms: "cultural substitution" is no longer forced to be a distinct leaf that secretly duplicates cross-modal conflict — it is simply `textual × contradictory × culturally-loaded`. Factors compose; leaves don't.

### 4.2 The four-pillar tree (presentation view only)

The original tree (Visual / Textual / Cross-modal / Cultural, 16 leaves) is retained **as named regions of factor space** for communication and for mapping to prior benchmarks — e.g. *"salience manipulation"* = `visual × irrelevant × consistent × neutral`. A full leaf→factor mapping table ships with the taxonomy chapter. Claims and measurements are made at the factor level.

---

## 5. Positioning (Sep 2026)

| Area | Representative work | They measure | We add |
|---|---|---|---|
| Visual distraction | [Distract-Bench](https://arxiv.org/abs/2606.08894), [Idis](https://arxiv.org/abs/2511.21397) | accuracy + uptake under one factor | multi-factor causal attribution |
| Textual distraction | GSM-IC, [DIM-Bench](https://aclanthology.org/2025.acl-long.957/), [DistractionIF](https://arxiv.org/abs/2605.29491) | accuracy drop, text-only | multimodal, multilingual, interactions |
| Cross-modal conflict | [CLASH](https://github.com/tpopordanoska/clash), [CrossCheck-Bench](https://arxiv.org/abs/2511.21717), [VisualTextTrap](https://arxiv.org/html/2604.17375) | conflict detection accuracy | conflict as one axis in a causal design |
| Conflict internals | [knowledge-conflict probing](https://arxiv.org/html/2602.14518v1) | decodability of conflict states | probes as auxiliary signal, causally validated |
| Cultural VQA | [CVQA](https://cvqa-benchmark.org/), [CulturalGround](https://aclanthology.org/2025.emnlp-main.1232.pdf), [MMAC](https://aclanthology.org/2026.acl-long.989.pdf) | *that* models fail culturally | typed causal *why*, per sample |
| Reliability | [VLM-DeflectionBench](https://aclanthology.org/2026.acl-long.1307.pdf) | abstention behavior | attribution of non-abstained failures |

**Novelty claim (revised):** not "we unify benchmarks," but — *a unified **causal** framework for attributing multimodal VQA failures under controlled factorial interventions, with amortized single-pass approximation and validation on natural multilingual failures.* The benchmark is the instrument, not the contribution.

---

## 6. Framework design

### 6.0 Architecture

```
 LAYER 0 · FACTOR TAXONOMY (5 axes; 4-pillar tree as view)
      │ defines operators, validity checks, signals
      ▼
 LAYER 1 · FACTORIAL INTERVENTION ENGINE
      substrate: CVQA ~2k items, ≥20 country–language pairs
      per item: k ≤ 3 factors → full 2^k perturbation lattice
      validity: semantic / evidence / prior three-way check (§6.2)
      │ lattice of conditions per item
      ▼
 LAYER 2 · MODEL RUNNER
      4 open VLMs (+CulturalPangea) w/ traces & hidden states;
      1–2 closed models behavioral-only
      │ answers, CoT traces, logits, states
      ▼
 LAYER 3a · CAUSAL GROUND TRUTH          LAYER 3b · BEHAVIORAL SIGNALS
      Shapley attribution φ_m over            S1 drift · S2 attr-ratio ·
      the 2^k lattice; interaction            S3 NLI conflict · S4 uptake ·
      indices for factor pairs                S6 language delta · (S5 probe, aux)
      │ ground-truth α(x)                     │ signal vector s(x)
      └──────────────┬───────────────────────┘
                     ▼
 LAYER 4 · AMORTIZED ATTRIBUTOR
      learns s(x) → α̂(x); validated against Layer 3a on held-out items
      │
      ▼
 LAYER 5 · FAILURE PROFILES + NATURAL-FAILURE AUDIT
      per language · culture · model · factor;
      constructive repair interventions on natural errors (§8.3)
```

### 6.1 Substrate and factorial design (Layer 1)

- **Substrate:** CVQA (10k items, 30 countries, 31 languages, local+English parallel text); augmented with CulturalGround long-tail entities. Working set **~2,000 base items**, stratified over ≥20 country–language pairs.
- **Factorial lattice:** each item is assigned k ≤ 3 perturbation factors drawn from the operator library (below). All 2^k combinations are generated (≤8 conditions/item → ≤16k inference calls per model). This is the design that makes interaction measurable (RQ1) and Shapley attribution exact.

**Operator library** (each operator = one factor instantiation, tagged with its 5-axis coordinates):

| Operator | Axis coordinates |
|---|---|
| Diffusion-inserted salient irrelevant object (Idis recipe) | visual · irrelevant · consistent · neutral |
| Salience re-composition (distractor enlarged/centered) | visual · irrelevant · consistent · neutral |
| Irrelevant plausible fact prepended (GSM-IC style) | textual · irrelevant · consistent · neutral |
| Wrong same-category local entity mentioned | textual · relevant · contradictory · culturally-loaded |
| Contradictory caption (CLASH recipe) | cross-modal · relevant · contradictory · neutral |
| Text overlay asserting wrong answer (VisualTextTrap recipe) | cross-modal · relevant · contradictory · neutral |
| Western-default substitution in context ("this is pizza") | textual · relevant · contradictory · culturally-loaded |
| Biased-prior phrasing, local vs English parallel | textual · relevant · consistent · culturally-loaded |
| Long-tail entity swap (CulturalGround) | visual · relevant · consistent · culturally-loaded · knowledge-required |

### 6.2 Perturbation validity: three-way check (upgraded from v1)

Every generated condition must pass:

1. **Semantic preservation** — ground-truth answer unchanged (judge model + native-speaker human audit on 10% stratified sample).
2. **Evidence preservation** — the decisive visual evidence region is untouched (IoU check between perturbation mask and annotated evidence region; text perturbations must not paraphrase away needed information).
3. **Prior-shift measurement** — the change in the *text-only* answer prior induced by the perturbation, measured on the language backbone (P(answer | question+context, no image), perturbed vs clean). Prior shift is **not** grounds for rejection — it is *recorded as a covariate*, because "semantically irrelevant but statistically informative" perturbations are precisely one of the phenomena under study. Attributions are reported with and without prior-shift stratification.

### 6.3 Model runner (Layer 2)

- **Open (traces + hidden states):** Qwen2.5-VL-7B, Qwen2.5-VL-72B, InternVL2.5, LLaVA-OneVision, + CulturalPangea (culture-tuned comparison point).
- **Closed (behavioral only):** 1–2 API models (GPT-5-class / Gemini-class) — used to test the behavioral-signal-only attributor variant, not for probe experiments.
- Per (item, condition): sampled answers (n=8 decodes for probabilistic correctness), full CoT trace, answer logits, mid-layer hidden states (open models).

### 6.4 Causal ground truth (Layer 3a) — the centerpiece

Let `M` be the factor set injected for item x (|M| = k ≤ 3), and for any subset `S ⊆ M` let

```
p(S) = P(correct answer | apply exactly the factors in S)
```

estimated from n sampled decodes (or ground-truth-answer logit mass).

**Causal contribution of factor m** = its Shapley value over the lattice:

```
φ_m(x) = Σ_{S ⊆ M\{m}}  [ |S|!(k−|S|−1)! / k! ] · ( p(S) − p(S ∪ {m}) )
```

(φ_m > 0 ⇔ factor m harms correctness, averaged over all contexts of other factors.)

**Attribution vector** (formal answer to "what does α = 0.6 mean"):

```
α_m(x) = max(0, φ_m(x)) / Σ_j max(0, φ_j(x))
```

α_m(x) is *the normalized causal contribution of factor m to the correctness drop, under the intervention distribution of the lattice*. Nothing else.

**Interaction index:** pairwise Shapley interaction values quantify compositional effects (visual × cultural, etc.). Leave-one-out deltas (`p(M\{m}) − p(M)`) are reported as a baseline attribution method that the factorial design is expected to beat under interaction — itself an experimental result.

**Why Shapley, not leave-one-out:** with interacting factors (the compositional setting of RQ1), leave-one-out double-counts or hides shared effects; Shapley is the unique attribution satisfying efficiency/symmetry/null-player over the lattice we already paid to compute.

### 6.5 Behavioral signals (Layer 3b) — explicitly non-causal

| Signal | Definition | Status |
|---|---|---|
| S1 drift | 1 − cos(E(R_clean), E(R_pert)) + ROUGE-L delta (I-ScienceQA) | behavioral |
| S2 attribute ratio | fraction of trace attributes about the distractor (Idis) | behavioral |
| S3 NLI conflict | contradiction probability, claim vs image-grounded description | behavioral |
| S4 uptake DRR/HFR | trace references distractor / references-and-fails (Distract-Bench) | behavioral |
| S6 language delta | local-vs-English accuracy/drift difference (MMAC-style) | behavioral |
| S5 conflict probe | linear probe on mid-layer states | **auxiliary diagnostic only** — decodability ≠ utilization; reported separately, never used as causal evidence |

Traces are treated as *behavioral evidence about what the model expressed*, never as ground truth about what caused the answer. All causal claims route through Layer 3a interventions.

### 6.6 Amortized attributor (Layer 4)

- **Task:** predict α̂(x) from s(x) (single clean+perturbed pass) — an *amortization* of the expensive 2^k lattice, not an independent source of truth.
- **Training target:** Layer 3a Shapley attributions (not injected-mode labels — this is the fix for the "perturbation recognition" critique; the label is *measured causal effect*, so a perturbation that was injected but caused nothing gets weight ≈ 0).
- **Models:** logistic/GBT (interpretable coefficients) + small MLP ablation.
- **Validation:** held-out items *and* held-out factor combinations; correlation with ground-truth α; calibration (ECE).

### 6.7 Failure profiles (Layer 5)

- Per country–language pair: causal-factor mixture histograms (RQ2).
- Per model: which factor families dominate; culture-tuned (CulturalPangea) vs general models.
- Interaction atlas: which factor pairs super-additively break which models (RQ1).

---

## 7. Natural-failure audit (RQ4) — promoted to headline experiment

Natural CVQA errors have no injected perturbation to remove, so attribution is validated **constructively**:

**Repair operators** (inverse interventions), one per hypothesized factor:

| Hypothesized cause | Repair |
|---|---|
| visual irrelevant object | inpaint suspected distractor out |
| textual distraction | neutralize/remove suspect context sentence |
| cross-modal conflict | correct or remove the conflicting caption/overlay |
| cultural prior (language) | translate question to English / to local language |
| knowledge gap | supply the relevant cultural fact in context (RAG-style) |

If repair r flips the answer to correct, factor(r) has causal evidence for this failure.

**Triple-agreement protocol** (~300 natural failures, 3 raters each, native speakers for culturally loaded items):

1. **Human ↔ Human**: inter-rater agreement on diagnosed cause (target κ ≥ 0.6);
2. **Human ↔ CULPRIT**: does the attributor match human diagnosis;
3. **CULPRIT ↔ Intervention**: does the attributor match repair-based causal evidence.

Reporting all three separates "humans are consistent" from "humans are right" from "the model is right" — no single agreement is treated as sufficient.

---

## 8. Evaluation plan

**Attribution quality (synthetic, ground truth available):** correlation of α̂ with Shapley α (per factor and aggregate), top-1 causal-factor accuracy, calibration, robustness under held-out factor combinations.

**Baselines:**
1. Accuracy-drop-only analysis (status quo);
2. Leave-one-out attribution (shows factorial/Shapley is needed under interaction);
3. Single-signal attributors (S1-only, S4-only);
4. Zero-shot LLM-judge cause labeling;
5. Classifier trained on *injected-mode labels* (the v1 design) — expected to inflate on synthetic data and degrade on natural failures; this ablation directly demonstrates the perturbation-recognition failure mode the review identified.

**Ablations:** drop each signal; behavioral-only vs +probes; English-only vs multilingual; pillar-level (4-way) vs factor-level attribution; prior-shift stratification.

**Compositional stress test (RQ1 killer experiment):** single factors vs all pairs vs triples; measure whether attribution correctly splits credit when factors interact (ground truth = pairwise Shapley interactions).

---

## 9. Deliverables

1. **CULPRIT-Bench** — ~2k clean items × factorial lattices (~12–16k conditions) with 5-axis factor tags, validity covariates (incl. prior shift), and Shapley ground-truth attributions for 5 open models.
2. **Attribution pipeline** — intervention engine + quantifier bank + amortized attributor (open source).
3. **Failure atlas** — causal factor profiles per language, culture, model; interaction atlas.
4. **Natural-failure audit** — repair-validated, human-validated attribution on real CVQA errors.
5. **Thesis/paper** — factor taxonomy, causal framework, results.

Dataset size is not claimed as a contribution; the controlled factorial design and causal validity are.

---

## 10. Timeline (9 months)

| Months | Phase | Milestone |
|---|---|---|
| 1–2 | Factor taxonomy formalization; substrate curation (2k items, ≥20 pairs); operator library spec | Frozen taxonomy + data card |
| 2–4 | Intervention engine + three-way validity protocol; generate lattices | CULPRIT-Bench v0.5 |
| 4–5 | Model runner (4 open + CulturalPangea); collect answers/traces/states | Raw run dataset |
| 5–6 | Layer 3a Shapley ground truth + interaction indices; Layer 3b signals | Causal ground-truth release |
| 6–7 | Amortized attributor + all baselines/ablations; closed-model behavioral runs | RQ1/RQ3 results |
| 7–8 | Natural-failure audit: repairs + triple-agreement human study | RQ4 results |
| 8–9 | Failure atlas, writing, artifact packaging | Thesis draft + release |

**Cut-line:** if compute/time binds — drop closed models, reduce to k ≤ 2 factors (lattice ≤ 4), keep RQ1 pairs only. The causal framework claim survives all cuts; RQ4 is protected last.

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Factorial lattice cost explodes | Medium | k ≤ 3 cap; 2k base items; 7B models first; n=8 decodes only where correctness is borderline |
| Injected factor causes nothing (φ ≈ 0 everywhere) | Medium | That is a *finding* (model robust to that factor); operator library iterated in month 3 pilot |
| Perturbations shift priors rather than distract | Certain (by design) | Prior shift measured as covariate, stratified analyses; phenomenon studied, not hidden |
| Factors not separable even causally | Low–Medium | Shapley handles shared credit honestly; fallback to pillar-level attribution remains novel |
| Native-reviewer recruitment for cultural validity | High | Full instrumentation only for languages with recruitable reviewers; rest transfer-eval |
| Natural-failure repairs are themselves confounded (inpainting artifacts etc.) | Medium | Repair-validity check mirrors §6.2; placebo repairs (edit non-suspect region) as control |
| CVQA contamination in newer models | Medium | Injected/perturbed content is novel; contamination check reported; hidden-test protocol where available |

---

## 12. Why this version is defensible

1. **The central claim is now provable by construction.** Attribution ground truth comes from interventions the framework itself performs; the learned component only amortizes it and is scored against it.
2. **α has one formal meaning** — normalized Shapley causal contribution over the intervention lattice — closing the "what does 0.6 mean" attack.
3. **Interactions are measured, not assumed away.** The factorial design + interaction indices turn the taxonomy-overlap criticism into the killer experiment.
4. **Behavioral vs causal evidence is never conflated.** Traces, similarity metrics, and probes are labeled behavioral/auxiliary; every causal statement routes through do-operations.
5. **External validity is a first-class experiment** (RQ4 with constructive repairs + triple agreement), not an afterthought.
6. **Graceful degradation** at every cut-line; each fallback (pillar-level, k=2, open-only) is still a publishable causal-attribution result.

---

## 13. Headline result template

> *CULPRIT-VQA attributes multimodal failures to causal factors with X top-1 agreement against intervention ground truth (vs Y for leave-one-out and Z for an injected-label classifier), maintains W agreement with repair-validated causes on natural multilingual failures, and exposes cultural-prior × conflict interactions that accuracy-only evaluation cannot see.*

---

## 14. Changelog v1 → v2 (review response)

| Review issue | Disposition |
|---|---|
| Attributor may learn perturbation signatures, not causes | **Accepted.** Interventional Shapley attribution is now ground truth; classifier demoted to amortization; v1 design kept only as an ablation baseline |
| Cultural leaves aren't distractions; leaves overlap | **Accepted.** 5-axis factor space replaces the leaf tree as the formal object; tree kept as presentation view |
| α ∈ Δ¹⁶ underdefined | **Accepted.** α defined as normalized Shapley causal contribution |
| Suggested `C_m = P(do(−m)) − P(do(all))` | **Adopted in spirit, upgraded**: leave-one-out is biased under interaction; Shapley over the factorial lattice used instead, leave-one-out kept as baseline |
| Traces/probes aren't causal evidence | **Accepted.** Behavioral vs causal split; S5 auxiliary only |
| Answer-preservation insufficient (semantic vs evidence vs prior) | **Accepted.** Three-way validity protocol; prior shift measured as covariate |
| RQ4 should be central; triple agreement | **Accepted.** Constructive repair operators specified (review left this unspecified — natural failures have no perturbation to remove); Human↔Human, Human↔CULPRIT, CULPRIT↔Intervention all reported |
| Compositional failures as killer experiment | **Accepted.** Factorial design makes it native (RQ1) |
| Model list overambitious | **Partially accepted.** Trimmed; CulturalPangea retained (cheap, essential for cultural axis) |
| Add "Input Uncertainty" failure family | **Rejected.** Scope creep into corruption-robustness literature; explicitly out of scope (§2) |
| Dataset size as contribution | **Accepted.** Reframed |
| "Unification" novelty weak | **Accepted.** Claim rewritten as causal-attribution framework |
