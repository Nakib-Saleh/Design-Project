# CULPRIT-VQA — Literature Review

A year-sorted survey of the work CULPRIT-VQA (see [PROPOSAL_CULPRIT_VQA.md](PROPOSAL_CULPRIT_VQA.md)) builds
on and positions against: the VQA task itself, language-prior/shortcut bias, the vision-language
model lineage used as the Layer-2 runner, visual/textual distraction and cross-modal conflict,
cultural/multilingual VQA, and causal attribution / Shapley-value explainability.

**Verification standard:** every entry below was confirmed via live web search (arXiv, ACL Anthology,
CVF/openaccess, NeurIPS/ICML proceedings, or the official page) during compilation on 2026-09-13 —
titles, authors, venues, and links are not drawn from memory alone. Where a discrepancy was found
between how this project's own proposal describes a paper and what the paper actually contains,
it is flagged in **[Citation caveats](#citation-caveats)** at the end rather than silently corrected.

---

## Pre-2014 foundations (causal attribution / game theory)

These predate the VQA task itself but are the mathematical basis of CULPRIT-VQA's Layer 3a.

- **1953** — [A Value for n-Person Games](https://www.rand.org/pubs/papers/P295.html) — Lloyd S. Shapley (in *Contributions to the Theory of Games II*, Princeton University Press) — the Shapley value itself: the unique attribution rule satisfying efficiency/symmetry/null-player, which CULPRIT-VQA's φ_m is a direct instantiation of.
- **1999** — [An axiomatic approach to the concept of interaction among players in cooperative games](https://link.springer.com/article/10.1007/s001820050125) — Grabisch & Roubens (*International Journal of Game Theory*, 28) — the Shapley interaction index, the formal basis for CULPRIT-VQA's pairwise factor-interaction measurement (RQ1).

---

## 2014

- **2014** — [A Multi-World Approach to Question Answering about Real-World Scenes based on Uncertain Input](https://arxiv.org/abs/1410.0210) — Malinowski & Fritz (NeurIPS) — DAQUAR, the first VQA-style dataset/benchmark.

## 2015

- **2015** — [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468) — Antol et al. (ICCV) — defines the open-ended VQA task at scale; the canonical task formulation CULPRIT-VQA's causal-attribution layer is built on top of.
- **2015** — [Exploring Models and Data for Image Question Answering](https://arxiv.org/abs/1505.02074) — Ren, Kiros & Zemel (NeurIPS) — COCO-QA; caption-derived question generation, an early source of answer-distribution bias.
- **2015** — [Visual Madlibs: Fill in the blank Image Generation and Question Answering](https://arxiv.org/abs/1506.00278) — Yu, Park, Berg & Berg (ICCV) — templated fill-in-the-blank QA, relevant to failure modes tied to question phrasing/template.

## 2016

- **2016** — [Visual7W: Grounded Question Answering in Images](https://arxiv.org/abs/1511.03416) — Zhu, Groth, Bernstein & Fei-Fei (CVPR) — adds object-level visual grounding to QA, enabling grounding-failure vs. reasoning-failure attribution.
- **2016** — [Yin and Yang: Balancing and Answering Binary Visual Questions](https://arxiv.org/abs/1511.05099) — Zhang, Goyal, Summers-Stay, Batra & Parikh (CVPR) — first dataset-balancing approach to counter language priors in binary VQA.
- **2016** — [Analyzing the Behavior of Visual Question Answering Models](https://arxiv.org/abs/1606.07356) — Agrawal, Batra & Parikh (EMNLP) — early behavioral evidence that VQA models are "myopic," answering from partial question text while ignoring the image.
- **2016** — ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) — Ribeiro, Singh & Guestrin (KDD) — LIME; a contrasting *correlational* local-attribution baseline against which CULPRIT-VQA's causal (interventional) attribution should be positioned.

## 2017

- **2017** — [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890) — Johnson et al. (CVPR) — synthetic dataset with per-question reasoning-step annotations; a template for pinpointing which reasoning stage fails.
- **2017** — [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837) — Goyal, Khot, Summers-Stay, Batra & Parikh (CVPR) — VQA v2.0; paired complementary images quantify language-prior bias vs. genuine visual reasoning.
- **2017** — [An Analysis of Visual Question Answering Algorithms](https://arxiv.org/abs/1703.09684) — Kafle & Kanan (ICCV) — TDIUC, 12 question-type categories and metrics designed to expose per-category failure modes.
- **2017** — [Visual Question Answering: Datasets, Algorithms, and Future Challenges](https://arxiv.org/abs/1610.01465) — Kafle & Kanan (*CVIU*, survey) — critical review of VQA dataset/metric/algorithm limitations.
- **2017** — [Visual Question Answering: A Survey of Methods and Datasets](https://arxiv.org/abs/1607.05910) — Wu, Teney, Wang, Shen, Dick & van den Hengel (survey) — complementary architecture/dataset survey.
- **2017** — ["Why Should I Trust You?" companion baseline —] [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) — Lundberg & Lee (NeurIPS) — **SHAP**: unifies additive feature-attribution methods via the Shapley value; the direct methodological ancestor of CULPRIT-VQA's φ_m.
- **2017** — [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) — Sundararajan, Taly & Yan (ICML) — Integrated Gradients; a contrasting gradient-based (non-causal) attribution baseline.

## 2018

- **2018** — [VizWiz Grand Challenge: Answering Visual Questions from Blind People](https://arxiv.org/abs/1802.08218) — Gurari et al. (CVPR) — real-world, low-quality-image VQA with unanswerable questions; a natural source of input-quality failure distinct from reasoning failure.
- **2018** — [Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering](https://arxiv.org/abs/1712.00377) — Agrawal, Batra, Parikh & Kembhavi (CVPR) — introduces **VQA-CP** and GVQA, showing models exploit train/test answer-distribution correlations rather than image grounding.

## 2019

- **2019** — [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506) — Hudson & Manning (CVPR) — scene-graph-grounded, program-annotated questions enabling fine-grained causal tracing of compositional-reasoning failure.
- **2019** — [Towards VQA Models That Can Read](https://arxiv.org/abs/1904.08920) — Singh et al. (CVPR) — TextVQA; isolates OCR/text-reading failure as distinct from visual/semantic reasoning failure.
- **2019** — [OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge](https://arxiv.org/abs/1906.00067) — Marino, Rastegari, Farhadi & Mottaghi (CVPR) — isolates "missing external knowledge" as a distinct failure cause — the direct precursor of CULPRIT-VQA's knowledge axis.
- **2019** — [Cycle-Consistency for Robust Visual Question Answering](https://arxiv.org/abs/1902.05660) — Shah, Chen, Rohrbach & Parikh (CVPR) — VQA-Rephrasings; models are brittle to linguistic rephrasing of the same question.
- **2019** — [RUBi: Reducing Unimodal Biases for Visual Question Answering](https://proceedings.neurips.cc/paper/2019/hash/51d92be1c60d1db1d2e5e7a07da55b26-Abstract.html) — Cadène, Dancette, Ben-younes, Cord & Parikh (NeurIPS) — dynamically down-weights training examples answerable from the question alone.
- **2019** — [Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases](https://arxiv.org/abs/1909.03683) — Clark, Yatskar & Zettlemoyer (EMNLP) — Learned-Mixin+H (LMH), an ensemble debiasing method pairing a bias-only model with a robust one.
- **2019** — [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265) — Lu et al. (NeurIPS) — first two-stream co-attention VLM; founds the pre-LLM VLM lineage.
- **2019** — [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) — Tan & Bansal (EMNLP-IJCNLP) — three-encoder cross-modality transformer, contemporaneous VQA-oriented VLM.

## 2020

- **2020** — [Overcoming Language Priors in VQA via Decomposed Linguistic Representations](https://ojs.aaai.org/index.php/AAAI/article/view/6776) — Jing, Wu, Zhang, Jia & Wu (AAAI) — decomposes questions into type/object/concept representations to reduce language-prior shortcuts.
- **2020** — [Counterfactual Samples Synthesizing for Robust Visual Question Answering](https://arxiv.org/abs/2003.06576) — Chen, Yan, Xiao, Zhang, Pu & Zhuang (CVPR) — CSS; synthesizes counterfactual samples with masked critical objects/words to force genuine multimodal grounding.
- **2020** — [Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing](https://arxiv.org/abs/1912.07538) — Agarwal, Shetty & Fritz (CVPR) — IV-VQA/CV-VQA; image-editing consistency checks reveal spurious visual correlations.
- **2020** — [Overcoming Language Priors with Self-supervised Learning for Visual Question Answering](https://arxiv.org/abs/2012.11528) — Zhu, Mao, Liu, Zhang, Wang & Zhang (IJCAI) — self-supervised auxiliary task on auto-balanced data.
- **2020** — [Investigating Gender Bias in Language Models Using Causal Mediation Analysis](https://papers.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html) — Vig, Gehrmann, Belinkov, Qian, Nevo, Singer & Shieber (NeurIPS, Spotlight) — causal mediation analysis for interpretability; a direct methodological cousin of CULPRIT-VQA's do-operations.
- **2020** — [The Shapley Taylor Interaction Index](https://arxiv.org/abs/1902.05622) — Sundararajan, Dhamdhere & Agarwal (ICML, Google) — generalizes the Shapley value to k-order feature interactions; the modern basis for CULPRIT-VQA's pairwise interaction indices.
- **2020** — [UNITER: UNiversal Image-TExt Representation Learning](https://arxiv.org/abs/1909.11740) — Chen et al. (ECCV) — single-stream transformer unifying four vision-language pretraining tasks.

## 2021

- **2021** — [Counterfactual VQA: A Cause-Effect Look at Language Bias](https://arxiv.org/abs/2006.04315) — Niu, Tang, Zhang, Lu, Hua & Wen (CVPR) — **CF-VQA**; uses causal-effect/counterfactual inference to isolate and subtract the direct language-bias effect — the closest prior VQA work to CULPRIT-VQA's own causal framing.
- **2021** — [Roses Are Red, Violets Are Blue... But Should VQA Expect Them To?](https://openaccess.thecvf.com/content/CVPR2021/html/Kervadec_Roses_Are_Red_Violets_Are_Blue..._but_Should_VQA_Expect_CVPR_2021_paper.html) — Kervadec et al. (CVPR) — GQA-OOD; fine-grained OOD compositional VQA separating rare vs. frequent answer accuracy.
- **2021** — [Check It Again: Progressive Visual Question Answering via Visual Entailment](https://arxiv.org/abs/2106.04605) — Si, Lin et al. (ACL) — SAR; re-ranks candidate answers via visual entailment to reduce language-prior reliance.
- **2021** — [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al. (OpenAI, ICML) — web-scale contrastive image-text pretraining; its vision encoder becomes the standard backbone for nearly all later VLMs.
- **2021** — [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) — Kim et al. (ICML) — removes CNN/region-detector pipelines for a pure patch-based transformer.
- **2021** — [xGQA: Cross-Lingual Visual Question Answering](https://arxiv.org/abs/2109.06082) — Pfeiffer, Geigle, Kamath, Steitz, Roth, Vulić & Gurevych (Findings of ACL 2022) — extends GQA to 7 typologically diverse languages, an early cross-lingual VQA benchmark.

## 2022

- **2022** — [Language Prior Is Not the Only Shortcut: A Benchmark for Shortcut Learning in VQA](https://arxiv.org/abs/2210.04692) — Si, Meng, Zheng et al. (Findings of EMNLP) — VQA-VS; generalizes shortcut-learning study beyond language-only bias, showing debiasing methods overfit to that one shortcut type.
- **2022** — [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) — Li et al. (Salesforce, ICML) — unifies understanding/generation via bootstrapped caption filtering.
- **2022** — [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) — Alayrac et al. (DeepMind, NeurIPS) — bridges frozen vision/language models via gated cross-attention; establishes the "frozen-backbone + adapter" VLM paradigm.
- **2022** — [MaXM: Towards Multilingual Visual Question Answering](https://arxiv.org/abs/2209.05401) — Changpinyo, Xue, Yarom, Thapliyal, Szpektor, Amelot, Chen & Soricut (Google, Findings of EMNLP 2023) — multilingual crossmodal VQA across 7 languages.
- **2022** — [MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks](https://arxiv.org/abs/2212.08158) — Parcalabescu & Frank (ACL 2023) — applies Shapley values to quantify text-vs-image contribution in VL model predictions — the most direct prior application of Shapley attribution to multimodal models.

## 2023

- **2023** — [Large Language Models Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093) — Shi, Chen, Misra, Scales, Dohan, Chi, Schärli & Zhou (ICML) — introduces **GSM-IC**, the origin paper for textual/irrelevant-context distraction (CULPRIT-VQA's "irrelevant plausible fact" operator).
- **2023** — [Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355) — Li et al. (EMNLP) — **POPE**; polling-based binary-query benchmark for object hallucination.
- **2023** — [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://arxiv.org/abs/2310.14566) — Guan et al. (CVPR 2024; posted Oct 2023) — disentangles language-hallucination (prior override) from visual-illusion failure via control question pairs.
- **2023** — [Survey of Social Bias in Vision-Language Models](https://arxiv.org/abs/2309.14381) — Lee, Bang, Lovenia, Cahyawijaya, Dai & Fung — survey of bias/fairness in VLMs, background for the cultural-bias framing.
- **2023** — [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Li et al. (Salesforce, ICML) — the lightweight Q-Former bridging frozen vision encoders and frozen LLMs, reused across later open VLMs.
- **2023** — [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — Liu et al. (NeurIPS) — GPT-4-generated instruction data + a linear vision-to-LLM projector; launches the open-source "visual instruction tuning" family.
- **2023** — [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744) — Liu et al. — MLP projector + academic VQA data; the strong open-source LLaVA baseline.
- **2023** — [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](https://arxiv.org/abs/2305.06500) — Dai et al. (Salesforce, NeurIPS) — instruction-aware Q-Former extending BLIP-2.
- **2023** — [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966) — Bai et al. (Alibaba) — founds the Qwen-VL lineage used directly as a Layer-2 model family in this project.
- **2023** — [GPT-4V(ision) System Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf) — OpenAI — closed, behavioral-only multimodal model.
- **2023** — [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) — Gemini Team, Google DeepMind — natively multimodal transformer trained jointly across text/image/audio/video.

## 2024

- **2024** — [Losing Visual Needles in Image Haystacks: Vision Language Models are Easily Distracted in Short and Long Contexts](https://arxiv.org/abs/2406.16851) — Sharma, Saxon & Wang (EMNLP Findings) — LoCoVQA; visual distraction via many distractor images, logarithmic accuracy decay.
- **2024** — [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238) — Chen et al. (CVPR) — scales the vision encoder itself to 6B params, challenging the "small adapter, frozen big encoder" paradigm.
- **2024** — [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326) — Li et al. — unifies single-image/multi-image/video understanding; used directly as a Layer-2 model in this project.
- **2024** — [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) — Wang et al. (Alibaba) — Naive Dynamic Resolution + M-RoPE for native-resolution multimodal positional encoding.
- **2024** — [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling (InternVL2.5)](https://arxiv.org/abs/2412.05271) — Chen et al. — first open MLLM to exceed 70% on MMMU.
- **2024** — [GPT-4o System Card](https://arxiv.org/abs/2410.21276) — OpenAI — natively multimodal any-to-any closed model.
- **2024** — [Pangea: A Fully Open Multilingual Multimodal LLM for 39 Languages](https://arxiv.org/abs/2410.16153) — Yue, Song, Asai, Kim, Nyandwi, Khanuja, Kantharuban, Sutawika, Ramamoorthy & Neubig (CMU) — CLIP-ViT + Qwen2-7B multilingual/multicultural VLM; direct architectural predecessor of CulturalPangea.
- **2024** — [Peacock: A Family of Arabic Multimodal Large Language Models and Benchmarks](https://arxiv.org/abs/2403.01031) — Alwajih, Nagoudi, Bhatia, Mohamed & Abdul-Mageed — introduces **Henna**, an Arabic cultural VQA benchmark (11 Arab countries).
- **2024** — [CVQA: Culturally-diverse Multilingual Visual Question Answering Benchmark](https://arxiv.org/abs/2406.05967) — Romero et al. (NeurIPS, Datasets & Benchmarks Track) — 10k questions, 30 countries, 31 languages; **the actual substrate benchmark this project's Layer 1 is built on.**

## 2025

- **2025** — [LLMs can be easily Confused by Instructional Distractions](https://aclanthology.org/2025.acl-long.957/) — Hwang, Kim, Koo, Kang, Bae & Jung (ACL) — **DIM-Bench**; textual distraction where embedded text is mistaken for an instruction.
- **2025** — [On the robustness of multimodal language model towards distractions](https://arxiv.org/abs/2502.09818) — Liu, Chen, Wang & Zhang — direct visual-vs-textual distraction comparison; textual distraction found more damaging.
- **2025** — [CLASH: A Benchmark for Cross-Modal Contradiction Detection](https://arxiv.org/abs/2511.19199) — Popordanoska et al. (CVPR 2026; code: [github.com/tpopordanoska/clash](https://github.com/tpopordanoska/clash)) — COCO-based image–caption contradiction detection at object/attribute level.
- **2025** — [Understanding the Effects of Distractors on Reasoning Vision-Language Models](https://arxiv.org/abs/2511.21397) — Bae, Ok, Mo & Lee — introduces **Idis**; visual distraction varied along semantic/numerical dimensions (see caveat below — not diffusion-based).
- **2025** — [CrossCheck-Bench: Diagnosing Compositional Failures in Multimodal Conflict Resolution](https://arxiv.org/abs/2511.21717) — Tian et al. — 15k-item hierarchical cross-modal contradiction benchmark; models fail at compositional multi-clue conflict reasoning.
- **2025** — [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) — Bai et al. (Alibaba) — native dynamic-resolution ViT trained from scratch; **the exact Qwen2.5-VL-3B backbone used as a Layer-2 model in this project.**
- **2025** — [Grounding Multilingual Multimodal LLMs With Cultural Knowledge](https://arxiv.org/abs/2508.07414) — Nyandwi, Song, Khanuja & Neubig (CMU, EMNLP) — introduces **CulturalGround** (22M culturally-rich VQA pairs, 42 countries/39 languages) and **CulturalPangea** (a CulturalGround-tuned Pangea-7B); the culture-tuned comparison model referenced in this project's proposal §6.3.

## 2026

- **2026** — [Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning](https://arxiv.org/abs/2602.14518) — Tang, Wang, Lu, Chen, Chen, Sun, Li, Lyu, Nan & Zeng — knowledge-conflict types are linearly decodable in mid-to-late layers during MLLM chain-of-thought reasoning.
- **2026** — [VLM-RobustBench: A Comprehensive Benchmark for Robustness of Vision-Language Models](https://arxiv.org/abs/2603.06148) — (ICML) — 49 corruption/augmentation types; the corruption-robustness literature this project explicitly scopes *out* of its own failure taxonomy.
- **2026** — [Benchmarking Deflection and Hallucination in Large Vision-Language Models](https://aclanthology.org/2026.acl-long.1307/) — Moratelli, Davis, Ribeiro, Byrne & Iglesias (ACL) — measures whether models abstain ("deflect") vs. hallucinate under noisy/insufficient evidence (referred to informally as "VLM-DeflectionBench" in this project's proposal).
- **2026** — [When Text Hijacks Vision: Benchmarking and Mitigating Text Overlay-Induced Hallucination in Vision Language Models](https://arxiv.org/abs/2604.17375) — Cui, Qi, Geng, Zhang, Han & Guo — introduces **VisualTextTrap**; on-screen text overlay contradicting scene content causes hallucination (see caveat below — video domain, not static image VQA).
- **2026** — [The Curse of Helpfulness: Inverse Scaling Law in Robustness to Distractor Instructions via DistractionIF](https://arxiv.org/abs/2605.29491) — Su, Xu, Chen, Zheng, Zhang, Zhou & Zhang — larger (text-only) models are *more* prone to treating embedded editorial text as instructions.
- **2026** — [ReactBench: A Cause-Driven Benchmark for Multimodal Hallucination via Systematic Evaluation](https://arxiv.org/abs/2605.29579) — Zhou, Jia, Wu, Shen, Li, Wu & Lin — adversarial-image benchmark targeting *why* MLLMs hallucinate, not just detection.
- **2026** — [When Image and Text Disagree: Cross-Modal Evidence Conflict in Multimodal Retrieval-Augmented Generation](https://aclanthology.org/2026.magmar-main.3/) — Catapang (MAGMaR workshop @ ACL) — cross-modal conflict types (factual/temporal/entity/granularity) in multimodal RAG.
- **2026** — [SADL: What to Ignore? A Benchmark for Subject-Aware Distractor Localization](https://arxiv.org/abs/2606.30393) — Nguyen, Luong, Nguyen & Tran — visual distractor identification/removal; models over-apply removal, suppressing legitimate content.
- **2026** — [Are Reasoning Vision-Language Models Robust to Semantic Visual Distractions?](https://arxiv.org/abs/2606.08894) — Sun, Zhan, Ma, See, Wang, Wang, Li, Cui, Cai, Sun, Lin & Batista-Navarro — introduces **Distract-Bench**; reasoning VLMs incorporate irrelevant distractors into their own chain-of-thought, causing wrong conclusions.
- **2026** — [Defying Distractions in Multimodal Tasks: A Novel Benchmark for Large Vision-Language Models](https://doi.org/10.1109/tpami.2026.3655641) — (IEEE TPAMI, Vol. 48, Issue 6) — IR-VQA benchmark + Positive/Negative Consistency metrics for multimodal distractibility (visual and textual).
- **2026** — [MMAC: A Multilingual, Multimodal Alignment Framework for Cultural Grounding Evaluation](https://aclanthology.org/2026.acl-long.989/) — Zheng, Liu, Chakraborty et al. (ACL) — 27k questions, 10 languages, cross-modal (text/image/speech) cultural grounding evaluation.

---

## Top 25 most important papers for CULPRIT-VQA

Ranked by how load-bearing each paper is for the project's actual design decisions (causal framing,
substrate, model runner, or the specific failure factors studied) — not by citation count or fame.

1. **Antol et al., 2015 — [VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468)** — defines the task CULPRIT-VQA's entire causal-attribution layer sits on top of.
2. **Shapley, 1953 — [A Value for n-Person Games](https://www.rand.org/pubs/papers/P295.html)** — the mathematical object (φ_m) at the center of Layer 3a.
3. **Lundberg & Lee, 2017 — [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)** — the modern ML lineage that made Shapley-value attribution practical; direct methodological ancestor.
4. **Sundararajan, Dhamdhere & Agarwal, 2020 — [The Shapley Taylor Interaction Index](https://arxiv.org/abs/1902.05622)** — formal basis for the pairwise interaction indices central to RQ1.
5. **Parcalabescu & Frank, 2022/2023 — [MM-SHAP](https://arxiv.org/abs/2212.08158)** — the closest prior work applying Shapley attribution specifically to multimodal (vision-language) model contributions.
6. **Niu et al., 2021 — [Counterfactual VQA: A Cause-Effect Look at Language Bias](https://arxiv.org/abs/2006.04315)** — the closest prior *causal* framing of VQA failure, which CULPRIT-VQA generalizes from single-factor language bias to a multi-factor factorial lattice.
7. **Goyal et al., 2017 — [Making the V in VQA Matter (VQA v2)](https://arxiv.org/abs/1612.00837)** — established that accuracy gaps conflate causes — the problem statement CULPRIT-VQA is a direct answer to.
8. **Agrawal et al., 2018 — [Don't Just Assume; Look and Answer (VQA-CP / GVQA)](https://arxiv.org/abs/1712.00377)** — the foundational demonstration that VQA "accuracy" can be a language-prior artifact, motivating factor-level attribution instead of aggregate accuracy.
9. **Hudson & Manning, 2019 — [GQA](https://arxiv.org/abs/1902.09506)** — compositional, scene-graph-grounded questions; the template for attributing failure to a specific reasoning step.
10. **Marino et al., 2019 — [OK-VQA](https://arxiv.org/abs/1906.00067)** — established "missing knowledge" as a distinct, non-perceptual failure cause — CULPRIT-VQA's knowledge axis.
11. **Vig et al., 2020 — [Causal Mediation Analysis for Gender Bias in LMs](https://papers.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html)** — demonstrates do-operation-style causal interventions for bias attribution outside VQA, validating the general approach.
12. **Romero et al., 2024 — [CVQA](https://arxiv.org/abs/2406.05967)** — the actual substrate dataset CULPRIT-VQA's Layer 1 factorial lattice is built on.
13. **Nyandwi, Song, Khanuja & Neubig, 2025 — [CulturalGround / CulturalPangea](https://arxiv.org/abs/2508.07414)** — the culture-tuned model and dataset used as CULPRIT-VQA's comparison point on the culturality axis.
14. **Radford et al., 2021 — [CLIP](https://arxiv.org/abs/2103.00020)** — the vision-encoder backbone underlying essentially every VLM CULPRIT-VQA could run as Layer 2.
15. **Bai et al., 2025 — [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)** — the exact model family used for the project's main 150-item real experiment.
16. **Li et al., 2024 — [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)** — the exact second model used for the cross-model comparison experiment.
17. **Li et al., 2023 — [BLIP-2](https://arxiv.org/abs/2301.12597)** — the Q-Former bridging pattern that underlies the broader open-VLM lineage CULPRIT-VQA's runners belong to.
18. **Shi et al., 2023 — [GSM-IC: LLMs Can Be Easily Distracted by Irrelevant Context](https://arxiv.org/abs/2302.00093)** — origin paper for the "irrelevant plausible fact" textual-distraction operator.
19. **Sun et al., 2026 — [Distract-Bench](https://arxiv.org/abs/2606.08894)** — direct namesake/basis for the project's visual-distraction operator and positioning-table comparison.
20. **Bae, Ok, Mo & Lee, 2025 — [Idis](https://arxiv.org/abs/2511.21397)** — closest prior visual-distractor dataset design (see citation caveat: not diffusion-based as the proposal implies).
21. **Hwang et al., 2025 — [DIM-Bench](https://aclanthology.org/2025.acl-long.957/)** — closest prior work on text-as-instruction distraction, positioned against in the proposal's table.
22. **Popordanoska et al., 2025 — [CLASH](https://arxiv.org/abs/2511.19199)** — closest prior cross-modal contradiction-detection benchmark, the conceptual basis for the "contradictory caption" operator.
23. **Cui et al., 2026 — [VisualTextTrap](https://arxiv.org/abs/2604.17375)** — basis for the "text overlay asserting wrong answer" operator (see citation caveat: video domain).
24. **Tian et al., 2025 — [CrossCheck-Bench](https://arxiv.org/abs/2511.21717)** — shows models fail at *compositional* multi-clue conflict — direct evidence for why CULPRIT-VQA studies factor interactions rather than single factors.
25. **Tang et al., 2026 — [Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning](https://arxiv.org/abs/2602.14518)** — closest prior work on the auxiliary S3 (NLI conflict) behavioral signal.

---

## Citation caveats

Two discrepancies surfaced during verification between how `PROPOSAL_CULPRIT_VQA.md` §6.1 describes an
operator recipe and what the cited paper actually contains. Worth checking before the proposal is
finalized or submitted:

1. **"Diffusion-inserted salient irrelevant object (Idis recipe)"** — the real Idis paper (Bae, Ok, Mo &
   Lee, 2025, arXiv:2511.21397) varies distractors along *semantic and numerical* dimensions; it does
   **not** use diffusion-model image editing to insert distractors. If the operator itself is meant to
   literally use diffusion-based insertion, a different source paper is needed for that specific
   technique, or the description should be reworded to match what Idis actually does.
2. **"Text overlay asserting wrong answer (VisualTextTrap recipe)"** — the real VisualTextTrap paper
   (Cui et al., 2026, arXiv:2604.17375) is about **video** VLMs ("Text Hijacks Vision... in video
   understanding tasks"), not static-image VQA. If the operator needs an image-only analogue, either
   adapt the recipe description accordingly or find an image-domain source.

Both papers are real and correctly identified otherwise — only the specific mechanism/domain
attributed to them diverges from the source.
