# Causal XAI for LLM Detectors

Causal validation of explanations of LLM-generated text detectors — SIWY project

## Topic

Detectors of LLM-generated text (systems that classify text as human- or AI-written) are commonly explained with **attribution methods** that highlight the tokens driving a decision. But do the highlighted tokens actually influence the detector's verdict? This project validates explanations **causally**: we apply a set of causal interventions on the text and measure whether the features indicated by explanation methods genuinely affect detector decisions.

We also study **explanation washing** — situations where explanations change even though the model's actual decision mechanism has not, which can give a false sense of interpretability.

## Research question

Which explanation methods point to features that are genuinely used by LLM-generated-text detectors, as measured by causal interventions on the text?

## Tech stack

| Area | Choice |
|---|---|
| Language & environment | Python ≥ 3.10, dependencies managed with **uv** | 
| Code quality | **ruff** (lint + format) | 
| Version control | Git + GitHub, **Conventional Commits** |
| CLI & automation | **typer** CLI (`main.py`) and a **justfile** for build / experiment automation | 
| Models & data | Hugging Face **Transformers** and **datasets** | 
| XAI | **shap** (Partition Explainer), **inseq** / Captum (Integrated Gradients) | 
| Causal interventions | Custom module (`src/interventions`) for token deletion / replacement | 
| Experiment tracking | **Weights & Biases** | 
| Configuration | YAML / JSON files (`configs/`) | 

## Quick start

```bash
uv sync                # install dependencies for dry-run mode
just submodules        # fetch external/M4 and external/fast-detect-gpt
just pipeline          # whole pipeline of thw project (for now only skeleton with prints: development in the process..)
just                   # list recipes (data, train, eval-detector, explain, evaluate, washing, test, lint)
uv run python main.py --help
```

## Planned outputs

- A **benchmark** for causal validation of explanations of LLM text detectors
- A **comparison** of explainability methods
- A **case study** of explanation washing
- A set of **recommendations** for auditing LLM-generated-text detectors

## Deliverables (per SIWY regulations)

- [x] Design proposal (schedule by dates, bibliography, planned experiments, tech stack)
- [x] Documented literature analysis (annotated table of related papers)
- [x] Functional prototype demonstration
- [ ] Documentation, usage instructions, and tests
- [ ] 3–5 minute summary video
- [ ] Scientific article describing the project and results
- [ ] Final public presentation

## Documentation

- [Design proposal](docs/design_proposal.pdf) — schedule, bibliography, planned experiments, tech stack
- [Related work](docs/related_work.md) — annotated table of related papers
- [Weekly notes](docs/weekly_notes.md) — progress log and meeting notes of the project

---

## Conclusions

### 1. Research questions

Based on the literature review we choose the following details of our research:
1. **Research questions:**
	1. **RQ1 (Faithfulness baseline)**. When we remove the tokens that an explanation method says are most important, does the detector's decision change as predicted? We test this with standard measures (comprehensiveness, sufficiency, deletion-flip curves) and compare our results with what earlier studies found for text classifiers in general.
	2. **RQ2 (Operator dependence)**. Does the answer depend on how we remove or change a token: deleting it, replacing it with a realistic substitute, or paraphrasing the surrounding text?
	3. **RQ3 (Explanation washing)**. Can we build a detector that makes the same decisions as the original one but gives different explanations, and can our causal tests tell these two detectors apart?

### 2. Detectors

| Detector | Type | Why |
|---|---|---|
| **RoBERTa-base fine-tuned on M4** | Supervised, white-box | Main detector. Full gradient access, so all explanation methods apply; standard baseline in M4GT-Bench; also the model we manipulate for washing (RQ3). Low cost (~125M params, one 12-16 GB GPU). |
| **Fast-DetectGPT** | Zero-shot, probability curvature | Comparison from a different detector family. Score is a sum of per-token values, so it has built-in evidence to compare with attributions. Needs a scoring LLM (GPT-Neo 2.7B / GPT-J 6B), one 16-24 GB GPU. |

### 3. Explanation methods

| Method | Type | Why |
|---|---|---|
| **SHAP** (Partition Explainer) | Perturbation-based, black-box | Best method for MGT detectors in Schoenegger et al. (2024); signed attributions allow tests in both directions. Needs a reference value (a choice to ablate). |
| **Integrated Gradients** | Gradient-based, white-box (RoBERTa only) | Different mechanism from SHAP, so (dis)agreement is informative. Needs a baseline (e.g. pad/mask embedding). |
| **Random attribution** | Control | Null for deletion curves; draw 5-10 random vectors per text for confidence intervals. |

Occlusion (leave-one-out) is dropped: it uses the same operator as the deletion test, so its faithfulness is inflated by construction.

### 4. Interventions

- **Deletion / masking** of the top-K tokens.
- **Replacement** with a realistic substitute (synonym or masked-LM suggestion).

### 5. Washing experiment (RQ3)

Following adversarial model manipulation (Heo et al. 2019) and the theory of Anders et al. (2020), fine-tune a copy of RoBERTa with a loss made of two parts: (1) output agreement with the original (e.g. KL on logits), (2) a penalty that pushes attributions elsewhere (e.g. IG mass onto punctuation/stopwords). Then check: are predictions still (almost) identical, did explanations change, and do comprehensiveness/sufficiency expose the new explanations as non-causal?

### 6. Dataset

**M4 (English subset only)**, a few thousand texts, split train / val / test (e.g. 70 / 15 / 15) for fine-tuning RoBERTa and running the interventions.

### 7. Metrics

- **Detector sanity check:** Macro-F1, AUROC.
- **Causal faithfulness:** comprehensiveness, sufficiency (top-K tokens removed / kept; definitions from ERASER, DeYoung et al. 2020), deletion curve and its AUC, always against the random baseline.

### 8. Order of work

1. RoBERTa + SHAP / IG / random + deletion (basic RQ1).
2. Add replacement operator (operator comparison, RQ2).
3. Washing on RoBERTa (RQ3).
4. Optional: Fast-DetectGPT, paraphrase, ground truth from Task 3.
