# Causal XAI for LLM Detectors

Causal validation of explanations of LLM-generated text detectors — SIWY project

## Topic

Detectors of LLM-generated text (systems that classify text as human- or AI-written) are commonly explained with **attribution methods** that highlight the tokens driving a decision. But do the highlighted tokens actually influence the detector's verdict? This project validates explanations **causally**: we apply a set of causal interventions on the text and measure whether the features indicated by explanation methods genuinely affect detector decisions.

We also study **explanation washing** — situations where explanations change even though the model's actual decision mechanism has not, which can give a false sense of interpretability.

## Research question

Which explanation methods point to features that are genuinely used by LLM-generated-text detectors, as measured by causal interventions on the text?

## Planned outputs

- A **benchmark** for causal validation of explanations of LLM text detectors
- A **comparison** of explainability methods
- A **case study** of explanation washing
- A set of **recommendations** for auditing LLM-generated-text detectors

## Deliverables (per SIWY regulations)

- [ ] Design proposal (schedule by dates, bibliography, planned experiments, tech stack)
- [ ] Documented literature analysis (annotated table of related papers)
- [ ] Functional prototype demonstration
- [ ] Documentation, usage instructions, and tests
- [ ] 3–5 minute summary video
- [ ] Scientific article describing the project and results
- [ ] Final public presentation

## Status

Repository initialized — design proposal in progress.
