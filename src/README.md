# src layout

Implemented so far: `datasets/`, `detectors/`, `explainers/`. Still skeletons (print only): `interventions/` (deletion only), `evaluation/`, `experiments/`, `washing/`. `main.py all` runs the stages in pipeline order.

| Package | Purpose |
|---|---|
| `datasets/` | M4 (English): load -> preprocess -> split -> save |
| `detectors/` | `Detector` interface; RoBERTa-base (main), Fast-DetectGPT (optional); train / evaluate stages |
| `explainers/` | `Explainer` interface (word-level tokens, signed attributions, `explain_draws` for multi-draw nulls); SHAP Partition (black-box), Integrated Gradients via Captum (white-box, RoBERTa only), seeded random baseline; `just explain` saves JSONL to `results/explain/<name>/<split>.jsonl` |
| `interventions/` | Deletion, replacement |
| `evaluation/` | Faithfulness metrics (comprehensiveness, sufficiency, deletion AUC) |
| `experiments/` | `faithfulness.py` (shared step), `rq1.py` (faithfulness), `rq2.py` (operators), `rq3.py` (washing) |
| `washing/` | Explanation-washing fine-tuning (RQ3) |
| `utils/` | YAML/JSON config loading, W&B tracking |
| `cli.py` | typer CLI (`main.py`), one command per pipeline stage |

## Pipeline

`just submodules` -> `just data` -> `just train` -> `just eval-detector` -> `just explain` -> `just evaluate` (RQ1+RQ2) -> `just washing` (RQ3)
or everything at once: `just pipeline` (= `uv run python main.py all`).
