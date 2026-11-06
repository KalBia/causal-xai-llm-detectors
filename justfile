# Run `just` to list recipes. Requires: uv, just.
config := "configs/default.yaml"

default:
    @just --list

# Install dependencies (creates .venv)
sync:
    uv sync

# Fetch git submodules (external code, e.g. M4)
submodules:
    git submodule update --init --recursive

# Whole pipeline end to end (dry run by default, see `dry_run` in the config)
pipeline:
    uv run python main.py all --config {{config}}

# Prepare M4 (English) splits
data:
    uv run python main.py data --config {{config}}

# Fine-tune RoBERTa detector on M4
train:
    uv run python main.py train --config {{config}}

# Predict on the test split, report Macro-F1 / AUROC
eval-detector:
    uv run python main.py eval-detector --config {{config}}

# Compute attributions (SHAP / IG / random)
explain:
    uv run python main.py explain --config {{config}}

# Experiments RQ1 (faithfulness) + RQ2 (operators)
evaluate:
    uv run python main.py evaluate --config {{config}}

# Experiment RQ3: washed detector
washing:
    uv run python main.py washing --config {{config}}

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format . && uv run ruff check --fix .

clean:
    find . -name __pycache__ -type d -prune -exec rm -rf {} +
    rm -rf .pytest_cache .ruff_cache
