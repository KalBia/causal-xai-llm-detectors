"""Typer CLI. Each command loads a YAML/JSON config and runs one pipeline stage."""

from pathlib import Path

import typer

from src.utils.config import load_config

app = typer.Typer(help="Causal validation of explanations of LLM text detectors.", no_args_is_help=True)

ConfigOpt = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="YAML/JSON config")


def _banner(stage: str, cfg: dict) -> None:
    typer.echo(f"=== [{stage}] seed: {cfg.get('seed')} ===")


def _run(stage: str, config: Path):
    cfg = load_config(config)
    _banner(stage, cfg)
    return cfg


@app.command()
def data(config: Path = ConfigOpt):
    """Load M4 (English), preprocess, split train/val/test."""
    from src.datasets import prepare_data

    prepare_data(_run("data", config))


@app.command()
def train(config: Path = ConfigOpt):
    """Fine-tune the detector on the train split."""
    from src.detectors import train_detector

    train_detector(_run("train", config))


@app.command("eval-detector")
def eval_detector(config: Path = ConfigOpt, split: str = "test"):
    """Predict on a split and report Macro-F1 / AUROC."""
    from src.detectors import evaluate_detector

    evaluate_detector(_run("eval-detector", config), split)


@app.command()
def explain(config: Path = ConfigOpt):
    """Compute attributions (SHAP / IG / random)."""
    from src.detectors import build_detector
    from src.explainers.run import run_explain

    cfg = _run("explain", config)
    run_explain(cfg, build_detector(cfg))


@app.command()
def evaluate(config: Path = ConfigOpt):
    """Experiments RQ1 (faithfulness) and RQ2 (operator comparison)."""
    from src.detectors import build_detector
    from src.experiments.rq1 import run_rq1
    from src.experiments.rq2 import run_rq2

    cfg = _run("evaluate", config)
    det = build_detector(cfg)
    run_rq1(cfg, det)
    run_rq2(cfg, det)


@app.command()
def washing(config: Path = ConfigOpt):
    """Experiment RQ3: washed detector vs original."""
    from src.detectors import build_detector
    from src.experiments.rq3 import run_rq3

    cfg = _run("washing", config)
    run_rq3(cfg, build_detector(cfg))


@app.command("all")
def all_stages(config: Path = ConfigOpt):
    """Run the whole pipeline end to end (smoke test of the skeleton)."""
    for fn in (data, train, eval_detector, explain, evaluate, washing):
        fn(config) if fn is not eval_detector else fn(config, "test")
    typer.echo("=== pipeline finished ===")
