"""Smoke test: every stage of the skeleton runs, in order."""

import pytest
from typer.testing import CliRunner

from src.cli import app


@pytest.fixture(autouse=True)
def _no_real_data(monkeypatch):
    """The data stage needs the M4 submodule (~2 GB, absent in CI) and writes to data/: stub it here.
    The real logic is covered by tests/test_datasets.py."""
    monkeypatch.setattr("src.datasets.prepare_data", lambda cfg: print("[dataset pipeline] stubbed in tests"))
     # detectors need torch, data and a checkpoint, none of which exist in CI
    monkeypatch.setattr("src.detectors.train_detector", lambda cfg: print("[train] stubbed in tests"))
    monkeypatch.setattr("src.detectors.RobertaDetector.predict_proba", lambda self, texts: [0.5] * len(texts))


def test_each_stage_runs():
    for cmd in ("data", "train", "eval-detector", "explain", "evaluate", "washing"):
        res = CliRunner().invoke(app, [cmd])
        assert res.exit_code == 0, f"{cmd}: {res.output}"


def test_full_pipeline_order():
    res = CliRunner().invoke(app, ["all"])
    assert res.exit_code == 0, res.output
    stages = [f"=== [{s}]" for s in ("data", "train", "eval-detector", "explain", "evaluate", "washing")]
    positions = [res.output.index(s) for s in stages]
    assert positions == sorted(positions)
    assert "pipeline finished" in res.output
