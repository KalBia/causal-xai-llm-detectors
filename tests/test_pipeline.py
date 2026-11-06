"""Smoke test: every stage of the skeleton runs, in order."""

from typer.testing import CliRunner

from src.cli import app


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
