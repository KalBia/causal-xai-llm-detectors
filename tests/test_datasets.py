"""Unit tests of the M4 split/save logic on synthetic data (no submodule needed)."""

import pandas as pd

from src.datasets.m4 import save_splits, split, undersample

CFG = {"split": {"train": 0.7, "val": 0.15, "test": 0.15}}


def _frame(n_human=200, n_machine=400):
    rows = []
    for src in ("a", "b"):
        for i in range(n_human):
            rows.append({"text": f"h-{src}-{i}", "label": 0, "source": src, "model": "human", "family": "human"})
        for fam in ("x", "y"):
            for i in range(n_machine // 2):
                rows.append({"text": f"m-{src}-{fam}-{i}", "label": 1, "source": src, "model": fam, "family": fam})
    return pd.DataFrame(rows)


def test_undersample_balances_classes_and_respects_cap():
    out = undersample(_frame(), max_samples=200, seed=1)
    assert len(out) <= 200
    assert out.groupby(["source", "label"]).size().nunique() == 1  # every (source, label) cell equal


def test_split_is_exhaustive_disjoint_and_stratified():
    df = _frame()
    splits = split(df, CFG, seed=1)
    assert sum(map(len, splits.values())) == len(df)
    train, val, test = (set(splits[k]["text"]) for k in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)
    # proportions per (source, family) cell are close to the requested ones
    for _, cell in splits["train"].groupby(["source", "family"]):
        full = df[(df["source"] == cell["source"].iloc[0]) & (df["family"] == cell["family"].iloc[0])]
        assert abs(len(cell) / len(full) - 0.7) < 0.02


def test_split_is_reproducible():
    a = split(_frame(), CFG, seed=3)["train"]
    b = split(_frame(), CFG, seed=3)["train"]
    assert a["text"].tolist() == b["text"].tolist()


def test_save_splits_roundtrip(tmp_path):
    splits = split(_frame(), CFG, seed=1)
    save_splits(splits, {"dir": str(tmp_path / "out")})
    for name, d in splits.items():
        back = pd.read_json(tmp_path / "out" / f"{name}.jsonl", lines=True)
        assert back["text"].tolist() == d["text"].tolist()
