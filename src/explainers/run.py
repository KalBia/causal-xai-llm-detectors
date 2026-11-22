"""Explain stage: compute attributions with every configured explainer and save them as JSONL."""

import json
from pathlib import Path

from . import build_explainers
from .base import Explainer

_DRY_RUN_TEXTS = ["example text to explain"]


def _load_texts(cfg: dict) -> list[str]:
    """Texts from the data split if present, else a dry-run sample (run `just data` first)."""

    explain_cfg = cfg.get("explain", {})
    split = explain_cfg.get("split", "test")
    n_samples = explain_cfg.get("n_samples", 100)

    path = Path(cfg.get("data", {}).get("dir", "data/")) / f"{split}.jsonl"
    if not path.exists():
        print(f"[explain] {path} not found --- dry-run on a sample text.")
        return _DRY_RUN_TEXTS

    texts = []
    with path.open() as f:
        for line in f:
            texts.append(json.loads(line).get("text", ""))
    texts = texts[:n_samples]
    print(f"[explain] loaded {len(texts)} texts from {path}")
    return texts


def run_explain(cfg: dict, detector) -> None:
    explain_cfg = cfg.get("explain", {})
    split = explain_cfg.get("split", "test")
    out_dir = Path(explain_cfg.get("output_dir", "results/explain"))
    texts = _load_texts(cfg)

    explainers = build_explainers(cfg, detector)
    if getattr(detector, "model", None) is None and hasattr(detector, "load"):
        try:
            detector.load()
            print("[explain] loaded detector checkpoint")
        except Exception as e:
            print(f"[explain] checkpoint not loadable ({e}); white-box explainers will be skipped")

    for name, explainer in explainers.items():
        out_path = out_dir / name / f"{split}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # store all draws only for explainers that override explain_draws (e.g. random)
        store_draws = type(explainer).explain_draws is not Explainer.explain_draws

        with out_path.open("w") as f:
            for i, text in enumerate(texts):
                tokens, attributions = explainer.explain(text)
                record = {
                    "id": i,
                    "text": text,
                    "tokens": tokens,
                    "attributions": [float(a) for a in attributions],
                }

                if store_draws:
                    record["draws"] = [[float(a) for a in d] for d in explainer.explain_draws(text)]
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[explain] {name}: saved {len(texts)} explanations -> {out_path}")
