"""Explain stage: compute attributions with every configured explainer."""

from . import build_explainers


def run_explain(cfg: dict, detector) -> None:
    for explainer in build_explainers(cfg, detector).values():
        explainer.explain("example text to explain")
    print("[explain] save attributions")
