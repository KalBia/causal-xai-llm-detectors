"""Explainer factory."""

from .base import Explainer
from .integrated_gradients import IntegratedGradientsExplainer
from .random_baseline import RandomExplainer
from .shap_explainer import ShapExplainer


def build_explainers(cfg: dict, detector, names: list[str] | None = None) -> dict[str, Explainer]:
    avail = {
        "shap": lambda: ShapExplainer(detector, **cfg.get("shap", {})),
        "integrated_gradients": lambda: IntegratedGradientsExplainer(detector, **cfg.get("integrated_gradients", {})),
        "random": lambda: RandomExplainer(cfg["random_explainer"]["n_draws"], cfg.get("seed", 42)),
    }

    explainers = {}
    for n in names or cfg["explainers"]:
        try:
            explainers[n] = avail[n]()
        except TypeError as e:
            print(f"[explain] skipping {n}: {e}")  # e.g. IG before the detector checkpoint exists
    return explainers
