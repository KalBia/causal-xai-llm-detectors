"""Explainer factory."""

from .base import Explainer
from .integrated_gradients import IntegratedGradientsExplainer
from .random_baseline import RandomExplainer
from .shap_explainer import ShapExplainer


def build_explainers(cfg: dict, detector, names: list[str] | None = None) -> dict[str, Explainer]:
    avail = {
        "shap": lambda: ShapExplainer(detector),
        "integrated_gradients": lambda: IntegratedGradientsExplainer(detector),
        "random": lambda: RandomExplainer(cfg["random_explainer"]["n_draws"]),
    }
    return {n: avail[n]() for n in names or cfg["explainers"]}
