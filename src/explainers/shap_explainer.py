"""SHAP (Partition Explainer): perturbation-based, black-box. Skeleton: prints only."""

from .base import Explainer


class ShapExplainer(Explainer):
    name = "shap"

    def __init__(self, detector):
        self.detector = detector

    def explain(self, text):
        print(f"[explain] shap: Partition Explainer on {self.detector.name}")
        return text.split(), [0.0] * len(text.split())
