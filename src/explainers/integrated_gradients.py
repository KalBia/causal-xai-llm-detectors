"""Integrated Gradients (gradient-based, white-box) via inseq / Captum. Skeleton: prints only."""

from .base import Explainer


class IntegratedGradientsExplainer(Explainer):
    name = "integrated_gradients"

    def __init__(self, detector):
        self.detector = detector

    def explain(self, text):
        print(f"[explain] integrated_gradients: IG on {self.detector.name}")
        return text.split(), [0.0] * len(text.split())
