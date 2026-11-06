"""Random attribution control (null for deletion curves)."""

from .base import Explainer


class RandomExplainer(Explainer):
    name = "random"

    def __init__(self, n_draws: int = 10):
        self.n_draws = n_draws

    def explain(self, text):
        print(f"[explain] random: {self.n_draws} random attribution draws")
        return text.split(), [0.0] * len(text.split())
