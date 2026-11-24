"""Random attribution control (null baseline for deletion curves)."""

import hashlib

import numpy as np
from numpy.typing import NDArray

from .base import Explainer


class RandomExplainer(Explainer):
    """Null explainer: n_draws random attribution vectors per text.

    Deterministic per (seed, text), so results are reproducible.
    """

    name = "random"

    def __init__(self, n_draws: int = 10, seed: int = 17):
        self.n_draws = n_draws
        self.seed = seed

    def explain(self, text: str) -> tuple[list[str], NDArray[np.float64]]:
        tokens, draws = self._draws(text)
        return tokens, draws[0]

    def explain_draws(self, text: str) -> list[NDArray[np.float64]]:
        return self._draws(text)[1]

# -----------------------------------------------------------------------------------------------

    def _rng_for(self, text: str) -> np.random.Generator:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return np.random.default_rng(self.seed * 2**32 + int.from_bytes(digest[:4], "little"))

    def _draws(self, text: str) -> tuple[list[str], list[NDArray[np.float64]]]:
        tokens = text.split()
        rng = self._rng_for(text)
        draws = [rng.standard_normal(len(tokens)).astype(np.float64) for _ in range(self.n_draws)]
        return tokens, draws
