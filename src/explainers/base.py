"""Explainer interface: one signed attribution per token."""

from abc import ABC, abstractmethod


class Explainer(ABC):
    name: str

    @abstractmethod
    def explain(self, text: str) -> tuple[list[str], "np.ndarray"]:  # noqa: F821
        """Return (tokens, attributions) with len(tokens) == len(attributions).

        Positive attribution = evidence for the 'AI-generated' class.
        """
