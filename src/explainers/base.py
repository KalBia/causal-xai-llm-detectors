"""Explainer interface: one signed attribution per token."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class Explainer(ABC):
    name: str

    @abstractmethod
    def explain(self, text: str) -> tuple[list[str], NDArray[np.float64]]:
        """Return (tokens, attributions) with len(tokens) == len(attributions).

        Tokens are WORD-LEVEL (consistent with text.split()) so that src/interventions
        can apply edits by token index.
        Positive attribution = evidence for the 'AI-generated' class.
        """

    def explain_draws(self, text: str) -> list[NDArray[np.float64]]:
        """Multiple attribution vectors for one text (null baseline / confidence intervals).

        Default: a single draw. Tokens are the same as in explain(text).
        """
        return [self.explain(text)[1]]

@runtime_checkable
class WhiteBoxDetector(Protocol):
    """Detector with gradient access (e.g. RoBERTa). Fast-DetectGPT does NOT satisfy this.

    SHAP only needs predict_proba (black-box). Integrated Gradients additionally needs the
    tokenizer + model + device to backpropagate through embeddings.
    """

    name: str
    tokenizer: object # Hugging Face tokenizer
    model: object
    device: str

    def predict_proba(self, texts: list[str]) -> list[float]: ...
