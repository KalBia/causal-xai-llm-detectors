"""Detector interface shared by RoBERTa and (optionally) Fast-DetectGPT."""

from abc import ABC, abstractmethod


class Detector(ABC):
    name: str

    @abstractmethod
    def fit(self) -> None: ...

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> list[float]:
        """P(AI-generated) per text."""
