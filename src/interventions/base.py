"""Intervention interface: modify the top-K tokens of a text."""

from abc import ABC, abstractmethod


class Intervention(ABC):
    name: str

    @abstractmethod
    def apply(self, tokens: list[str], indices: list[int]) -> str:
        """Return the new text after intervening on `tokens[i]` for i in indices."""
