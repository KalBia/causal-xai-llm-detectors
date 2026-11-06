"""Replacement with a realistic substitute (synonym / masked-LM). Skeleton: prints only."""

from .base import Intervention


class Replacement(Intervention):
    name = "replacement"

    def apply(self, tokens, indices):
        print(f"[intervention] replacement: substitute tokens at {indices}")
        return " ".join(tokens)
