"""Intervention factory."""

from .base import Intervention
from .deletion import Deletion
from .replacement import Replacement


def build_interventions(cfg: dict, names: list[str] | None = None) -> dict[str, Intervention]:
    avail = {"deletion": Deletion, "replacement": Replacement}
    return {n: avail[n]() for n in names or cfg["interventions"]}
