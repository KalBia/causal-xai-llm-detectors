"""Detector factory + train / evaluate stages."""

from .base import Detector
from .fast_detectgpt import FastDetectGPT
from .roberta import RobertaDetector

REGISTRY = {"roberta-base": RobertaDetector, "fast-detectgpt": FastDetectGPT}


def build_detector(cfg: dict, name: str | None = None) -> Detector:
    return REGISTRY[name or cfg["detector"]["name"]](cfg)


def train_detector(cfg: dict) -> Detector:
    det = build_detector(cfg)
    det.fit()
    return det


def evaluate_detector(cfg: dict, split: str = "test") -> None:
    det = build_detector(cfg)
    print(f"[eval-detector] load checkpoint {cfg['detector']['checkpoint_dir']}")
    det.predict_proba(["example text"])
    print(f"[eval-detector] Macro-F1 / AUROC on {split} split")
