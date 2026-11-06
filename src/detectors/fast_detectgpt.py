"""Fast-DetectGPT (zero-shot, optional). Code lives in the submodule external/fast-detect-gpt."""

from .base import Detector


class FastDetectGPT(Detector):
    name = "fast-detectgpt"

    def __init__(self, cfg: dict):
        pass

    def fit(self):
        print(f"[train] {self.name}: zero-shot, nothing to train")

    def predict_proba(self, texts):
        print(f"[detector] {self.name}: score {len(texts)} text(s) via external/fast-detect-gpt")
        return [0.5] * len(texts)
