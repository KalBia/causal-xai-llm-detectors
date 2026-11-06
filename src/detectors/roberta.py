"""RoBERTa-base fine-tuned on M4 (main detector). Skeleton: prints only."""

from .base import Detector


class RobertaDetector(Detector):
    name = "roberta-base"

    def __init__(self, cfg: dict):
        self.cfg = cfg["detector"]

    def fit(self):
        d = self.cfg
        print(f"[train] {self.name}: fine-tune on train/val, epochs={d['epochs']} bs={d['batch_size']} lr={d['lr']}")
        print(f"[train] save checkpoint -> {d['checkpoint_dir']}")

    def predict_proba(self, texts):
        print(f"[detector] {self.name}: predict P(AI) for {len(texts)} text(s)")
        return [0.5] * len(texts)
