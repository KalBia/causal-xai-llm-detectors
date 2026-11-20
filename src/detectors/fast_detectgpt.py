"""Fast-DetectGPT (zero-shot, optional): wrapper around the code in external/fast-detect-gpt.

Needs a GPU (about 16 GB), `just submodules` and the packages `datasets`, `matplotlib`, `scipy`, `scikit-learn`, `tqdm`.
The probabilities are approximate, because the calibration was not fitted on M4.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from .base import Detector

SCRIPTS = Path("external/fast-detect-gpt/scripts")


class FastDetectGPT(Detector):
    name = "fast-detectgpt"

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("fast_detectgpt", {})
        self.impl = None

    def fit(self):
        print(f"[train] {self.name}: zero-shot, nothing to train")

    def predict_proba(self, texts):
        if self.impl is None:
            import torch

            sys.path.insert(0, str(SCRIPTS))
            from local_infer import FastDetectGPT as Authors
            
            model = self.cfg.get("model", "gpt-neo-2.7B")  # sampling = scoring model (keys of the authors' table)
            self.impl = Authors(
                SimpleNamespace(
                    sampling_model_name=model,
                    scoring_model_name=model,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    cache_dir=self.cfg.get("cache_dir", str(Path.home() / ".cache" / "huggingface")),
                )
            )
        return [float(self.impl.compute_prob(t)[0]) for t in texts]
