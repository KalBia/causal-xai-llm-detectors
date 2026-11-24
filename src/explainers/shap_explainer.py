"""SHAP (Partition Explainer): perturbation-based, black-box, word-level."""

import numpy as np
from numpy.typing import NDArray
from shap import maskers
from shap.explainers import PartitionExplainer

from .base import Explainer


class _WordTokenizer:
    """Minimal HF-style tokenizer for shap.maskers.Text: whitespace, word-level tokens.

    shap's Text masker requires the HF PreTrainedTokenizer subset: __call__ returns
    a dict with "input_ids" and, when asked, "offset_mapping" (char span per token).
    """

    def __call__(self, text: str, return_offsets_mapping: bool = False) -> dict:
        tokens = text.split()
        spans = []
        pos = 0
        for token in tokens:
            start = text.index(token, pos)
            spans.append((start, start + len(token)))
            pos = start + len(token)
        result = {"input_ids": list(range(len(tokens)))}

        if return_offsets_mapping:
            result["offset_mapping"] = spans

        return result


class ShapExplainer(Explainer):
    """Word-level SHAP Partition Explainer over detector.predict_proba = P(AI)."""

    name = "shap"

    def __init__(self, detector, mask_token: str = "[MASK]", max_evals: int = 500):
        self.detector = detector
        masker = maskers.Text(_WordTokenizer(), mask_token=mask_token)
        self._explainer = PartitionExplainer(self._model_fn, masker)
        self.max_evals = max_evals

    def _model_fn(self, texts) -> NDArray[np.float64]:
        # shap hands us masked texts as strings, score each with the detector
        return np.asarray(self.detector.predict_proba([str(t) for t in texts]), dtype=np.float64)

    def explain(self, text: str) -> tuple[list[str], NDArray[np.float64]]:
        tokens = text.split()
        sv = self._explainer([text], max_evals=self.max_evals, silent=True)
        values = np.asarray(sv.values, dtype=np.float64).reshape(-1)
        assert values.shape[0] == len(tokens), f"shap returned {values.shape[0]} values for {len(tokens)} tokens"
        return tokens, values
