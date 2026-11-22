"""Integrated Gradients (gradient-based, white-box) via Captum. RoBERTa only."""

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from numpy.typing import NDArray

from .base import Explainer, WhiteBoxDetector


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Char span (start, end) of each whitespace-separated word."""
    spans = []
    pos = 0
    for word in text.split():
        start = text.index(word, pos)
        spans.append((start, start + len (word)))
        pos = start + len(word)
    return spans


class IntegratedGradientsExplainer(Explainer):
    """Integrated Gradients over the input embedding layer. Subword attributions summed into words."""

    name = "integrated_gradients"

    def __init__(self, detector, n_steps: int = 50, baseline: str = "pad"):
        if not isinstance(detector, WhiteBoxDetector):
            raise TypeError(
                "integrated_gradients requires a white-box detector (e.g. RoBERTa), "
                f"got {type(detector).__name__}"
            )

        if baseline != "pad":
            raise ValueError(f"unsupported baseline: {baseline!r}")

        self.detector = detector
        self.n_steps = n_steps
        self._lig = LayerIntegratedGradients(self._forward, detector.model.get_input_embeddings())

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.detector.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits[:,1] # logit of the "AI-generated" class

    def explain(self, text: str) -> tuple[list[str], NDArray[np.float64]]:
        tokenizer = self.detector.tokenizer
        enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = enc["input_ids"].to(self.detector.device)
        attention_mask = enc["attention_mask"].to(self.detector.device)
        offsets = enc["offset_mapping"][0].tolist()
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
        attrs = self._lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=attention_mask,
            n_steps=self.n_steps,
        )
        token_attrs = attrs.sum(dim=-1)[0].detach().cpu().numpy()  # one value per subword
        words = text.split()
        spans = _word_spans(text)
        word_attrs = np.zeros(len(words), dtype=np.float64)
        for (start, end), attr in zip(offsets, token_attrs, strict=False):
            if end <= start:  # special tokens like <s> have (0, 0) offsets
                continue
            for i, (ws, we) in enumerate(spans):
                if start < we and end > ws:  # subword overlaps this word
                    word_attrs[i] += attr
                    break
        return words, word_attrs
