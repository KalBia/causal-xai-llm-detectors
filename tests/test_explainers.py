import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.explainers import build_explainers
from src.explainers.base import WhiteBoxDetector
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.explainers.random_baseline import RandomExplainer
from src.explainers.shap_explainer import ShapExplainer


def test_factory_skips_ig_for_blackbox_detector():
    cfg = {"explainers": ["shap", "integrated_gradients", "random"], "random_explainer": {"n_draws": 2}, "seed": 1}
    explainers = build_explainers(cfg, detector=KeywordDetector())
    assert set(explainers) == {"shap", "random"}


def test_random_contract_and_determinism():
    exp = RandomExplainer(n_draws=10, seed=42)
    text = "the quick brown fox jumps"
    tokens, attrs = exp.explain(text)
    assert tokens == text.split()
    assert len(attrs) == len(tokens)
    assert attrs.dtype == np.float64
    _, again = exp.explain(text)
    np.testing.assert_array_equal(attrs, again)  # same seed + text -> same values
    _, other = RandomExplainer(n_draws=10, seed=7).explain(text)
    assert not np.allclose(attrs, other)


def test_random_draws():
    exp = RandomExplainer(n_draws=5, seed=42)
    draws = exp.explain_draws("a b c d")
    assert len(draws) == 5
    assert all(len(d) == 4 for d in draws)
    assert len({tuple(d) for d in draws}) == 5  # all draws distinct
    _, first = exp.explain("a b c d")
    np.testing.assert_array_equal(first, draws[0])  # explain == first draw


def test_factory_uses_config():
    cfg = {"explainers": ["random"], "random_explainer": {"n_draws": 3}, "seed": 1}
    explainers = build_explainers(cfg, detector=None)  # only "random" is constructed
    assert set(explainers) == {"random"}
    assert len(explainers["random"].explain_draws("x y")) == 3


def test_whitebox_protocol():
    class DummyWhiteBox:
        name = "dummy"
        tokenizer = object()
        model = object()
        device = "cpu"

        def predict_proba(self, texts):
            return [0.5] * len(texts)

    assert isinstance(DummyWhiteBox(), WhiteBoxDetector)
    assert not isinstance(object(), WhiteBoxDetector)
    assert not isinstance(RandomExplainer(), WhiteBoxDetector)


class KeywordDetector:
    """Black-box dummy: P(AI) = sigmoid(count of 'great')."""

    name = "keyword-dummy"

    def predict_proba(self, texts):
        return [1.0 / (1.0 + math.exp(-text.lower().count("great"))) for text in texts]


def test_shap_word_level_and_additivity():
    det = KeywordDetector()
    exp = ShapExplainer(det, mask_token="[MASK]", max_evals=200)
    text = "this is great stuff today"  # 5 words -> 2^5 subsets, exact within max_evals
    tokens, values = exp.explain(text)
    assert tokens == text.split()
    assert values.shape == (5,)
    f_full = det.predict_proba([text])[0]
    f_all_masked = det.predict_proba(["[MASK]"])[0] # masker collapses consecutive masks to one [MASK]
    # SHAP additivity: attributions sum to f(x) - f(baseline)
    np.testing.assert_allclose(values.sum(), f_full - f_all_masked, atol=1e-6)


def test_shap_sign_convention():
    exp = ShapExplainer(KeywordDetector(), max_evals=200)
    tokens, values = exp.explain("this is great stuff today")
    assert values[tokens.index("great")] > 0  # the positive keyword gets positive attribution


class _TinyTokenizer:
    """Whitespace tokenizer with the HF-subset API the IG explainer uses."""

    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        tokens = text.split()
        ids, spans = [], []
        pos = 0
        for tok in tokens:
            start = text.index(tok, pos)
            spans.append((start, start + len(tok)))
            pos = start + len(tok)
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = spans
        if return_tensors == "pt":
            result = {k: torch.tensor([v]) for k, v in result.items()}
        return result


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab_size=64, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim, padding_idx=0)
        self.head = torch.nn.Linear(dim, 2)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.emb(input_ids).mean(dim=1)))


class TinyWhiteBoxDetector:
    name = "tiny"
    device = "cpu"

    def __init__(self):
        torch.manual_seed(0)
        self.tokenizer = _TinyTokenizer()
        self.model = _TinyModel()

    def predict_proba(self, texts):
        with torch.no_grad():
            ids = torch.tensor([[self.tokenizer.vocab.get(t, 0) for t in x.split()] for x in texts])
            logits = self.model(input_ids=ids).logits
        return torch.softmax(logits, dim=-1)[:, 1].tolist()


def test_ig_requires_whitebox():
    class NotLoaded:
        name = "not-loaded"
        tokenizer = None
        model = None

        def predict_proba(self, texts):
            return [0.5] * len(texts)

    with pytest.raises(TypeError, match="white-box"):
        IntegratedGradientsExplainer(KeywordDetector())  # black-box dummy from the SHAP tests
    with pytest.raises(TypeError, match="white-box"):
        IntegratedGradientsExplainer(NotLoaded())  # protocol attrs present, but model not loaded


def test_ig_completeness_and_word_level():
    det = TinyWhiteBoxDetector()
    exp = IntegratedGradientsExplainer(det, n_steps=32)
    text = "this is great stuff today"
    words, attrs = exp.explain(text)
    assert words == text.split()
    assert attrs.shape == (5,)
    # IG completeness: attributions sum to f(x) - f(baseline); the tiny model is
    # linear in the embeddings, so the n_steps approximation is exact
    enc = det.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        f_x = det.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0, 1].item()
        f_base = det.model(input_ids=torch.zeros_like(enc["input_ids"]),
            attention_mask=enc["attention_mask"]).logits[0, 1].item()
    np.testing.assert_allclose(attrs.sum(), f_x - f_base, atol=1e-4)


def test_run_explain_dry_run_writes_jsonl(tmp_path):
    from src.explainers.run import run_explain

    cfg = {
        "data": {"dir": str(tmp_path / "data")},  # missing -> dry-run fallback
        "explainers": ["random"],
        "random_explainer": {"n_draws": 3},
        "seed": 1,
        "explain": {"split": "test", "n_samples": 10, "output_dir": str(tmp_path / "out")},
    }
    run_explain(cfg, detector=None)
    lines = (tmp_path / "out" / "random" / "test.jsonl").read_text().splitlines()
    assert len(lines) == 1  # one dry-run sample text
    rec = json.loads(lines[0])
    assert rec["tokens"] == "example text to explain".split()
    assert len(rec["attributions"]) == len(rec["tokens"])
    assert len(rec["draws"]) == 3  # random overrides explain_draws -> draws stored


def test_run_explain_loads_split_and_caps_n_samples(tmp_path):
    from src.explainers.run import run_explain

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text('{"text": "hello world foo"}\n{"text": "another sample text"}\n')
    cfg = {
        "data": {"dir": str(data_dir)},
        "explainers": ["random"],
        "random_explainer": {"n_draws": 2},
        "seed": 1,
        "explain": {"split": "test", "n_samples": 1, "output_dir": str(tmp_path / "out")},
    }
    run_explain(cfg, detector=None)
    lines = (tmp_path / "out" / "random" / "test.jsonl").read_text().splitlines()
    assert len(lines) == 1  # n_samples=1 caps the split
    assert json.loads(lines[0])["text"] == "hello world foo"
