import math

import numpy as np

from src.explainers import build_explainers
from src.explainers.base import WhiteBoxDetector
from src.explainers.random_baseline import RandomExplainer
from src.explainers.shap_explainer import ShapExplainer


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
