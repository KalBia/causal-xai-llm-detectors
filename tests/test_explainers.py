import numpy as np

from src.explainers import build_explainers
from src.explainers.base import WhiteBoxDetector
from src.explainers.random_baseline import RandomExplainer


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
