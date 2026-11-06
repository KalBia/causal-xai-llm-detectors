"""Faithfulness metrics (ERASER, DeYoung et al. 2020) and detector sanity metrics. Skeleton: signatures only."""


def comprehensiveness(p_full: float, p_removed: float) -> float:
    """p(full) - p(text without top-K tokens). High = tokens were necessary."""
    ...


def sufficiency(p_full: float, p_only_topk: float) -> float:
    """p(full) - p(only top-K tokens). Low = tokens alone suffice."""
    ...


def deletion_auc(ks: list[float], probs: list[float]) -> float:
    """Area under the deletion curve (prob of predicted class vs. K)."""
    ...


def macro_f1(y_true, y_pred) -> float: ...


def auroc(y_true, scores) -> float: ...
