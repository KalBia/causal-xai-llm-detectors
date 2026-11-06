"""RQ1 (faithfulness baseline): are top-K tokens of SHAP / IG more important than random ones?"""

from src.experiments.faithfulness import run_faithfulness


def run_rq1(cfg: dict, detector) -> None:
    print("[rq1] faithfulness baseline: deletion, all explainers")
    run_faithfulness(cfg, detector, intervention_names=["deletion"])
