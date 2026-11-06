"""RQ2 (operator dependence): does faithfulness change with deletion vs replacement?"""

from src.experiments.faithfulness import run_faithfulness


def run_rq2(cfg: dict, detector) -> None:
    print("[rq2] operator comparison: all interventions x all explainers")
    run_faithfulness(cfg, detector)
