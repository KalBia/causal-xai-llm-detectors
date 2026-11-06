"""Shared experiment step: explainer x intervention -> faithfulness metrics. Skeleton: prints only."""

from src.explainers import build_explainers
from src.interventions import build_interventions


def run_faithfulness(cfg: dict, detector, explainer_names=None, intervention_names=None) -> None:
    explainers = build_explainers(cfg, detector, explainer_names)
    interventions = build_interventions(cfg, intervention_names)
    for iname, intervention in interventions.items():
        for ename, explainer in explainers.items():
            tokens, _ = explainer.explain("example text to explain")
            new_text = intervention.apply(tokens, [0])
            detector.predict_proba([new_text])
            print(
                f"[faithfulness] op={iname} expl={ename} top_k={cfg['top_k']}: "
                "comprehensiveness / sufficiency / deletion AUC"
            )
