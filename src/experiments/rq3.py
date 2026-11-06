"""RQ3 (explanation washing): same predictions, different explanations - do causal tests tell them apart?"""

from src.experiments.faithfulness import run_faithfulness
from src.washing.train import train_washed


def run_rq3(cfg: dict, detector) -> None:
    washed = train_washed(cfg, detector)
    print("[rq3] compare predictions and attributions: original vs washed")
    for label, det in (("original", detector), ("washed", washed)):
        print(f"[rq3] faithfulness of {label} detector")
        run_faithfulness(cfg, det, ["shap", "random"], ["deletion"])
