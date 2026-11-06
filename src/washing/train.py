"""Explanation washing (RQ3). Skeleton: prints only."""


def train_washed(cfg: dict, detector):
    w = cfg["washing"]
    print(
        f"[washing] fine-tune copy of {detector.name}: KL(logits) + {w['lambda_expl']} * attribution penalty "
        f"(target: {w['target_tokens']})"
    )
    return detector
