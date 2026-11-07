"""M4 (English subset): load -> preprocess -> split -> save. Skeleton: prints only."""
import json
import re
from pathlib import Path
import pandas as pd

M4_DIR = Path("external/M4/data")
EN_SOURCES = ["wikipedia", "reddit", "wikihow", "arxiv"]
# peerread excluded: human_text/machine_text are lists of reviews, not plain text
# Generators excluded: bloomz echoes the prompt inside machine_text (a detector would learn the prompt, not the style).
# Can be re-added later (using machine_abstract / machine_answer) if we run out of data.
EXCLUDED_MODELS = {"bloomz"}
HUMAN_KEYS = ("human_text", "text", "abstract")
PROMPT_LIKE = re.compile(r"\s*(Write|Please write|I will ask you)\b")  # leftover prompt echoes (e.g. wikihow_dolly2)


def load_m4(data_config: dict) -> pd.DataFrame:
    print(f"[dataset pipeline] 1/4 load {data_config['name']} ({data_config['language']}) from external/M4")

    rows = []
    for path in sorted(M4_DIR.glob("*.jsonl")):
        src, _, model_name = path.stem.lower().partition("_")
        if src not in EN_SOURCES or model_name in EXCLUDED_MODELS:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue

                r = json.loads(line)
                human = next((r[k] for k in HUMAN_KEYS if r.get(k)), None)
                machine = r.get("machine_text")
                if not isinstance(human, str) or not isinstance(machine, str) or PROMPT_LIKE.match(machine):
                    continue

                model = r.get("model") or model_name

                rows.append({"text": human, "label": 0, "source": src, "model": "human"})
                rows.append({"text": machine, "label": 1, "source": src, "model": model})
    return pd.DataFrame(rows)


def preprocess(data_config: dict):
    print(f"[dataset pipeline] 2/4 preprocess: clean text, drop duplicates, subsample to {data_config['max_samples']}")


def split(data_config: dict):
    print(f"[dataset pipeline] 3/4 stratified split {data_config['split']}")


def save_splits(data_config: dict):
    print(f"[dataset pipeline] 4/4 save train/val/test to {data_config['dir']}")


def prepare_data(cfg: dict) -> None:
    data_config = cfg["data"]
    loaded_dataset = load_m4(data_config)
    print(type(loaded_dataset))
    # preprocess(data_config)
    # split(data_config)
    # save_splits(data_config)


if __name__ == 'main':
    cfg = 'configs/default.yaml'
    data_config = cfg["data"]
    loaded_dataset = load_m4(data_config)
    # print(type(loaded_dataset))
