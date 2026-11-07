"""M4 (English subset): load -> preprocess -> split -> save. See README.md in this directory."""

import json
import re
from pathlib import Path

import pandas as pd


def model_family(model: str, families: dict) -> str:
    """Map a raw `model` string to its family by prefix; unknown names become their own family."""
    m = model.lower()
    return next((fam for prefix, fam in families.items() if m.startswith(prefix.lower())), m)


def load_m4(data_config: dict) -> pd.DataFrame:
    print(
        f"[dataset pipeline] 1/4 load {data_config['name']} ({data_config['language']}) from {data_config['raw_dir']}"
    )

    sources = {s.lower() for s in data_config["domain"]}
    excluded = {m.lower() for m in data_config.get("excluded_models", [])}
    human_keys = data_config["human_text_keys"]
    machine_key = data_config["machine_text_key"]
    families = data_config["model_families"]

    rows = []
    for path in sorted(Path(data_config["raw_dir"]).glob("*.jsonl")):
        src, _, model_name = path.stem.lower().partition("_")
        if src not in sources or model_name in excluded:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                r = json.loads(line)
                human = next((r[k] for k in human_keys if r.get(k)), None)
                machine = r.get(machine_key)
                model = r.get("model") or model_name

                rows.append({"text": human, "label": 0, "source": src, "model": "human", "family": "human"})
                family = model_family(model, families)
                rows.append({"text": machine, "label": 1, "source": src, "model": model, "family": family})
    return pd.DataFrame(rows)


def preprocess(df: pd.DataFrame, data_config: dict, seed: int = 42) -> pd.DataFrame:
    print(
        "[dataset pipeline] 2/4 preprocess: clean text, drop duplicates, balance classes, "
        f"cap at {data_config['max_samples']}"
    )
    n0 = len(df)

    # 1. drop rows without plain-text content (missing / list-valued fields)
    df = df[df["text"].map(lambda t: isinstance(t, str))].copy()

    # 2a. normalise whitespace, drop empty texts
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"] != ""]

    # 2b. strip leading separator junk (WikiHow human texts start with "; ,", ",", ...), then drop emptied texts
    df["text"] = df["text"].str.replace(data_config["strip_leading_regex"], "", regex=True).str.strip()
    df = df[df["text"] != ""]

    # 3a. drop machine texts that still start with an echoed prompt
    prompt_like = re.compile(r"\s*(" + "|".join(map(re.escape, data_config["prompt_prefixes"])) + r")\b")
    df = df[~((df["label"] == 1) & df["text"].str.match(prompt_like))]

    # 3b. drop too-short texts
    df = df[df["text"].str.split().str.len() >= data_config["min_words"]]
    n1 = len(df)

    # 4. texts that occur with both labels are ambiguous -> drop all their copies
    n_labels = df.groupby("text")["label"].transform("nunique")
    df = df[n_labels == 1]
    n2 = len(df)

    # 5. exact duplicates (e.g. the same human text repeated for every generator)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    n3 = len(df)

    # 6. undersample to balance classes (and sources), capped at max_samples
    df = undersample(df, data_config["max_samples"], seed)

    print(f"  rows: {n0} -> {n1} cleaned -> {n2} no label conflicts -> {n3} dedup -> {len(df)} balanced")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print(df[df["label"] == 1].groupby(["source", "family"]).size().unstack(fill_value=0))
    return df


def _sample_evenly(group: pd.DataFrame, n: int, by: str, seed: int) -> pd.DataFrame:
    """Draw n rows with (as far as possible) equal counts per `by` value; shortfalls are redistributed."""
    parts = {k: g for k, g in group.groupby(by)}
    take = dict.fromkeys(parts, 0)
    remaining = min(n, len(group))
    while remaining > 0:
        open_ = [k for k in parts if take[k] < len(parts[k])]
        share = max(remaining // len(open_), 1)
        for k in open_:
            add = min(share, len(parts[k]) - take[k], remaining)
            take[k] += add
            remaining -= add
            if remaining == 0:
                break
    return pd.concat([parts[k].sample(n=take[k], random_state=seed) for k in parts if take[k]])


def undersample(df: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    """Balanced dataset: per source, equal number of human (0) and machine (1) texts.

    Machine texts are spread evenly across generator families, sources contribute equally.
    Total size is at most `max_samples`.
    """
    sources = sorted(df["source"].unique())
    # per-source, per-class quota: limited by the scarcest class in the scarcest source and by max_samples
    per_class = min(df.groupby(["source", "label"]).size().min(), max_samples // (2 * len(sources)))

    parts = []
    for src in sources:
        sdf = df[df["source"] == src]
        parts.append(sdf[sdf["label"] == 0].sample(n=per_class, random_state=seed))
        parts.append(_sample_evenly(sdf[sdf["label"] == 1], per_class, by="family", seed=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def split(df: pd.DataFrame, data_config: dict, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Stratified split by (source, family); human texts are their own family.

    Split names and proportions come from `data.split` (e.g. train/val/test).

    Every (source, family) cell is shuffled and cut with the proportions from `data.split`, so label, domain
    and generator family have (almost) the same share in each split. Texts are unique after `preprocess`,
    hence no text can end up in two splits.
    """
    fractions = data_config["split"]  # {name: fraction}; names and their order come from the config
    assert len(fractions) >= 2, "data.split needs at least two splits"
    assert abs(sum(fractions.values()) - 1) < 1e-9, "data.split must sum to 1"
    print(f"[dataset pipeline] 3/4 stratified split {fractions} by (source, family)")

    parts = {name: [] for name in fractions}
    for _, cell in df.groupby(["source", "family"]):
        cell = cell.sample(frac=1, random_state=seed)
        start, cum = 0, 0.0
        for i, (name, frac) in enumerate(fractions.items()):
            cum += frac
            # the last split takes the remainder, so that no row is lost to rounding
            end = len(cell) if i == len(fractions) - 1 else round(len(cell) * cum)
            parts[name].append(cell.iloc[start:end])
            start = end

    splits = {k: pd.concat(v).sample(frac=1, random_state=seed).reset_index(drop=True) for k, v in parts.items()}

    assert sum(map(len, splits.values())) == len(df)
    seen: set[str] = set()
    for d in splits.values():
        texts = set(d["text"])
        assert not (seen & texts), "text leaked between splits"
        seen |= texts

    for name, d in splits.items():
        n_human = int((d["label"] == 0).sum())
        print(f"  {name:5} n={len(d):5}  human={n_human:5}  machine={len(d) - n_human:5}")
    return splits


def save_splits(splits: dict[str, pd.DataFrame], data_config: dict) -> None:
    out = Path(data_config["dir"])
    print(f"[dataset pipeline] 4/4 save train/val/test to {out}/")
    out.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        d.to_json(out / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)


def prepare_data(cfg: dict) -> None:
    data_config = cfg["data"]
    seed = cfg.get("seed", 42)

    loaded_dataset = load_m4(data_config)
    if loaded_dataset.empty:
        raise FileNotFoundError(
            f"no M4 records found in {data_config['raw_dir']} for sources {data_config['domain']} "
            "(run `just submodules`)"
        )

    dataset = preprocess(loaded_dataset, data_config, seed)
    splits = split(dataset, data_config, seed)
    save_splits(splits, data_config)
