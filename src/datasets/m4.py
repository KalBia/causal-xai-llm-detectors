"""M4 (English subset): load -> preprocess -> split -> save. Skeleton: prints only."""


def load_m4(d):
    print(f"[data] 1/4 load {d['name']} ({d['language']}) from external/M4 (git submodule)")


def preprocess(d):
    print(f"[data] 2/4 preprocess: clean text, drop duplicates, subsample to {d['max_samples']}")


def split(d):
    print(f"[data] 3/4 stratified split {d['split']}")


def save_splits(d):
    print(f"[data] 4/4 save train/val/test to {d['dir']}")


def prepare_data(cfg: dict) -> None:
    d = cfg["data"]
    load_m4(d)
    preprocess(d)
    split(d)
    save_splits(d)
