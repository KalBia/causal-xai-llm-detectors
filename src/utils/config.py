"""YAML / JSON config loading."""

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config format: {path.suffix}")
