import json

from src.utils.config import load_config


def test_load_yaml_and_json(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\n")
    (tmp_path / "a.json").write_text(json.dumps({"x": 1}))
    assert load_config(tmp_path / "a.yaml") == load_config(tmp_path / "a.json") == {"x": 1}
