import json
from pathlib import Path


def load_config(filename):
    return json.loads(Path(__file__).with_name(filename).read_text(encoding="utf-8"))


CONFIG = load_config("default.json")
