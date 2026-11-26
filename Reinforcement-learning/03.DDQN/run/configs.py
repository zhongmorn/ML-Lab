import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_config(env_name=None):
    values = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
    if env_name is not None and env_name != values["env_name"]:
        raise KeyError(env_name)
    values["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SimpleNamespace(**values)
