import json
from pathlib import Path
from types import SimpleNamespace

import torch

CONFIG_FILES = {
    "CartPole-v1": "cartpole.json",
}


def load_config(env_name):
    # 环境参数集中在对应的 JSON 文件。
    path = Path(__file__).with_name(CONFIG_FILES[env_name])
    values = json.loads(path.read_text(encoding="utf-8"))
    # 训练设备与模型使用同一个设置。
    values["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SimpleNamespace(**values)
