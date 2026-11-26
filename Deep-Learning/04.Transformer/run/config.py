import json
from pathlib import Path


# 读取模型、训练和生成参数，各模块沿用同一份配置。
_config = json.loads(Path(__file__).with_name("configs.json").read_text(encoding="utf-8"))
MODEL_CONFIG = _config["model"]
DEFAULT_STEPS = _config["steps"]
BATCH_SIZE = _config["batch_size"]
LEARNING_RATE = _config["learning_rate"]
DEFAULT_SEED = _config["seed"]
CPU_THREADS = _config["cpu_threads"]
LOG_INTERVAL = _config["log_interval"]
LOSS_AVERAGE_WINDOW = _config["loss_average_window"]
MAX_NEW_TOKENS = _config["max_new_tokens"]

# 从实验目录定位训练语料、评估问题和检查点。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA = PROJECT_ROOT / "data" / "train.txt"
EVAL_DATA = PROJECT_ROOT / "data" / "eval.txt"
CHECKPOINT = PROJECT_ROOT / "output" / "weights.pt"
