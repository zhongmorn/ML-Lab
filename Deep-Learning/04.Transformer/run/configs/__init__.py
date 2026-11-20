from pathlib import Path


# 模型宽度、注意力头数与上下文长度。
MODEL_CONFIG = {
    "width": 48,
    "num_heads": 1,
    "max_length": 64,
}

# 训练预算、批大小、学习率及复现实验和日志设置。
DEFAULT_STEPS = 1200
BATCH_SIZE = 32
LEARNING_RATE = 0.003
DEFAULT_SEED = 42
CPU_THREADS = 1
LOG_INTERVAL = 200
LOSS_AVERAGE_WINDOW = 100

# 生成长度上限与默认数据、权重路径。
MAX_NEW_TOKENS = 48

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = PROJECT_ROOT / "data" / "train.txt"
EVAL_DATA = PROJECT_ROOT / "data" / "eval.txt"
CHECKPOINT = PROJECT_ROOT / "output" / "weights.pt"
