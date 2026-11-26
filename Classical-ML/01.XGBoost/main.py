import json
from pathlib import Path

from run.model import run_experiment


if __name__ == "__main__":
    # 从配置文件读取参数，结果保存在算法目录中。
    root = Path(__file__).resolve().parent
    config = json.loads((root / "run/configs.json").read_text(encoding="utf-8"))
    run_experiment(config, root / "output")
