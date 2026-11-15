"""JSON、CSV 日志；模型 checkpoint 由 PPO.save/load 管理。"""

import csv
import json
from pathlib import Path


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_evaluation_episodes(path: str | Path, result: dict, checkpoint: str) -> None:
    """保存一次策略评估的逐回合回报，供按Episode绘图。"""
    returns, lengths = result["returns"], result["lengths"]
    if len(returns) != len(lengths) or len(returns) != result["episodes"]:
        raise ValueError("Evaluation returns and lengths must match the episode count")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "episode_return", "episode_length", "seed", "checkpoint"])
        writer.writeheader()
        for i, (episode_return, length) in enumerate(zip(returns, lengths), 1):
            writer.writerow(dict(episode=i, episode_return=episode_return, episode_length=length,
                                 seed=result.get("seeds", range(result["seed"], result["seed"] + len(returns)))[i - 1],
                                 checkpoint=checkpoint))



class CSVLogger:
    """逐行落盘，训练中断时仍保留已经写出的指标。"""

    def __init__(self, path: Path, fields: list[str]):
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

