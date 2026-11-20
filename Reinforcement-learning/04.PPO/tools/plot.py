"""Train / Eval / Loss：训练轮次、固定三种子阴影与最终逐回合评估。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def _read_csv(path: Path, required: tuple[str, ...], empty: tuple[str, ...] = (),
              optional: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = required + tuple(name for name in optional if name in reader.fieldnames)
        columns = {name: [] for name in names}
        for row in reader:
            if not any(value and value.strip() for value in row.values()):
                continue
            for name in columns:
                raw = row[name]
                empty_value = name in empty and (raw is None or not raw.strip())
                columns[name].append(np.nan if empty_value else float(raw))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _columns(rows: list[dict], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.asarray([row[name] for row in rows], dtype=float) for name in names}


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (prefix[end] - prefix[start]) / (end - start)


def _evaluation_returns(run_dir: Path, log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = log_dir / "evaluation_episodes.csv"
    if path.is_file():
        data = _read_csv(path, ("episode", "episode_return"))
        return data["episode"], data["episode_return"]
    # 历史实验仍可使用真实的逐回合结果，不能把监控评估次数当回合。
    summary = run_dir / "summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        returns = np.asarray(data.get("final_evaluation", {}).get("returns", []), dtype=float)
        if returns.ndim != 1 or not np.isfinite(returns).all():
            raise ValueError("summary.json 的逐回合 returns 无效")
        return np.arange(1, len(returns) + 1), returns
    return np.array([]), np.array([])


def _format_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)


def _empty(ax: plt.Axes) -> None:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)


def _positive_index(values: np.ndarray, name: str) -> None:
    if np.any(values <= 0) or np.any(values != np.floor(values)) or np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} 必须为严格递增的正整数")


def plot_training(run_dir: str | Path, window: int = 20,
                  output: str | Path | None = None) -> dict[str, Path]:
    """新实验逐轮画固定三种子回报的均值与标准差；旧日志保留原坐标。"""
    run_dir = Path(run_dir)
    log_dir = run_dir / "logs" if (run_dir / "logs").is_dir() else run_dir
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else None
    round_path = log_dir / "training_rounds.csv"
    round_protocol = results is not None or round_path.is_file()
    if round_protocol:
        round_columns = ("round", "seed_1", "return_1", "seed_2", "return_2", "seed_3", "return_3",
                         "mean_return", "std_return")
        update_columns = ("round", "policy_loss", "value_loss")
        if results is not None:
            rounds = _columns(results["training"], round_columns)
            data = _columns(results["updates"], update_columns)
        else:
            rounds = _read_csv(round_path, round_columns)
            data = _read_csv(log_dir / "updates.csv", update_columns)
        train_x, updates = rounds["round"], data["round"]
        _positive_index(train_x, "训练轮次")
        _positive_index(updates, "损失轮次")
        if not np.array_equal(train_x, updates):
            raise ValueError("训练记录与更新记录的轮次不一致")
        seeds = np.column_stack([rounds[f"seed_{i}"] for i in (1, 2, 3)])
        if len(seeds) and (np.any(seeds != seeds[0]) or len(np.unique(seeds[0])) != 3):
            raise ValueError("每轮必须使用相同的三个不同评估种子")
        returns = np.column_stack([rounds[f"return_{i}"] for i in (1, 2, 3)])
        train_return = returns.mean(axis=1)
        train_std = returns.std(axis=1)
        if not (np.allclose(train_return, rounds["mean_return"]) and
                np.allclose(train_std, rounds["std_return"])):
            raise ValueError("训练均值/标准差必须来自该轮三个种子的真实回报")
        xlabel = "Round"
    else:
        data = _read_csv(log_dir / "updates.csv",
                         ("optimizer_steps", "recent_return_mean", "policy_loss", "value_loss"),
                         empty=("recent_return_mean",), optional=("gradient_step",))
        optimizer_steps = data["optimizer_steps"]
        if np.any(optimizer_steps <= 0) or np.any(optimizer_steps != np.floor(optimizer_steps)):
            raise ValueError("optimizer_steps 必须记录真实的正整数更新次数")
        updates = data.get("gradient_step", np.cumsum(optimizer_steps))
        _positive_index(updates, "累计梯度更新步数")
        mask = np.isfinite(data["recent_return_mean"])
        train_x, train_return = updates[mask], data["recent_return_mean"][mask]
        train_std = None
        xlabel = "Update"
    if results is not None:
        eval_returns = np.asarray(results["evaluation"]["returns"], dtype=float)
        eval_episodes = np.arange(1, len(eval_returns) + 1)
    else:
        eval_episodes, eval_returns = _evaluation_returns(run_dir, log_dir)
    # 图片与训练日志一起保存在当前实验目录，显式指定时使用目标目录。
    destination = run_dir if output is None else Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_reward": destination / "training.png",
        "evaluation_reward": destination / "evaluation.png",
        "loss": destination / "loss.png",
    }

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        if len(train_return):
            if round_protocol:
                ax.plot(train_x, train_return, color="#087f8c", linewidth=1.5)
                ax.fill_between(train_x, train_return - train_std, train_return + train_std,
                                color="#087f8c", alpha=0.2, linewidth=0)
            else:
                ax.plot(train_x, train_return, color="#94a3b8", alpha=0.45, linewidth=0.8)
                ax.plot(train_x, _rolling_mean(train_return, window), color="#087f8c", linewidth=1.7)
        else:
            _empty(ax)
        _format_axis(ax, "Train", xlabel, "Return")
        fig.savefig(paths["training_reward"], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        if len(eval_returns):
            ax.plot(eval_episodes, eval_returns, "o-", color="#d36b2c", markersize=4, linewidth=1.5)
        else:
            _empty(ax)
        _format_axis(ax, "Eval", "Episode", "Return")
        fig.savefig(paths["evaluation_reward"], dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        _format_axis(ax, "Loss", xlabel, "Loss")
        if len(updates):
            actor, = ax.plot(updates, data["policy_loss"], color="#087f8c", label="Actor (left)")
            critic_axis = ax.twinx()
            critic_axis.set_ylabel("Loss")
            critic_axis.spines["right"].set_visible(True)
            critic, = critic_axis.plot(updates, data["value_loss"], color="#d36b2c", label="Critic (right)")
            ax.set_zorder(critic_axis.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.legend(handles=[actor, critic], fontsize=9)
        else:
            _empty(ax)
        fig.savefig(paths["loss"], dpi=160)
        plt.close(fig)
    return paths
