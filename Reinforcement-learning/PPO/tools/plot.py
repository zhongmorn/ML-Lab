"""Train / Eval / Loss：训练轮次、固定三种子阴影与最终逐回合评估。"""

from __future__ import annotations

import argparse
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
    if not path.is_file():
        raise FileNotFoundError(f"找不到日志：{path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} 缺少列：{', '.join(sorted(missing))}")
        columns = {name: [] for name in required + tuple(name for name in optional if name in reader.fieldnames)}
        for line, row in enumerate(reader, 2):
            if not any(value and value.strip() for value in row.values()):
                continue
            for name in columns:
                raw = row[name]
                if name in empty and (raw is None or not raw.strip()):
                    columns[name].append(np.nan)
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path.name} 第{line}行 {name} 不是数值") from error
                if not np.isfinite(value):
                    raise ValueError(f"{path.name} 第{line}行 {name} 不是有限值")
                columns[name].append(value)
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


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
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window 必须为正整数")
    run_dir = Path(run_dir)
    log_dir = run_dir / "logs" if (run_dir / "logs").is_dir() else run_dir
    round_path = log_dir / "training_rounds.csv"
    round_protocol = round_path.is_file()
    if round_protocol:
        rounds = _read_csv(round_path, (
            "round", "seed_1", "return_1", "seed_2", "return_2", "seed_3", "return_3",
            "mean_return", "std_return"))
        data = _read_csv(log_dir / "updates.csv", ("round", "policy_loss", "value_loss"))
        train_x, updates = rounds["round"], data["round"]
        _positive_index(train_x, "训练轮次")
        _positive_index(updates, "损失轮次")
        if not np.array_equal(train_x, updates):
            raise ValueError("training_rounds.csv 与 updates.csv 的训练轮次不一致")
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
    eval_episodes, eval_returns = _evaluation_returns(run_dir, log_dir)
    destination = Path(output) if output is not None else run_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / f"{name}.png" for name in ("training_reward", "evaluation_reward", "loss")}

    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        try:
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
        finally:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        try:
            if len(eval_returns):
                ax.plot(eval_episodes, eval_returns, "o-", color="#d36b2c", markersize=4, linewidth=1.5)
            else:
                _empty(ax)
            _format_axis(ax, "Eval", "Episode", "Return")
            fig.savefig(paths["evaluation_reward"], dpi=160)
        finally:
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
        try:
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
        finally:
            plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Train、Eval、Loss 三张图")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--window", type=int, default=20, help="仅用于旧日志的训练回报平滑窗口")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path)
    args = parser.parse_args()
    try:
        paths = plot_training(args.run_dir, args.window, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"绘图失败：{error}\n")
    for name, path in paths.items():
        print(f"Saved {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
