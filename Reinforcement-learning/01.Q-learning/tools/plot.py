from collections import Counter

import matplotlib

# 使用无窗口后端，直接保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run.configs import CONFIG
from environment import background


def plot_learning_curve(rows, output_dir):
    episodes = [row["episode"] for row in rows]
    returns = [row["return"] for row in rows]
    successes = [row["success"] for row in rows]
    window = min(CONFIG["reporting"]["rolling_window"], len(rows))
    weights = np.ones(window) / window
    # 把每个滑动窗口的统计值画在窗口最后一个回合的位置。
    rolling_episodes = episodes[window - 1:]
    average_returns = np.convolve(returns, weights, mode="valid")
    success_rates = np.convolve(successes, weights, mode="valid") * 100

    fig, (returns_ax, success_ax) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, height_ratios=(1.5, 1), constrained_layout=True,
    )
    returns_ax.plot(episodes, returns, color="#82a6c8", alpha=0.24, linewidth=0.8, label="Episode return")
    returns_ax.plot(rolling_episodes, average_returns, color="#173e68", linewidth=1.8, label=f"{window}-episode mean")
    returns_ax.set(title="Q-learning training in the slippery maze", ylabel="Return")
    returns_ax.legend(loc="lower right", frameon=False)
    success_ax.plot(rolling_episodes, success_rates, color="#176c63", linewidth=1.8)
    success_ax.set(xlabel="Episode", ylabel="Success rate (%)", ylim=(0, 105))
    success_ax.set_title(f"{window}-episode moving window", loc="left", fontsize=10)
    for ax in (returns_ax, success_ax):
        ax.grid(axis="y", color="#e4eaf0")
        ax.set_axisbelow(True)
    fig.savefig(output_dir / "training.png", dpi=180)
    plt.close(fig)


def plot_greedy_path(path, output_dir):
    fig, ax = plt.subplots(figsize=(6.1, 6.1), constrained_layout=True)
    background.draw_background(ax, legend_title="Counts mark repeated positions")
    ax.set_title(f"Greedy evaluation route ({len(path) - 1} steps)", pad=12)

    # 实际走过的有向边各画一次，往返时让两个箭头错开。
    moves = dict.fromkeys((cell, next_cell) for cell, next_cell in zip(path, path[1:]) if cell != next_cell)
    for (row, col), (next_row, next_col) in moves:
        offset = 0.11 if ((next_row, next_col), (row, col)) in moves else 0.0
        dx = (row - next_row) * offset
        dy = (next_col - col) * offset
        ax.annotate(
            "", xy=(next_col + dx, next_row + dy), xytext=(col + dx, row + dy),
            arrowprops={"arrowstyle": "-|>", "color": "#176c63", "lw": 1.8, "shrinkA": 4, "shrinkB": 5},
        )

    # 原地停留和再次经过同一格均计入格内的访问次数。
    for (row, col), visits in Counter(path).items():
        if visits > 1:
            ax.text(
                col + 0.30, row + 0.30, f"{visits}×", ha="center", va="center", fontsize=8, color="#176c63",
                bbox={"facecolor": "white", "edgecolor": "none", "boxstyle": "round,pad=0.12", "alpha": 0.9},
            )
    fig.savefig(output_dir / "route.png", dpi=180)
    plt.close(fig)


def plot_results(rows, path, output_dir):
    plot_learning_curve(rows, output_dir)
    plot_greedy_path(path, output_dir)
