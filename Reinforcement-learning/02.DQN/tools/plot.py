from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

def smooth_curve(points, factor=0.9):
    """使用指数移动平均平滑曲线"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_learning_curve(returns, env_name, algo_name, save_path):
    """绘制单条训练回报曲线"""
    plt.figure(figsize=(10, 5))
    plt.title(f"{algo_name} Training on {env_name}")
    plt.xlabel('Episodes')
    plt.ylabel('Returns')

    # 绘制原始数据
    episodes = range(1, len(returns) + 1)
    plt.plot(episodes, returns, alpha=0.3, color='blue', label='Raw')

    # 绘制平滑数据
    if len(returns) > 10:
        smoothed = smooth_curve(returns)
        plt.plot(episodes, smoothed, color='red', label='Smoothed')

    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()


def plot_evaluation_returns(returns, env_name, save_path):
    episodes = range(1, len(returns) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(
        episodes, returns, color="#255A94", linewidth=1.3,
        marker="o", markersize=3.5, label="Episode return",
    )
    if len(returns) > 0:
        ax.set_xlim(0.5, len(returns) + 0.5)
        average_return = mean(returns)
        ax.axhline(
            average_return, color="#AB481E", linestyle="--", linewidth=1.5,
            label=f"Mean: {average_return:.2f}",
        )

    ax.set_title(f"DQN evaluation on {env_name}", loc="left", pad=28)
    ax.set(xlabel="Evaluation episode", ylabel="Episode return")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=1))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDE3EA", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, borderaxespad=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_animation(frames, path, fps=30):
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=1000 // fps, loop=0,
    )
