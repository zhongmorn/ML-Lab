from pathlib import Path

import matplotlib

# 使用非交互后端，训练结束后直接保存图像。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_training(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 本次绘图统一设置中文字体和坐标轴样式。
    style = {"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(style):
        # 绘制真实批次损失，并标注同一参考批的训练前后损失和实验信息。
        history = report["loss_history"]
        steps = [point["step"] for point in history]
        losses = [point["loss"] for point in history]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.plot(steps, losses, color="#2362A6", linewidth=2)
        ax.set(xlabel="已完成的参数更新次数", ylabel="训练交叉熵", title="下一词预测的训练损失")
        ax.set_xlim(0, steps[-1])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.text(0.97, 0.92, f"固定批初始损失：{report['initial_loss']:.6f}\n固定批最终损失：{report['final_loss']:.6f}\n随机种子：{report['seed']} · {report['documents']} 段文本", transform=ax.transAxes, ha="right", va="top")
        fig.savefig(output_dir / "training.png", dpi=180)
        plt.close(fig)
        # 展示前八步的前两名候选；概率沿用完整词表 softmax，不重新归一化。
        trace = report["trace"][:8]
        positions = list(range(len(trace)))
        first = [item["probability"] for item in trace]
        second = [item["runner_up"]["probability"] for item in trace]
        fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
        first_bars = ax.bar([i - 0.19 for i in positions], first, width=0.36, color="#237C78", label="第一名（实际生成）")
        second_bars = ax.bar([i + 0.19 for i in positions], second, width=0.36, color="#D88838", label="第二名（候选）")
        ax.bar_label(first_bars, labels=[f"{item['token']}\n{item['probability']:.2%}" for item in trace], padding=5)
        ax.bar_label(second_bars, labels=[f"{item['runner_up']['token']}\n{item['runner_up']['probability']:.3%}" for item in trace], padding=6, color="#9A5316")
        ax.set_xticks(positions, [f"第 {i + 1} 步" for i in positions])
        ax.set(xlabel="自回归生成顺序（每一步沿第一名继续生成）", ylabel="词元的 softmax 概率", title=f"留出问题：{report['question']}（前 {len(trace)} 步）")
        ax.set_ylim(0, 1.24)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="upper right", ncol=2, frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.savefig(output_dir / "prediction.png", dpi=180)
        plt.close(fig)
