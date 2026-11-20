import matplotlib.pyplot as plt

# 各图共用模型颜色与标记，方便对应训练和验证结果。
MODEL_STYLES = {
    "Linear": ("#2563a6", "o"),
    "Deep": ("#b45309", "s"),
}


def plot_scatter(features, labels, save_path):
    """
    绘制特征与标签的散点图并保存图片
    
    参数:
        features: 特征矩阵
        labels: 标签数据
        save_path: 图片保存路径（如'./scatter_plot.png'）
    """
    # 并排查看两个输入特征与同一标签的关系。
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.scatter(features[:, 0].numpy(), labels.numpy(), s=8, alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Label')

    plt.subplot(122)
    plt.scatter(features[:, 1].numpy(), labels.numpy(), s=8, alpha=0.6)
    plt.xlabel('Feature 2')
    plt.ylabel('Label')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"散点图已保存至: {save_path}")
    plt.close()


def plot_loss(losses, save_path):
    """绘制损失随训练轮次变化的折线图"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, len(losses))
    plt.ylim(0, max(losses) * 1.1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"loss已保存至: {save_path}")
    plt.close()


def plot_comparison(results, output_dir):
    # 按数据条件展示训练曲线，共享坐标并用对数纵轴比较不同数量级。
    fig, axes = plt.subplots(
        len(results), 1, figsize=(6.2, 6.5), sharex=True, sharey=True, constrained_layout=True,
    )
    for ax, (dataset, models) in zip(axes, results.items()):
        for name, (color, _) in MODEL_STYLES.items():
            losses = models[name]["losses"]
            ax.plot(range(1, len(losses) + 1), losses, label=name, color=color, linewidth=1.6)
        ax.set_title(dataset, loc="left")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].set_ylabel("Training MSE (log scale)")
    axes[-1].set_xlabel("Epoch")
    fig.savefig(output_dir / "training.png", dpi=180)
    plt.close(fig)

    # 分别比较验证 MAE 和 RMSE，用纵向错位避免两个模型的点重叠。
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.5), sharex=True, sharey=True)
    rows = list(range(len(results)))
    for ax, metric in zip(axes, ("mae", "rmse")):
        for model_index, (name, (color, marker)) in enumerate(MODEL_STYLES.items()):
            offsets = [row + (model_index - 0.5) * 0.26 for row in rows]
            values = [models[name][metric] for models in results.values()]
            ax.scatter(values, offsets, label=name, color=color, marker=marker, s=45, zorder=3)
        ax.set_title(f"Validation {metric.upper()}", loc="left")
        ax.set_yticks(rows)
        ax.set_yticklabels(results)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_xscale("log")
    axes[0].invert_yaxis()
    axes[-1].set_xlabel("Error (log scale)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.89, bottom=0.10, hspace=0.28)
    fig.savefig(output_dir / "validation.png", dpi=180)
    plt.close(fig)
