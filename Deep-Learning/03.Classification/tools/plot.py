import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_predictions(
    images, labels, predictions, label_names, output_dir,
    filename, title,
):
    # 用两行五列展示固定测试样例，绿色和红色分别标记正确与错误预测。
    figure = plt.figure(figsize=(12, 7.5))
    for index, (image, label, prediction) in enumerate(zip(images, labels, predictions), 1):
        plt.subplot(2, 5, index)

        # 去掉通道维度，并将归一化像素从 [-1, 1] 还原到 [0, 1]。
        plt.imshow(image.squeeze() * 0.5 + 0.5, cmap="gray")
        true_label = label_names[label]
        pred_label = label_names[prediction]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, 0.94], h_pad=3)
    figure.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(figure)


def plot_training_metrics(rows, output_dir):
    # 按轮次绘制训练、验证的损失与准确率。
    optimizer_name = rows[0]["optimizer"]
    epochs = [row["epoch"] for row in rows]
    for train_metric, val_metric, axis_label, filename in (
        ("train_loss", "val_loss", "Cross-entropy", "loss.png"),
        ("train_accuracy_pct", "val_accuracy_pct", "Accuracy (%)", "accuracy.png"),
    ):
        figure, axes = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
        axes.plot(
            epochs, [row[train_metric] for row in rows],
            label="Train", linewidth=2,
        )
        axes.plot(
            epochs, [row[val_metric] for row in rows],
            label="Validation", linewidth=2,
        )
        axes.set(
            xlabel="Epoch", ylabel=axis_label,
            title=f"{optimizer_name} ({epochs[-1]} epochs)",
        )
        axes.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        axes.grid(alpha=0.25)
        axes.legend()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def plot_dataset(images, labels, class_names):
    # 按类别展示代表样本，再统计全部训练像素的灰度分布。
    with plt.rc_context({"font.family": ["Heiti TC"], "axes.unicode_minus": False}):
        figure = plt.figure(figsize=(12, 6))
        for label, name in enumerate(class_names):
            index = np.flatnonzero(labels == label)[0]
            plt.subplot(2, 5, label + 1)
            plt.imshow(images[index], cmap="gray")
            plt.title(name, fontsize=10)
            plt.axis("off")
        figure.suptitle("每个类别的样本示例", fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.close(figure)

        # 将训练图像的所有像素展平，观察 0–255 范围内的像素分布。
        figure = plt.figure(figsize=(10, 4))
        plt.hist(images.ravel(), bins=50, color="gray", alpha=0.7)
        plt.title("训练集像素值分布（0-255）")
        plt.xlabel("像素值")
        plt.ylabel("频数")
        plt.grid(alpha=0.3)
        plt.show()
        plt.close(figure)
