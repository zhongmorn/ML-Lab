import matplotlib.pyplot as plt
import numpy as np


def plot_training(records, path):
    # 同时展示每轮的损失和准确率，横轴是已加入的树的数量。
    rounds = [row["round"] for row in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, label, color in (("train", "Training", "#255A94"), ("validation", "Validation", "#AB481E")):
        axes[0].plot(rounds, [row[f"{split}_logloss"] for row in records], label=label, color=color)
        axes[1].plot(rounds, [row[f"{split}_accuracy"] for row in records], label=label, color=color)
    axes[0].set(title="Log loss", ylabel="Binary cross-entropy")
    axes[1].set(title="Classification accuracy", ylabel="Accuracy", ylim=(0, 1.01))
    for ax in axes:
        ax.set_xlabel("Boosting round")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("XGBoost on Breast Cancer Wisconsin")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(matrix, path):
    # 行为真实类别，列为预测类别，恶性对应正类 1。
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ax.imshow(matrix, cmap="Blues", vmin=0)
    classes = ["Benign (0)", "Malignant (1)"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
           xlabel="Predicted label", ylabel="True label", title="Test confusion matrix")
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "#18344B"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, fontsize=15)
    fig.colorbar(display, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(records, path):
    # 展示排名前十的平均分裂增益重要性。
    top = list(reversed(records[:10]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.arange(len(top)), [row["importance"] for row in top], color="#255A94")
    ax.set(yticks=np.arange(len(top)), yticklabels=[row["feature"] for row in top],
           xlabel="Normalized average gain", title="Top 10 feature importances")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
