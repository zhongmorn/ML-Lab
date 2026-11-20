import json

import torch
from torch import nn, optim

from tools.plot import plot_predictions, plot_training_metrics

from .data import LABEL_NAMES, prepare_loaders
from .model import FashionNN

def train_epoch(model, loader, criterion, optimizer):
    """累计本轮训练批次的交叉熵与正确预测数。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 每批完成前向计算、反向传播与参数更新。
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 按样本数加权累计批次平均损失，避免末批样本较少时权重失真。
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total


def evaluate_model(model, loader):
    """用当前模型评估整份数据，返回逐样本平均交叉熵和准确率。"""

    # 评估时关闭梯度记录；交叉熵先求和，再按总样本数平均。
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


def run_training(output_dir, optimizer_name, seed=42, epochs=100):
    # 使用固定种子准备训练、验证和官方测试数据。
    torch.set_num_threads(2)
    train_loader, val_loader, test_loader = prepare_loaders(seed)
    print(
        f"训练 {len(train_loader.dataset)} 张，验证 {len(val_loader.dataset)} 张，"
        f"测试 {len(test_loader.dataset)} 张", flush=True,
    )

    # 使用相同的模型初始化与学习率，分别训练 SGD 和 Adam。
    torch.manual_seed(seed)
    model = FashionNN()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 两种优化器的指标、曲线和预测图分别保存，避免互相覆盖。
    experiment_dir = output_dir / optimizer_name.lower()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 固定十张测试样例，供不同轮次的预测图直接比较。
    samples = [test_loader.dataset[index] for index in range(10)]
    images = torch.stack([image for image, _ in samples])
    labels = torch.tensor([label for _, label in samples])

    # 逐轮记录训练与验证的损失和准确率。
    rows = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = evaluate_model(model, val_loader)
        row = {
            "optimizer": optimizer_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy_pct": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy_pct": val_accuracy,
        }
        print(
            f"{optimizer_name} 第 {epoch:3d}/{epochs} 轮  训练损失 {train_loss:.4f}  "
            f"训练准确率 {train_accuracy:.2f}%  验证损失 {val_loss:.4f}  "
            f"验证准确率 {val_accuracy:.2f}%", flush=True,
        )

        # 仅在第 10 轮与最终轮评估测试集，并绘制固定样例预测。
        if epoch in (10, epochs):
            _, test_accuracy = evaluate_model(model, test_loader)
            row["test_accuracy_pct"] = test_accuracy
            with torch.no_grad():
                predictions = model(images).argmax(dim=1)
            plot_predictions(
                images.numpy(), labels, predictions, LABEL_NAMES, experiment_dir,
                filename=f"predictions_{epoch}.png",
                title=f"{optimizer_name} | Epoch {epoch} | Test accuracy {test_accuracy:.2f}%",
            )
            print(
                f"{optimizer_name} 第 {epoch} 轮测试准确率 {test_accuracy:.2f}%", flush=True,
            )

        rows.append(row)

    results = {
        "config": {
            "optimizer": optimizer_name,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
        },
        "training": rows,
    }
    (experiment_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    # 根据完整历史输出训练与验证曲线。
    plot_training_metrics(rows, experiment_dir)
    return rows
