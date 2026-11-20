from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .data import get_data_loader, regression_target, synthetic_data
from .model import DeepRegressionModel, LinearRegressionModel
from tools.plot import plot_loss, plot_scatter


def train_model(model, train_loader, epochs=100, lr=0.03):
    """训练模型"""

    # 使用均方误差与 SGD 进行小批量训练，记录逐轮损失。
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0

        for X, y in train_loader:
            y_pred = model(X)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 对各批次 MSE 等权平均，最后一个较小批次也占相同权重。
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}')

    return losses


def verify(model1, model2, true_w, true_b, num_verify=5, seed=42):
    """验证两个模型的预测值：对比理论值、带噪声标签及两个模型的预测结果"""
    # 固定验证样本，同时保留无噪声理论值和带噪声标签。
    torch.manual_seed(seed)

    input_size = len(true_w)
    sample_X = torch.normal(0, 1, (num_verify, input_size))

    sample_y_true = regression_target(sample_X, true_w, true_b)
    sample_y_noisy = sample_y_true + torch.normal(0, 0.01, sample_y_true.shape)

    # 在评估模式下比较两个模型对同一批样本的预测。
    model1.eval()
    model2.eval()
    with torch.no_grad():
        pred1 = model1(sample_X)
        pred2 = model2(sample_X)

    # 输出逐样本对照表，绝对误差以带噪声标签为基准。
    widths = [50, 15, 15, 15, 15, 15, 15]
    headers = ["测试值", "理论值(无噪声)", "带噪声标签", "Linear模型预测值", "Linear模型误差", "Deep模型预测值", "Deep模型误差"]

    print("\n===== 两个模型验证结果对比 =====")
    header_line = ""
    for i in range(len(headers)):
        header_line += headers[i].ljust(widths[i]) + "|"
    print(header_line)

    sep_line = ""
    for w in widths:
        sep_line += "-" * w + "-----"
    sep_line += "----" 
    print(sep_line)

    for i in range(num_verify):
        x_vals = [f"{x:.4f}" for x in sample_X[i].numpy()]
        test_val_str = "[" + ", ".join(x_vals) + "]"

        true_val = sample_y_true[i].item()
        noisy_val = sample_y_noisy[i].item()
        p1 = pred1[i].item()
        p2 = pred2[i].item()
        e1 = abs(p1 - noisy_val)
        e2 = abs(p2 - noisy_val)

        row = (
            f"{test_val_str:<{widths[0]+3}}|"
            f"{true_val:<{widths[1]+6}.4f}|"
            f"{noisy_val:<{widths[2]+5}.4f}|"
            f"{p1:<{widths[3]+5}.6f}|"
            f"{e1:<{widths[4]+4}.6f}|"
            f"{p2:<{widths[5]+5}.6f}|"
            f"{e2:<{widths[6]+4}.6f}|"
        )
        print(row)


def train(output_dir: Path):
    # 固定随机种子，并统一两种模型的训练参数。
    seed = 42
    torch.manual_seed(seed)

    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    num_samples = 1000
    epochs = 100
    lr = 0.01
    num_verify = 10

    # 准备实验输出目录。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成默认非线性数据，查看特征与标签的关系。
    features, labels = synthetic_data(true_w, true_b, num_samples)
    print(f"\n特征矩阵形状: {features.shape}, 标签向量形状: {labels.shape}, 真实权重: {true_w}, 真实偏差: {true_b}\n")
    plot_scatter(features, labels, save_path=output_dir / "data.png")

    train_loader = get_data_loader(features, labels, batch_size=32)
    input_size = len(true_w)

    # 用同一批训练数据分别训练线性基线和深度模型，并保存损失曲线。
    print("\n开始训练线性模型...")
    model_linear = LinearRegressionModel(input_size)
    losses_linear = train_model(model_linear, train_loader, epochs=epochs, lr=lr)
    plot_loss(losses_linear, save_path=output_dir / "loss_linear.png")

    print("\n开始训练深度模型...")
    model_deep = DeepRegressionModel(input_size)
    losses_deep = train_model(model_deep, train_loader, epochs=epochs, lr=lr)
    plot_loss(losses_deep, save_path=output_dir / "loss_deep.png")

    # 用独立随机种子生成验证样本，逐条比较预测及误差。
    verify(model_linear, model_deep, true_w, true_b, num_verify=num_verify, seed=seed + 1)
