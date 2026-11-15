"""用固定数值演示 3→3→3 神经网络的前向传播、反向传播和一次参数更新。"""

import numpy as np


def sigmoid(z):
    """计算 sigmoid，避免直接对很大的正数取指数。"""
    return np.exp(-np.logaddexp(0.0, -z))


def forward(x, w1, b1, w2, b2):
    """每行是一条样本，返回两层的加权和及其激活值。"""
    hidden_input = x @ w1 + b1
    hidden = sigmoid(hidden_input)
    output_input = hidden @ w2 + b2
    predictions = sigmoid(output_input)
    return hidden_input, hidden, output_input, predictions


def loss(predictions, y):
    """每条样本取半平方误差之和，再对样本取平均。"""
    return 0.5 * np.sum((predictions - y) ** 2) / len(y)


def backward(x, y, hidden, predictions, w2):
    """用链式法则计算两层的权重与偏置梯度。"""
    output_delta = (predictions - y) * predictions * (1 - predictions) / len(x)
    hidden_delta = (output_delta @ w2.T) * hidden * (1 - hidden)

    grad_w1 = x.T @ hidden_delta
    grad_b1 = hidden_delta.sum(axis=0)
    grad_w2 = hidden.T @ output_delta
    grad_b2 = output_delta.sum(axis=0)
    return grad_w1, grad_b1, grad_w2, grad_b2


def main():
    x = np.array([[1.0, 2.0, -1.0]])
    y = np.array([[0.2, 0.8, 0.5]])
    w1 = np.array([[0.2, -0.1, 0.4], [0.3, 0.0, -0.2], [-0.1, 0.2, 0.2]])
    b1 = np.zeros(3)
    w2 = np.array([[0.1, -0.2, 0.3], [0.4, 0.1, -0.3], [-0.2, 0.3, 0.2]])
    b2 = np.zeros(3)
    parameters = (w1, b1, w2, b2)
    learning_rate = 0.5

    hidden_input, hidden, output_input, predictions = forward(x, *parameters)
    print(f"输入：{x[0]}")
    print(f"隐藏层加权和：{hidden_input[0]}")
    print(f"隐藏层激活：{hidden[0]}")
    print(f"输出层加权和：{output_input[0]}")
    print(f"输出值：{predictions[0]}")
    print(f"目标值：{y[0]}")
    print(f"初始损失：{loss(predictions, y):.6f}")

    gradients = backward(x, y, hidden, predictions, w2)
    for name, parameter, gradient in zip(("W1", "b1", "W2", "b2"), parameters, gradients):
        print(f"{name} 梯度：\n{gradient}")
        parameter -= learning_rate * gradient

    _, _, _, updated_predictions = forward(x, *parameters)
    print(f"更新后输出值：{updated_predictions[0]}")
    print(f"更新后损失：{loss(updated_predictions, y):.6f}")


if __name__ == "__main__":
    main()
