import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(x, w1, b1, w2, b2):
    # 两层都先做线性变换，再用 sigmoid 激活；每行是一条样本。
    hidden_input = x @ w1 + b1
    hidden = sigmoid(hidden_input)
    output_input = hidden @ w2 + b2
    predictions = sigmoid(output_input)
    return hidden_input, hidden, output_input, predictions


def loss(predictions, y):
    # 对输出维度求和、对样本取平均，与下面的反向传播保持一致。
    return 0.5 * np.sum((predictions - y) ** 2) / len(y)


def backward(x, y, hidden, predictions, w2):
    # 链式法则：输出误差乘 sigmoid 导数，再通过 W2 转置传回隐藏层。
    output_delta = (predictions - y) * predictions * (1 - predictions) / len(x)
    hidden_delta = (output_delta @ w2.T) * hidden * (1 - hidden)

    # 权重梯度用“输入转置 × 误差”计算，偏置梯度沿样本维累加。
    grad_w1 = x.T @ hidden_delta
    grad_b1 = hidden_delta.sum(axis=0)
    grad_w2 = hidden.T @ output_delta
    grad_b2 = output_delta.sum(axis=0)
    return grad_w1, grad_b1, grad_w2, grad_b2


def main():
    # 固定输入、目标和两层参数；权重矩阵的行对应上一层，列对应下一层。
    x = np.array([[1.0, 2.0, -1.0]])
    y = np.array([[0.2, 0.8, 0.5]])
    w1 = np.array([[0.2, -0.1, 0.4], [0.3, 0.0, -0.2], [-0.1, 0.2, 0.2]])
    b1 = np.zeros(3)
    w2 = np.array([[0.1, -0.2, 0.3], [0.4, 0.1, -0.3], [-0.2, 0.3, 0.2]])
    b2 = np.zeros(3)
    parameters = (w1, b1, w2, b2)
    learning_rate = 0.5

    # 正向传播并打印中间结果，便于和 README 的手算过程逐项核对。
    hidden_input, hidden, output_input, predictions = forward(x, *parameters)
    print(f"输入：{x[0]}")
    print(f"隐藏层加权和：{hidden_input[0]}")
    print(f"隐藏层激活：{hidden[0]}")
    print(f"输出层加权和：{output_input[0]}")
    print(f"输出值：{predictions[0]}")
    print(f"目标值：{y[0]}")
    print(f"初始损失：{loss(predictions, y):.6f}")

    # 反向传播得到四组梯度，按相同参数顺序执行一次梯度下降。
    gradients = backward(x, y, hidden, predictions, w2)
    for name, parameter, gradient in zip(("W1", "b1", "W2", "b2"), parameters, gradients):
        print(f"{name} 梯度：\n{gradient}")
        parameter -= learning_rate * gradient

    # 用更新后的参数复算，比较输出和损失的变化。
    _, _, _, updated_predictions = forward(x, *parameters)
    print(f"更新后输出值：{updated_predictions[0]}")
    print(f"更新后损失：{loss(updated_predictions, y):.6f}")


if __name__ == "__main__":
    main()
