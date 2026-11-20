import torch.nn as nn


class LinearRegressionModel(nn.Module):
    """线性回归模型"""
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        """前向传播"""
        return self.linear(x)


class DeepRegressionModel(nn.Module):
    """深度神经网络回归模型"""
    def __init__(self, input_size):
        super().__init__()
        # 两层 ReLU 隐藏层学习非线性关系，输出层直接给出连续预测值。
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """前向传播"""
        return self.layers(x)
