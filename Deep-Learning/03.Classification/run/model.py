import torch.nn as nn

class FashionNN(nn.Module):
    """Fashion-MNIST分类神经网络"""
    def __init__(self):
        super(FashionNN, self).__init__()

        # 将 1×28×28 图像展平为 784 维，经 128、64 维隐藏层输出 10 类 logits。
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """前向传播"""
        x = self.flatten(x)
        logits = self.layers(x)
        return logits
