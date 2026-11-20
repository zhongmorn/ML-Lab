import torch


def regression_target(features, w, b, *, nonlinear=True):
    """给定特征，计算无噪声标签。"""
    inputs = features
    # 非线性目标先对特征作 x·sin(x) 变换，再计算仿射组合。
    if nonlinear:
        inputs = features.sin() * features
    return inputs @ w + b


def synthetic_data(w, b, num_examples, *, nonlinear=True, noise_std=0.01, generator=None):
    """生成标准正态特征及带零均值高斯噪声的列向量标签。"""
    # 标准正态特征叠加零均值高斯标签噪声；标签列形状与模型输出一致。
    features = torch.normal(0, 1, (num_examples, len(w)), generator=generator)
    labels = regression_target(features, w, b, nonlinear=nonlinear)
    noise = torch.normal(0, noise_std, labels.shape, generator=generator)
    return features, (labels + noise).reshape(-1, 1)


def get_data_loader(features, labels, batch_size=32, *, generator=None):
    """创建数据加载器"""
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
