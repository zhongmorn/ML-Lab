from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

LABEL_NAMES = [
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]


def split_training_data(dataset, seed):
    """从官方训练集的每个类别中抽出 10% 用于验证。"""
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    # 每类独立打乱，按 9:1 分层划分，保持训练集与验证集的类别比例。
    for label in dataset.targets.unique():
        indices = (dataset.targets == label).nonzero(as_tuple=True)[0]
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        val_count = len(indices) // 10
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def to_tensor_dataset(dataset):
    """一次完成与 ToTensor、Normalize 相同的预处理，供多轮训练复用。"""

    # 将图像补为 N×1×28×28；像素先缩放到 [0, 1]，再按 (x-0.5)/0.5 归一化到 [-1, 1]。
    images = dataset.data.unsqueeze(1).float() / 255
    images = (images - 0.5) / 0.5
    return TensorDataset(images, dataset.targets)


def prepare_loaders(seed, batch_size=64):
    # 先分层划分索引，再复用整批归一化张量；官方测试集独立处理。
    official_train = load_dataset(train=True)
    train, val = split_training_data(official_train, seed)
    normalized_train = to_tensor_dataset(official_train)
    train = Subset(normalized_train, train.indices)
    val = Subset(normalized_train, val.indices)
    test = to_tensor_dataset(load_dataset(train=False))

    # 训练批次打乱，验证和测试保持顺序；三个加载器使用独立随机种子。
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, generator=torch.Generator().manual_seed(seed + 1)
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, generator=torch.Generator().manual_seed(seed + 2)
    )
    return train_loader, val_loader, test_loader


def load_dataset(train):
    """加载 Fashion-MNIST，首次运行时下载到实验目录。"""
    return datasets.FashionMNIST(
        root=str(DATA_ROOT),
        train=train,
        download=True,
    )
