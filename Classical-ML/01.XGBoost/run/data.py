import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data(seed=42):
    # 使用库内置的数据；将恶性设为正类 1，良性设为负类 0。
    dataset = load_breast_cancer()
    labels = (dataset.target == 0).astype(np.int64)
    indices = np.arange(len(labels))

    # 先留出 20% 测试集，再从剩余样本中留出 25% 作为验证集。
    train_validation, test = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels,
    )
    train, validation = train_test_split(
        train_validation, test_size=0.25, random_state=seed,
        stratify=labels[train_validation],
    )
    splits = {"train": train, "validation": validation, "test": test}
    return dataset.data, labels, dataset.feature_names, splits
