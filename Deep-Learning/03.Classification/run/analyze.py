from collections import Counter

from tools.plot import plot_dataset

from .data import load_dataset

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def run_analysis():
    # 读取官方训练、测试集，汇总样本规模与类别分布。
    print("正在加载Fashion-MNIST数据集...")
    train = load_dataset(train=True)
    test = load_dataset(train=False)

    print("数据集加载成功！开始分析...\n")
    print(f"训练集样本数：{len(train)}")
    print(f"测试集样本数：{len(test)}")
    print(f"图像尺寸：{tuple(train.data.shape[1:])} 像素（灰度图）")

    for name, split in (("训练集", train), ("测试集", test)):
        counts = Counter(CLASS_NAMES[label] for label in split.targets.tolist())
        print(f"\n{name}类别分布：")
        for label, count in counts.items():
            print(f"  {label}: {count}")

    # 展示各类样例与原始像素分布。
    plot_dataset(train.data.numpy(), train.targets.numpy(), CLASS_NAMES)


if __name__ == "__main__":
    run_analysis()
