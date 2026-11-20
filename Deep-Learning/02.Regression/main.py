from pathlib import Path

# train 运行默认实验；compare 对照三种数据条件。
MODE = "compare"


def main():
    # 根据源码位置保存结果，避免受启动目录影响。
    output_dir = Path(__file__).resolve().parent / "output"
    if MODE == "train":
        from run.train import train

        train(output_dir)
    elif MODE == "compare":
        from run.compare import compare

        compare(output_dir / "comparison")

if __name__ == "__main__":
    main()
