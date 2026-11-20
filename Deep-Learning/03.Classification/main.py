from pathlib import Path

from run.train import run_training

# 训练轮数和随机种子。
EPOCHS = 100
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().with_name("output")


def main():
    for optimizer_name in ("SGD", "Adam"):
        run_training(OUTPUT_DIR, optimizer_name, epochs=EPOCHS, seed=SEED)


if __name__ == "__main__":
    main()
