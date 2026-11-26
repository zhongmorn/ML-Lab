"""修改顶部设置后运行 python main.py。"""

from pathlib import Path
import json

from run import PPO, PPOConfig
from tools.plot import plot_training
from tools.save import write_json


ROOT = Path(__file__).resolve().parent

# 选择 train、evaluate 或 plot。
MODE = "train"
ENV_ID = "CartPole-v1"
CONFIG = ROOT / "run/configs.json"
# 当前结果为 seed42_run1；下一次新实验写入 seed42_run2。
RUN_NUMBER = 2

# 评估和重新绘图使用已有实验目录。
RUN_DIR = ROOT / "output/CartPole-v1/seed42_run1"
CHECKPOINT = RUN_DIR / "model/final.pt"
DEVICE = "cuda"
EVAL_EPISODES = 100
EVAL_SEED = 30000
DETERMINISTIC = True
RENDER = False


def main():
    if MODE == "train":
        config = PPOConfig.from_json(CONFIG, env_id=ENV_ID)
        run_name = f"seed{config.seed}"
        if RUN_NUMBER is not None:
            if RUN_NUMBER < 1:
                raise ValueError("RUN_NUMBER 必须为正整数")
            run_name += f"_run{RUN_NUMBER}"
        output_dir = ROOT / "output" / config.env_id.replace("/", "_") / run_name
        summary = PPO(config).train(output_dir)
        plot_training(output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif MODE == "evaluate":
        agent = PPO.load(CHECKPOINT, device=DEVICE)
        result = agent.evaluate(
            episodes=EVAL_EPISODES, seed=EVAL_SEED,
            deterministic=DETERMINISTIC, render=RENDER,
        )
        # 补充评估按模型和种子单独保存。
        suffix = "" if DETERMINISTIC else "_sample"
        name = f"eval_{CHECKPOINT.stem}_{EVAL_SEED}{suffix}.json"
        output = CHECKPOINT.parent.parent / name
        if output.exists():
            raise FileExistsError(f"评估结果已存在: {output}")
        write_json(output, {**result, "checkpoint": CHECKPOINT.name})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Evaluation saved to {output}")
    elif MODE == "plot":
        plot_training(RUN_DIR)
    else:
        raise ValueError("MODE 必须是 train、evaluate 或 plot")


if __name__ == "__main__":
    main()
