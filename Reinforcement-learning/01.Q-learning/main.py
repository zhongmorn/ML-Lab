from pathlib import Path

from run.configs import CONFIG
from run.experiment import run_experiment


# 奖励消融沿用同一套训练和评估参数，只覆盖普通步奖励。
if __name__ == "__main__":
    step_reward = CONFIG["reward_ablation"]["step_reward"]
    output_dir = Path(__file__).resolve().parent / "output" / "reward_ablation"
    run_experiment(output_dir, step_reward=step_reward)
