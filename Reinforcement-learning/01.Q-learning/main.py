from pathlib import Path
from run.configs import load_config
from run.experiment import run_experiment


# if __name__ == "__main__":
#     run_experiment(Path(__file__).resolve().parent / "output")



# 奖励消融只把普通步奖励改为 0，沿用默认的训练和评估参数。
if __name__ == "__main__":
    step_reward = load_config("reward_ablation.json")["step_reward"]
    output_dir = Path(__file__).resolve().parent / "output" / "reward_ablation"
    run_experiment(output_dir, step_reward=step_reward)