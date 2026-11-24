import json
from pathlib import Path

import torch

from run.agent import PolicyTrainer, RewardModel
from tools.plot import plot_training


ROOT = Path(__file__).resolve().parent


def sample_task(count, generator):
    # 两条对角线将均匀采样的坐标划分成四类。
    contexts = 2 * torch.rand(count, 2, generator=generator) - 1
    x, y = contexts.unbind(dim=1)
    correct = (x + y >= 0).long() + 2 * (x - y >= 0).long()
    return contexts, correct


def sample_preferences(count, generator):
    # 正确动作总是胜出，落选动作从另外三个动作中均匀抽取。
    contexts, chosen = sample_task(count, generator)
    offset = torch.randint(1, 4, (count,), generator=generator)
    return contexts, chosen, (chosen + offset) % 4


@torch.no_grad()
def evaluate(policy, contexts, correct):
    probabilities = policy(contexts).softmax(dim=-1)
    return {
        "accuracy": (probabilities.argmax(dim=-1) == correct).float().mean().item(),
        "correct_action_probability": probabilities.gather(1, correct[:, None]).mean().item(),
    }


def main():
    config = json.loads((ROOT / "run/configs/default.json").read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.manual_seed(config["train_seed"])
    train_generator = torch.Generator().manual_seed(config["train_seed"])
    eval_generator = torch.Generator().manual_seed(config["eval_seed"])
    eval_contexts, eval_chosen, eval_rejected = sample_preferences(config["eval_contexts"], eval_generator)

    reward_model = RewardModel(config)
    preferences = sample_preferences(config["preference_count"], train_generator)
    reward_training = reward_model.fit(*preferences, train_generator)
    with torch.no_grad():
        test_margin = reward_model.margin(eval_contexts, eval_chosen, eval_rejected)
        preference_accuracy = (test_margin > 0).float().mean().item()

    # PPO 阶段只用冻结的奖励模型打分，真实动作标签仅用于记录正确率。
    reward_model.scorer.requires_grad_(False)
    trainer = PolicyTrainer(config)
    before = evaluate(trainer.policy, eval_contexts, eval_chosen)
    policy_training = []
    for update in range(1, config["policy_updates"] + 1):
        contexts, correct = sample_task(config["contexts_per_update"], train_generator)
        policy_training.append({"update": update, **trainer.update(contexts, reward_model, correct)})

    after = evaluate(trainer.policy, eval_contexts, eval_chosen)
    results = {
        "config": config,
        "training": {"reward_model": reward_training, "ppo": policy_training},
        "evaluation": {"reward_model_preference_accuracy": preference_accuracy, "before": before, "after": after},
    }
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    plot_training(reward_training, policy_training, output / "training.png")
    print(f"奖励模型测试偏好准确率 {preference_accuracy:.2%}")
    print(f"策略测试贪心准确率 {before['accuracy']:.2%} → {after['accuracy']:.2%}")
    print(f"结果保存在 {output}")


if __name__ == "__main__":
    main()
