import json
from pathlib import Path

import torch

from run.agent import GRPO
from tools.plot import plot_training


ROOT = Path(__file__).resolve().parent


def sample_task(count, generator):
    # 每个坐标独立均匀采样，两条对角线把平面划成四类。
    contexts = 2 * torch.rand(count, 2, generator=generator) - 1
    x, y = contexts.unbind(dim=1)
    correct_actions = (x + y >= 0).long() + 2 * (x - y >= 0).long()
    return contexts, correct_actions


@torch.no_grad()
def evaluate(policy, contexts, correct_actions):
    probabilities = policy(contexts).softmax(dim=-1)
    return {
        "accuracy": (probabilities.argmax(dim=-1) == correct_actions).float().mean().item(),
        "correct_action_probability": probabilities.gather(1, correct_actions[:, None]).mean().item(),
    }


def main():
    config = json.loads((ROOT / "run/configs.json").read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.manual_seed(config["train_seed"])

    agent = GRPO(config)
    train_generator = torch.Generator().manual_seed(config["train_seed"])
    eval_generator = torch.Generator().manual_seed(config["eval_seed"])
    eval_batch = sample_task(config["eval_contexts"], eval_generator)
    before = evaluate(agent.policy, *eval_batch)

    training = []
    for update in range(1, config["updates"] + 1):
        batch = sample_task(config["contexts_per_update"], train_generator)
        training.append({"update": update, **agent.update(*batch)})

    # 前后用同一批独立测试输入，便于比较策略本身的变化。
    after = evaluate(agent.policy, *eval_batch)
    results = {
        "config": config,
        "training": training,
        "evaluation": {"before": before, "after": after},
    }

    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    plot_training(training, output / "training.png")
    print(f"测试集贪心准确率 {before['accuracy']:.2%} → {after['accuracy']:.2%}")
    print(f"结果保存在 {output}")


if __name__ == "__main__":
    main()
