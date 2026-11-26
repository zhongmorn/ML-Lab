import json
from pathlib import Path

import torch

from run.agent import DPO
from tools.plot import plot_training


ROOT = Path(__file__).resolve().parent


def sample_preferences(count, generator):
    # 两条对角线把坐标划成四类，每类对应一个优选动作。
    contexts = 2 * torch.rand(count, 2, generator=generator) - 1
    x, y = contexts.unbind(dim=1)
    chosen = (x + y >= 0).long() + 2 * (x - y >= 0).long()

    # 从剩下三个动作中等概率选出一个劣选动作。
    offsets = torch.randint(1, 4, (count,), generator=generator)
    rejected = (chosen + offsets) % 4
    return contexts, chosen, rejected


@torch.no_grad()
def evaluate(policy, contexts, chosen, rejected):
    probabilities = policy(contexts).softmax(dim=-1)
    chosen_prob = probabilities.gather(1, chosen[:, None]).squeeze(1)
    rejected_prob = probabilities.gather(1, rejected[:, None]).squeeze(1)
    return {
        "accuracy": (probabilities.argmax(dim=-1) == chosen).float().mean().item(),
        "chosen_probability": chosen_prob.mean().item(),
        "preference_accuracy": (chosen_prob > rejected_prob).float().mean().item(),
    }


def main():
    config = json.loads((ROOT / "run/configs.json").read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.manual_seed(config["train_seed"])

    agent = DPO(config)
    train_generator = torch.Generator().manual_seed(config["train_seed"])
    eval_generator = torch.Generator().manual_seed(config["eval_seed"])
    train_data = sample_preferences(config["train_pairs"], train_generator)
    eval_data = sample_preferences(config["eval_pairs"], eval_generator)
    before = evaluate(agent.policy, *eval_data)

    training = []
    for epoch in range(1, config["epochs"] + 1):
        # 训练期间重排同一份离线偏好数据。
        indices = torch.randperm(config["train_pairs"], generator=train_generator)
        for batch in indices.split(config["batch_size"]):
            agent.update(*(column[batch] for column in train_data))

        with torch.no_grad():
            loss = agent.preference_loss(*train_data).item()
        metrics = evaluate(agent.policy, *train_data)
        training.append({
            "epoch": epoch, "loss": loss,
            "preference_accuracy": metrics["preference_accuracy"],
        })

    after = evaluate(agent.policy, *eval_data)
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
    print(f"测试集偏好准确率 {before['preference_accuracy']:.2%} → {after['preference_accuracy']:.2%}")
    print(f"结果保存在 {output}")


if __name__ == "__main__":
    main()
