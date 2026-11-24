import math

import torch
from torch import nn


class GRPO:
    def __init__(self, config):
        self.group_size = config["group_size"]
        self.clip_epsilon = config["clip_epsilon"]
        self.kl_beta = config["kl_beta"]
        self.update_epochs = config["update_epochs"]

        # 零初始化使四个动作的概率相同，初始均匀分布作为固定参考策略。
        self.policy = nn.Linear(2, 4)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        self.reference_log_prob = -math.log(self.policy.out_features)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["learning_rate"])

    def _reference_kl(self, log_probs):
        # 动作空间有限，可以枚举四个动作精确计算 KL。
        return (log_probs.exp() * (log_probs - self.reference_log_prob)).sum(dim=-1).mean()

    def update(self, contexts, correct_actions):
        with torch.no_grad():
            # 每个输入独立采样一组动作，记录采样时的对数概率。
            old_log_probs = self.policy(contexts).log_softmax(dim=-1)
            actions = torch.multinomial(old_log_probs.exp(), self.group_size, replacement=True)
            old_action_log_probs = old_log_probs.gather(dim=1, index=actions)

            # 每组单独归一化奖励，同分组的优势为零。
            rewards = (actions == correct_actions[:, None]).float()
            group_means = rewards.mean(dim=1, keepdim=True)
            group_stds = rewards.std(dim=1, keepdim=True, unbiased=False)
            advantages = (rewards - group_means) / group_stds.clamp_min(1e-8)

        # 多轮更新复用同一批动作、旧概率和优势。
        for _ in range(self.update_epochs):
            log_probs = self.policy(contexts).log_softmax(dim=-1)
            action_log_probs = log_probs.gather(dim=1, index=actions)
            ratios = (action_log_probs - old_action_log_probs).exp()
            clipped_ratios = ratios.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
            objective = torch.minimum(ratios * advantages, clipped_ratios * advantages)
            loss = -objective.mean() + self.kl_beta * self._reference_kl(log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            final_log_probs = self.policy(contexts).log_softmax(dim=-1)
            return {
                "mean_reward": rewards.mean().item(),
                "mixed_group_rate": (group_stds > 0).float().mean().item(),
                "kl": self._reference_kl(final_log_probs).item(),
            }
