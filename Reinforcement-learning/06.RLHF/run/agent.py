import math

import torch
from torch import nn
from torch.nn import functional as F


class RewardModel:
    def __init__(self, config):
        self.scorer = nn.Linear(2, 4)
        self.optimizer = torch.optim.Adam(self.scorer.parameters(), lr=config["reward_learning_rate"])
        self.epochs = config["reward_epochs"]
        self.batch_size = config["reward_batch_size"]

    def margin(self, contexts, chosen, rejected):
        scores = self.scorer(contexts)
        return scores.gather(1, chosen[:, None]) - scores.gather(1, rejected[:, None])

    def fit(self, contexts, chosen, rejected, generator):
        history = []
        for epoch in range(1, self.epochs + 1):
            order = torch.randperm(len(contexts), generator=generator)
            total_loss = 0.0
            for indices in order.split(self.batch_size):
                # Bradley–Terry 损失提高胜出动作相对于落选动作的得分。
                margin = self.margin(contexts[indices], chosen[indices], rejected[indices])
                loss = F.softplus(-margin).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(indices)
            history.append({"epoch": epoch, "loss": total_loss / len(contexts)})
        return history

    def score(self, contexts, actions):
        scores = self.scorer(contexts)
        # 去掉每个输入的共同偏移，保留所有动作得分差。
        centered_scores = scores - scores.mean(dim=-1, keepdim=True)
        return centered_scores.gather(1, actions)


class PolicyTrainer:
    def __init__(self, config):
        self.policy = nn.Linear(2, 4)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        self.reference_log_prob = -math.log(self.policy.out_features)
        self.value = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        parameters = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config["policy_learning_rate"])
        self.epochs = config["policy_epochs"]
        self.clip_epsilon = config["clip_epsilon"]
        self.kl_beta = config["kl_beta"]
        self.value_coef = config["value_coef"]

    def _reference_kl(self, log_probs):
        # 在四个动作上求和，得到当前策略到初始均匀策略的精确 KL。
        return (log_probs.exp() * (log_probs - self.reference_log_prob)).sum(dim=-1).mean()

    def update(self, contexts, reward_model, correct_actions):
        with torch.no_grad():
            old_log_probs = self.policy(contexts).log_softmax(dim=-1)
            actions = torch.multinomial(old_log_probs.exp(), 1)
            old_action_log_probs = old_log_probs.gather(1, actions)
            rewards = reward_model.score(contexts, actions)
            # 单步任务中，回报就是即时奖励。基线为采样时的价值估计。
            advantages = rewards - self.value(contexts)

        for _ in range(self.epochs):
            log_probs = self.policy(contexts).log_softmax(dim=-1)
            ratios = (log_probs.gather(1, actions) - old_action_log_probs).exp()
            clipped_ratios = ratios.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.minimum(ratios * advantages, clipped_ratios * advantages).mean()
            value_loss = F.mse_loss(self.value(contexts), rewards)
            loss = policy_loss + self.value_coef * value_loss + self.kl_beta * self._reference_kl(log_probs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            updated_log_probs = self.policy(contexts).log_softmax(dim=-1)
            return {
                "sampled_accuracy": (actions == correct_actions[:, None]).float().mean().item(),
                "mean_model_reward": rewards.mean().item(),
                "kl": self._reference_kl(updated_log_probs).item(),
            }
