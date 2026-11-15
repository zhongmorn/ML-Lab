"""Actor/Critic 网络与动作分布。训练循环在 model.py。"""

import math

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F


def make_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int,
             output_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layer = nn.Linear(previous, size)
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers.extend([layer, nn.Tanh()])
        previous = size
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, output_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class DiscreteActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.net = make_mlp(observation_dim, hidden_sizes, action_dim, output_gain=0.01)

    def forward(self, observation: Tensor) -> Categorical:
        # 输入与更新处已有有限值检查，避免分布校验在每次 CUDA 推理时同步。
        return Categorical(logits=self.net(observation), validate_args=False)


class ContinuousActor(nn.Module):
    def __init__(self, observation_dim: int, low: np.ndarray, high: np.ndarray,
                 hidden_sizes: tuple[int, ...]):
        super().__init__()
        low = torch.as_tensor(np.asarray(low).reshape(-1), dtype=torch.float32)
        high = torch.as_tensor(np.asarray(high).reshape(-1), dtype=torch.float32)
        if not torch.isfinite(low).all() or not torch.isfinite(high).all() or not torch.all(high > low):
            raise ValueError("Continuous actions require finite, strictly increasing bounds")
        self.mean_net = make_mlp(observation_dim, hidden_sizes, low.numel(), output_gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(low.numel()))
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)

    def forward(self, observation: Tensor) -> Normal:
        mean = self.mean_net(observation)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return Normal(mean, std, validate_args=False)

    def transform(self, raw_action: Tensor) -> Tensor:
        return raw_action.tanh() * self.action_scale + self.action_bias

    def log_prob(self, distribution: Normal, raw_action: Tensor) -> Tensor:
        # log(1-tanh(z)^2) 的稳定形式，z 很大时也不会 log(0)。
        log_tanh_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        correction = self.action_scale.log() + log_tanh_jacobian
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, observation_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.net = make_mlp(observation_dim, hidden_sizes, 1, output_gain=1.0)

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation).squeeze(-1)


class ActorCritic(nn.Module):
    """输入 shape [batch, obs_dim]；连续动作始终在网络内部展平。"""

    def __init__(self, observation_dim: int, action_space: gym.Space,
                 hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.actor = DiscreteActor(observation_dim, action_space.n, hidden_sizes)
            self.action_dim = 1
        elif isinstance(action_space, gym.spaces.Box):
            self.actor = ContinuousActor(observation_dim, action_space.low, action_space.high, hidden_sizes)
            self.action_dim = int(np.prod(action_space.shape))
        else:
            raise TypeError("Only Discrete and finite-bounded Box actions are supported")
        self.critic = Critic(observation_dim, hidden_sizes)

    def value(self, observation: Tensor) -> Tensor:
        return self.critic(observation)

    def predict_actions(self, observation: Tensor, deterministic: bool = True) -> Tensor:
        """纯 Actor 推理：评估不计算 value、log_prob，也不构造确定性分布。"""
        if deterministic:
            if self.is_discrete:
                return self.actor.net(observation).argmax(dim=-1)
            return self.actor.transform(self.actor.mean_net(observation))
        distribution = self.actor(observation)
        raw_action = distribution.sample()
        return raw_action if self.is_discrete else self.actor.transform(raw_action)

    def sample_actions(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """轨迹采样的动作与旧概率；Critic 可在回合结束后批量计算。"""
        distribution = self.actor(observation)
        if self.is_discrete:
            raw_action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
            action = raw_action
            log_prob = distribution.log_prob(raw_action)
        else:
            raw_action = distribution.mean if deterministic else distribution.sample()
            action = self.actor.transform(raw_action)
            log_prob = self.actor.log_prob(distribution, raw_action)
        return action, raw_action, log_prob

    def act(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action, raw_action, log_prob = self.sample_actions(observation, deterministic)
        return action, raw_action, log_prob, self.critic(observation)

    def evaluate_actions(self, observation: Tensor, raw_action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.actor(observation)
        if self.is_discrete:
            log_prob = distribution.log_prob(raw_action.long())
            entropy = distribution.entropy()
        else:
            log_prob = self.actor.log_prob(distribution, raw_action)
            # 当前 squashed Gaussian 的 entropy；不是未变换高斯的解析 entropy。
            fresh_action = distribution.rsample()
            entropy = -self.actor.log_prob(distribution, fresh_action)
        return log_prob, entropy, self.critic(observation)

