"""PPO 配置、轨迹、GAE、训练与评估；网络定义见 network.py。"""

from dataclasses import asdict, dataclass, fields
from collections.abc import Iterator
from contextlib import ExitStack, closing
from pathlib import Path
import json
import math
import time

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from run.network import ActorCritic
from tools.seed import seed_everything
from tools.save import CSVLogger, write_json, write_evaluation_episodes
from output.report import update_readme_results

# ==================== 参数与设备 ====================

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This example supports cpu, cuda, cuda:N or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or install CUDA-enabled PyTorch")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        device = torch.device("cuda", index)
    return device



@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    train_rounds: int = 500
    batch_size: int = 64
    update_epochs: int = 10
    learning_rate: float = 3e-4
    reward_scale: float = 1.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    anneal_lr: bool = True
    normalize_advantage: bool = True
    hidden_sizes: tuple[int, ...] = (64, 64)
    monitor_seeds: tuple[int, ...] = (10_000, 10_001, 10_002)
    final_eval_episodes: int = 100
    final_eval_seed: int = 30_000
    eval_batch_size: int = 32
    seed: int = 42
    device: str = "cuda"
    torch_threads: int = 1

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(self.hidden_sizes)
        self.monitor_seeds = tuple(self.monitor_seeds)
        for name in ("train_rounds", "batch_size", "update_epochs", "final_eval_episodes", "eval_batch_size", "torch_threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.hidden_sizes or any(not isinstance(n, int) or n <= 0 for n in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip_coef must be in (0, 1)")
        for name in ("learning_rate", "max_grad_norm", "reward_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("value_coef", "entropy_coef", "target_kl"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if len(self.monitor_seeds) != 3 or len(set(self.monitor_seeds)) != 3:
            raise ValueError("monitor_seeds must contain exactly three distinct seeds")
        if any(not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in self.monitor_seeds):
            raise ValueError("monitor_seeds must be integers in [0, 2**32)")
        if not isinstance(self.final_eval_seed, int) or not 0 <= self.final_eval_seed < 2**32 - self.final_eval_episodes:
            raise ValueError("final_eval_seed plus final_eval_episodes must be below 2**32")
        if set(range(self.final_eval_seed, self.final_eval_seed + self.final_eval_episodes)) & set(self.monitor_seeds):
            raise ValueError("Final evaluation seeds must differ from the three monitoring seeds")
        if not self.env_id:
            raise ValueError("env_id cannot be empty")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        result["monitor_seeds"] = list(self.monitor_seeds)
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "PPOConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))



# ==================== Rollout 与 GAE ====================

def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                terminated: np.ndarray, truncated: np.ndarray,
                gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(a) for a in (rewards, values, next_values, terminated, truncated)]
    if not all(a.ndim == 1 and a.shape == arrays[0].shape for a in arrays):
        raise ValueError("GAE inputs must be one-dimensional arrays with identical shapes")
    rewards, values, next_values, terminated, truncated = arrays
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = 0.0
    for t in reversed(range(len(rewards))):
        bootstrap = 1.0 - float(terminated[t])
        continue_trace = 1.0 - float(bool(terminated[t]) or bool(truncated[t]))
        delta = float(rewards[t]) + gamma * bootstrap * float(next_values[t]) - float(values[t])
        next_advantage = delta + gamma * gae_lambda * continue_trace * next_advantage
        advantages[t] = next_advantage
    returns = advantages + values.astype(np.float32)
    return advantages, returns



@dataclass
class RolloutBatch:
    observations: Tensor
    raw_actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor

    def minibatches(self, batch_size: int) -> Iterator["RolloutBatch"]:
        indices = torch.randperm(len(self.observations), device=self.observations.device)
        for start in range(0, len(indices), batch_size):
            subset = indices[start:start + batch_size]
            yield RolloutBatch(**{name: value[subset] for name, value in vars(self).items()})



class RolloutBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, is_discrete: bool):
        self.capacity = capacity
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.raw_actions = np.zeros((capacity,) if is_discrete else (capacity, action_dim),
                                    dtype=np.int64 if is_discrete else np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.old_log_probs = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.reset()

    def reset(self) -> None:
        self.position = 0
        self.advantages = None
        self.returns = None

    def add(self, observation, raw_action, reward: float, value: float, next_value: float,
            old_log_prob: float, terminated: bool, truncated: bool) -> None:
        if self.position >= self.capacity:
            # 无固定时间上限的环境也能保存一个完整回合。
            for name in ("observations", "raw_actions", "rewards", "values", "next_values",
                         "old_log_probs", "terminated", "truncated"):
                array = getattr(self, name)
                expanded = np.zeros((self.capacity * 2, *array.shape[1:]), dtype=array.dtype)
                expanded[:self.capacity] = array
                setattr(self, name, expanded)
            self.capacity *= 2
        t = self.position
        self.observations[t] = observation
        self.raw_actions[t] = raw_action
        self.rewards[t] = reward
        self.values[t] = value
        self.next_values[t] = next_value
        self.old_log_probs[t] = old_log_prob
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.position += 1

    def finish(self, gamma: float, gae_lambda: float) -> None:
        if self.position == 0:
            raise RuntimeError("Cannot compute GAE for an empty rollout")
        sl = slice(0, self.position)
        self.advantages, self.returns = compute_gae(
            self.rewards[sl], self.values[sl], self.next_values[sl],
            self.terminated[sl], self.truncated[sl], gamma, gae_lambda)

    def as_batch(self, device: torch.device, normalize_advantage: bool = True) -> RolloutBatch:
        if self.advantages is None:
            raise RuntimeError("Call finish() before reading a rollout")
        sl = slice(0, self.position)
        advantages = torch.as_tensor(self.advantages.copy(), device=device)
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return RolloutBatch(
            observations=torch.as_tensor(self.observations[sl].copy(), device=device),
            raw_actions=torch.as_tensor(self.raw_actions[sl].copy(), device=device),
            old_log_probs=torch.as_tensor(self.old_log_probs[sl].copy(), device=device),
            old_values=torch.as_tensor(self.values[sl].copy(), device=device),
            advantages=advantages,
            returns=torch.as_tensor(self.returns.copy(), device=device),
        )



# ==================== PPO 训练与评估 ====================

def ppo_policy_loss(new_log_prob: Tensor, old_log_prob: Tensor,
                    advantages: Tensor, clip_coef: float) -> Tensor:
    ratio = (new_log_prob - old_log_prob).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantages
    return -torch.minimum(unclipped, clipped).mean()



def describe_spaces(env: gym.Env) -> dict:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("Observations must be a Box space; flatten/encode other spaces first")
    result = {"observation_shape": list(env.observation_space.shape)}
    if isinstance(env.action_space, gym.spaces.Discrete):
        result["action"] = {"type": "Discrete", "n": int(env.action_space.n),
                            "start": int(env.action_space.start)}
    elif isinstance(env.action_space, gym.spaces.Box):
        result["action"] = {"type": "Box", "shape": list(env.action_space.shape),
                            "low": env.action_space.low.reshape(-1).tolist(),
                            "high": env.action_space.high.reshape(-1).tolist()}
    else:
        raise TypeError("Actions must be Discrete or finite-bounded Box")
    return result



class PPO:
    def __init__(self, config: PPOConfig):
        self.config = config
        torch.set_num_threads(config.torch_threads)
        seed_everything(config.seed)
        self.device = resolve_device(config.device)
        with closing(gym.make(config.env_id)) as probe:
            self.space_info = describe_spaces(probe)
            self.observation_dim = int(np.prod(probe.observation_space.shape))
            self.action_space = probe.action_space
            self.network = ActorCritic(self.observation_dim, probe.action_space, config.hidden_sizes).to(self.device)
        # CUDA 使用融合 Adam，减少逐参数启动 CUDA kernel 的开销。
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate,
                                          eps=1e-5, fused=self.device.type == "cuda")
        self.global_step = 0
        self.update_count = 0
        self.best_mean_return = None

    def _observation_tensor(self, observation) -> Tensor:
        array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if array.shape[-1] != self.observation_dim or not np.isfinite(array).all():
            raise ValueError("Observation shape is incompatible or contains NaN/Inf")
        return torch.as_tensor(array, device=self.device)

    def _environment_action(self, action: Tensor):
        if self.network.is_discrete:
            return int(action.item()) + int(self.action_space.start)
        return action.cpu().numpy().reshape(self.action_space.shape).astype(self.action_space.dtype)

    @torch.no_grad()
    def predict(self, observation, deterministic: bool = True):
        action = self.network.predict_actions(self._observation_tensor(observation), deterministic)
        return self._environment_action(action[0])

    def update(self, batch: RolloutBatch) -> dict:
        """old_log_probs / advantages / returns 都是采样时固定的目标。"""
        self.network.train()
        metric_names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        weighted = torch.zeros(len(metric_names), device=self.device)
        total_samples = 0
        optimizer_steps = 0
        early_stopped = False
        max_kl = 0.0
        for _ in range(self.config.update_epochs):
            for minibatch in batch.minibatches(self.config.batch_size):
                new_log_prob, entropy, value = self.network.evaluate_actions(
                    minibatch.observations, minibatch.raw_actions)
                log_ratio = new_log_prob - minibatch.old_log_probs
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_coef).float().mean()
                # KL 提前停止需要一次同步；其余指标留在 GPU，更新结束再统一取回。
                kl_value = approx_kl.item()
                if not math.isfinite(kl_value):
                    raise FloatingPointError("Non-finite KL; inspect log probabilities and learning rate")
                max_kl = max(max_kl, kl_value)
                # 先检查再做这一个 minibatch 的更新；clip 不是 KL 硬约束。
                if self.config.target_kl > 0 and kl_value > 1.5 * self.config.target_kl:
                    early_stopped = True
                    break
                policy_loss = ppo_policy_loss(new_log_prob, minibatch.old_log_probs,
                                              minibatch.advantages, self.config.clip_coef)
                value_loss = 0.5 * (value - minibatch.returns).square().mean()
                mean_entropy = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * mean_entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm,
                                                error_if_nonfinite=True)
                self.optimizer.step()
                n = len(minibatch.observations)
                weighted += torch.stack((policy_loss.detach(), value_loss.detach(),
                                         mean_entropy.detach(), approx_kl, clip_fraction)) * n
                total_samples += n
                optimizer_steps += 1
            if early_stopped:
                break
        if total_samples == 0:
            raise RuntimeError("No optimizer step was performed; rollout policy and old probabilities disagree")
        metrics = dict(zip(metric_names, (weighted / total_samples).tolist()))
        with torch.no_grad():
            variance = batch.returns.var(unbiased=False)
            explained_variance = torch.where(variance > 1e-8,
                1 - (batch.returns - batch.old_values).var(unbiased=False) / variance.clamp_min(1e-8),
                torch.zeros_like(variance)).item()
        metrics.update(explained_variance=explained_variance, max_approx_kl=max_kl,
                       optimizer_steps=optimizer_steps, early_stopped=int(early_stopped),
                       learning_rate=self.optimizer.param_groups[0]["lr"])
        self.update_count += 1
        return metrics

    def train(self, output_dir: str | Path) -> dict:
        """每轮一个训练回合+PPO更新，然后按固定三种子监测；结束后评估最终模型。"""
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new directory")
        model_dir, logs_dir = output_dir / "model", output_dir / "logs"
        model_dir.mkdir(parents=True)
        logs_dir.mkdir()
        cfg = self.config
        write_json(output_dir / "config.json", cfg.to_dict())
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        print(f"Device: {self.device} ({device_name}); PyTorch {torch.__version__}", flush=True)
        started = time.perf_counter()
        starting_step, starting_update = self.global_step, self.update_count
        gradient_step = 0
        self.best_mean_return = None
        update_fields = ["round", "global_step", "update", "gradient_step", "policy_loss", "value_loss",
                         "entropy", "approx_kl", "clip_fraction", "explained_variance", "max_approx_kl",
                         "optimizer_steps", "early_stopped", "learning_rate"]
        round_fields = ["round", "training_episode_return", "training_episode_length",
                        "seed_1", "return_1", "seed_2", "return_2", "seed_3", "return_3",
                        "mean_return", "std_return"]
        with ExitStack() as stack:
            env = stack.enter_context(closing(gym.make(cfg.env_id)))
            capacity = getattr(env.spec, "max_episode_steps", None) or 1024
            buffer = RolloutBuffer(capacity, self.observation_dim,
                                   self.network.action_dim, self.network.is_discrete)
            episodes_log = stack.enter_context(closing(CSVLogger(logs_dir / "episodes.csv",
                ["round", "global_step", "episode_return", "episode_length"])))
            updates_log = stack.enter_context(closing(CSVLogger(logs_dir / "updates.csv", update_fields)))
            rounds_log = stack.enter_context(closing(CSVLogger(logs_dir / "training_rounds.csv", round_fields)))
            monitor_log = stack.enter_context(closing(CSVLogger(logs_dir / "monitor_episodes.csv",
                ["round", "seed", "episode_return", "episode_length"])))
            env.action_space.seed(cfg.seed)
            for round_number in range(1, cfg.train_rounds + 1):
                self.network.train()
                if cfg.anneal_lr:
                    lr = cfg.learning_rate * (1 - (round_number - 1) / cfg.train_rounds)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr
                observation, _ = env.reset(seed=cfg.seed if round_number == 1 else None)
                buffer.reset()
                episode_return, episode_length = 0.0, 0
                transitions, raw_actions, log_probs = [], [], []
                while True:
                    with torch.no_grad():
                        action, raw_action, log_prob = self.network.sample_actions(
                            self._observation_tensor(observation))
                    next_observation, reward, terminated, truncated, _ = env.step(self._environment_action(action[0]))
                    if not np.isfinite(reward):
                        raise FloatingPointError("Environment returned non-finite reward")
                    transitions.append((np.asarray(observation).reshape(-1).copy(), reward, terminated, truncated))
                    raw_actions.append(raw_action[0])
                    log_probs.append(log_prob[0])
                    self.global_step += 1
                    episode_return += float(reward)
                    episode_length += 1
                    observation = next_observation
                    if terminated or truncated:
                        break
                # 整回合采样期间网络不变：批量计算 value，缓存也只在回合结束统一传回。
                observations = np.stack([transition[0] for transition in transitions])
                with torch.no_grad():
                    values = self.network.value(torch.as_tensor(
                        observations, dtype=torch.float32, device=self.device)).cpu().numpy()
                    last_value = 0.0 if terminated else self.network.value(
                        self._observation_tensor(observation)).item()
                    sampled_raw = torch.stack(raw_actions).cpu().numpy()
                    sampled_log_probs = torch.stack(log_probs).cpu().numpy()
                for t, (state, reward, terminal, timeout) in enumerate(transitions):
                    next_value = values[t + 1] if t + 1 < len(values) else last_value
                    # 仅缩放训练目标；episode/monitor/final 日志保持环境原始回报。
                    buffer.add(state, sampled_raw[t], reward * cfg.reward_scale, values[t], next_value,
                               sampled_log_probs[t], terminal, timeout)
                episodes_log.write(dict(round=round_number, global_step=self.global_step,
                                        episode_return=episode_return, episode_length=episode_length))
                buffer.finish(cfg.gamma, cfg.gae_lambda)
                metrics = self.update(buffer.as_batch(self.device, cfg.normalize_advantage))
                gradient_step += metrics["optimizer_steps"]
                updates_log.write(dict(round=round_number, global_step=self.global_step,
                    update=self.update_count, gradient_step=gradient_step, **metrics))
                # 每轮相同的三个初始种子；监测只交互，不参与梯度更新。
                monitor = self.evaluate(episodes=3, seeds=cfg.monitor_seeds)
                row = dict(round=round_number, training_episode_return=episode_return,
                           training_episode_length=episode_length,
                           mean_return=monitor["mean_return"], std_return=monitor["std_return"])
                for i, seed in enumerate(cfg.monitor_seeds, 1):
                    row[f"seed_{i}"] = seed
                    row[f"return_{i}"] = monitor["returns"][i - 1]
                    monitor_log.write(dict(round=round_number, seed=seed,
                        episode_return=monitor["returns"][i - 1], episode_length=monitor["lengths"][i - 1]))
                rounds_log.write(row)
                if self.best_mean_return is None or monitor["mean_return"] >= self.best_mean_return:
                    self.best_mean_return = monitor["mean_return"]
                    self.save(model_dir / "best.pt")
                if round_number == 1 or round_number % 25 == 0 or round_number == cfg.train_rounds:
                    elapsed = time.perf_counter() - started
                    print(f"round={round_number:>3}/{cfg.train_rounds} "
                          f"return={monitor['mean_return']:.1f} +/-{monitor['std_return']:.1f} "
                          f"steps={self.global_step} elapsed={elapsed:.1f}s", flush=True)
            self.save(model_dir / "final.pt")
        # 当前网络就是第500轮训练后的final.pt，评估不再更新网络参数。
        final_eval = self.evaluate(episodes=cfg.final_eval_episodes, seed=cfg.final_eval_seed)
        write_json(logs_dir / "final_evaluation.json", {**final_eval, "checkpoint": "final.pt"})
        write_evaluation_episodes(logs_dir / "evaluation_episodes.csv", final_eval, checkpoint="final.pt")
        summary = {
            "env_id": cfg.env_id, "seed": cfg.seed, "device": str(self.device),
            "protocol": "episode_rounds_fixed_three_seeds",
            "training_rounds": cfg.train_rounds,
            "monitor_seeds": list(cfg.monitor_seeds), "monitor_episodes": 3 * cfg.train_rounds,
            "trained_steps_this_run": self.global_step - starting_step,
            "global_step": self.global_step, "updates_this_run": self.update_count - starting_update,
            "gradient_steps": gradient_step, "elapsed_seconds": time.perf_counter() - started,
            "final_evaluation": final_eval, "final_evaluation_checkpoint": "final.pt",
            "best_mean_return": self.best_mean_return,
        }
        write_json(output_dir / "summary.json", summary)
        update_readme_results()
        print(f"Final evaluation: {cfg.final_eval_episodes} episodes, "
              f"return={final_eval['mean_return']:.1f} +/-{final_eval['std_return']:.1f}", flush=True)
        return summary

    @torch.no_grad()
    def evaluate(self, episodes: int = 100, seed: int = 30_000, deterministic: bool = True,
                 render: bool = False, seeds: tuple[int, ...] | None = None) -> dict:
        if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if seeds is not None:
            if len(seeds) != episodes or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
                raise ValueError("Explicit seeds must match the episode count")
            episode_seeds = list(seeds)
            seed = episode_seeds[0]
        elif episodes > 0 and 0 <= seed < 2**32 - episodes:
            episode_seeds = list(range(seed, seed + episodes))
        else:
            raise ValueError("episodes must be positive and seed + episodes must be below 2**32")
        was_training = self.network.training
        self.network.eval()
        rewards, lengths = [0.0] * episodes, [0] * episodes
        devices = [self.device.index] if self.device.type == "cuda" else []
        try:
            # 独立环境与随机流，不让评估重置训练环境或消耗训练采样的 RNG。
            with torch.random.fork_rng(devices=devices), ExitStack() as stack:
                torch.random.default_generator.manual_seed(seed)
                if self.device.type == "cuda":
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(seed)
                # 各环境仍独立交互；每步把所有存活回合合成一次 GPU 推理和一次回传。
                batch_size = 1 if render else min(episodes, self.config.eval_batch_size)
                environments = [stack.enter_context(closing(gym.make(
                    self.config.env_id, render_mode="human" if render else None)))
                    for _ in range(batch_size)]
                if any(describe_spaces(env) != self.space_info for env in environments):
                    raise ValueError("Evaluation environment spaces do not match the checkpoint")
                for start in range(0, episodes, batch_size):
                    active = list(range(start, min(start + batch_size, episodes)))
                    observations = [environments[i - start].reset(seed=episode_seeds[i])[0] for i in active]
                    while active:
                        array = np.asarray(observations, dtype=np.float32).reshape(len(active), -1)
                        if array.shape[1] != self.observation_dim or not np.isfinite(array).all():
                            raise ValueError("Observation shape is incompatible or contains NaN/Inf")
                        actions = self.network.predict_actions(
                            torch.as_tensor(array, device=self.device), deterministic).cpu().numpy()
                        next_active, next_observations = [], []
                        for i, action in zip(active, actions):
                            if self.network.is_discrete:
                                env_action = int(action) + int(self.action_space.start)
                            else:
                                env_action = action.reshape(self.action_space.shape).astype(self.action_space.dtype)
                            observation, reward, terminated, truncated, _ = environments[i - start].step(env_action)
                            if not np.isfinite(reward):
                                raise FloatingPointError("Evaluation reward is NaN/Inf")
                            rewards[i] += float(reward)
                            lengths[i] += 1
                            if not (terminated or truncated):
                                next_active.append(i)
                                next_observations.append(observation)
                        active, observations = next_active, next_observations
        finally:
            self.network.train(was_training)
        return dict(env_id=self.config.env_id, episodes=episodes, seed=seed, seeds=episode_seeds, deterministic=deterministic,
                    mean_return=float(np.mean(rewards)), std_return=float(np.std(rewards)),
                    min_return=float(np.min(rewards)), max_return=float(np.max(rewards)),
                    mean_length=float(np.mean(lengths)), returns=rewards, lengths=lengths)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "config": self.config.to_dict(), "spaces": self.space_info,
                    "network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "global_step": self.global_step, "update_count": self.update_count,
                    "best_mean_return": self.best_mean_return}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda", load_optimizer: bool = False) -> "PPO":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        config_fields = {field.name for field in fields(PPOConfig)}
        config = PPOConfig(**{**{key: value for key, value in checkpoint["config"].items()
                                if key in config_fields}, "device": device})
        agent = cls(config)
        if checkpoint["spaces"] != agent.space_info:
            raise ValueError("Current environment spaces differ from the checkpoint")
        agent.network.load_state_dict(checkpoint["network"])
        if load_optimizer:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
            for group in agent.optimizer.param_groups:
                group["fused"] = agent.device.type == "cuda"
            # CPU 检查点的 Adam step 也是张量；融合 CUDA Adam 要求它随状态一起迁移。
            for state in agent.optimizer.state.values():
                for name, value in state.items():
                    if isinstance(value, Tensor):
                        state[name] = value.to(agent.device)
        agent.global_step = checkpoint["global_step"]
        agent.update_count = checkpoint["update_count"]
        agent.best_mean_return = checkpoint["best_mean_return"]
        return agent

