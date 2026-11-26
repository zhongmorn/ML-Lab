from contextlib import closing
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from tools.plot import plot_evaluation_returns, plot_learning_curve, save_animation
from tools.seed import set_seed

from .agent import DQN
from .configs import load_config
from .replay import ReplayBuffer

# 实验日志、图片和权重保存在此目录。
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


class Trainer:
    def __init__(self, agent, env, replay_buffer, cfg):
        self.agent = agent
        self.env = env
        self.buffer = replay_buffer
        self.cfg = cfg
        self.episode_records = []

    def train(self):
        """按配置的训练轮次完成回合，优化器学习率保持固定。"""
        return_list = []
        self.episode_records = []
        train_episodes = self.cfg.train_episodes

        with tqdm(total=train_episodes, desc='Training', unit='episode') as pbar:
            for episode in range(1, train_episodes + 1):
                # 第一次重置固定环境种子，之后沿用环境的随机数序列。
                state, _ = self.env.reset(seed=self.cfg.seed if episode == 1 else None)
                done = False
                episode_return = 0
                episode_length = 0
                episode_epsilon = self.agent.epsilon

                while not done:
                    # 1. 智能体选择动作
                    action = self.agent.take_action(state)

                    # 2. 环境执行动作
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    # 核心交互：让智能体执行动作action，环境返回5个关键反馈（Gym 0.26+版本标准返回格式）
                    # 返回值说明（按顺序对应）：
                    # next_state: 执行动作后的下一状态（CartPole中为4维向量：小车位置、速度、摆杆角度、角速度）
                    # reward: 环境对当前动作的即时奖励（CartPole中每坚持1步得1.0分，鼓励平衡）
                    # terminated: 任务终止标志（bool）→ True=摆杆倾倒/小车越界（任务本身结束）
                    # truncated: 截断标志（bool）→ True=达到最大步数（外部限制强制结束）
                    # _: 占位符，忽略环境返回的额外辅助信息（无需使用）

                    # 3. 回放时只用真正终止标记屏蔽下一状态的 Q 值。
                    self.buffer.add(state, action, reward, next_state, terminated)

                    state = next_state
                    episode_return += reward
                    episode_length += 1

                    # 4. 如果经验池够了，开始训练
                    if self.buffer.size() > self.cfg.minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = self.buffer.sample(self.cfg.batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        self.agent.update(transition_dict)

                # 只衰减 epsilon 探索率，优化器的学习率不变。
                self.agent.epsilon = max(
                    self.cfg.epsilon_end,
                    self.agent.epsilon * self.cfg.epsilon_decay
                )

                # 记录数据与更新进度条
                return_list.append(episode_return)
                self.episode_records.append({
                    "episode": episode,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "epsilon": episode_epsilon,
                    "lr": self.agent.optimizer.param_groups[0]["lr"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
                pbar.update(1)
                pbar.set_postfix({
                    'episode': episode,
                    'return': '%.2f' % np.mean(return_list[-10:]), # 最近10次平均
                    'eps': '%.2f' % self.agent.epsilon
                }, refresh=False)

        return return_list

class Evaluator:
    def __init__(self, agent, env, cfg):
        self.agent = agent
        self.env = env
        self.cfg = cfg
        self.episode_records = []

    def eval(self, episodes=None):
        """评估模型表现"""
        episodes = self.cfg.eval_episodes if episodes is None else episodes
        print(f"\n开始评估 {self.cfg.env_name}，共 {episodes} 回合...")
        total_rewards = []
        self.episode_records = []

        for i in range(episodes):
            state, _ = self.env.reset(seed=self.cfg.eval_seed + i)
            done = False
            ep_reward = 0
            episode_length = 0

            while not done:
                # 评估时 is_training=False，不使用 epsilon-greedy，只选最优
                action = self.agent.take_action(state, is_training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
                episode_length += 1
                state = next_state

            total_rewards.append(ep_reward)
            self.episode_records.append({
                "episode": i + 1,
                "eval_seed": self.cfg.eval_seed + i,
                "episode_return": ep_reward,
                "episode_length": episode_length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
            print(f"评估回合 {i+1}/{episodes}: 回报 = {ep_reward:.2f}")

        print(f"平均回报: {np.mean(total_rewards):.2f}")
        return total_rewards

    def render_episode(self):
        """
        录制一个回合的动画帧
        注意：env 必须以 render_mode='rgb_array' 初始化
        """
        state, _ = self.env.reset(seed=self.cfg.eval_seed + self.cfg.eval_episodes)
        frames = [self.env.render()]
        done = False

        while not done:
            action = self.agent.take_action(state, is_training=False)
            state, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            frames.append(self.env.render())

        return frames


def run_experiment(env_name=None):
    """训练、评估并保存实验结果。"""
    cfg = load_config(env_name)
    env_name = cfg.env_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    actual_config = {**vars(cfg), "device": str(cfg.device)}
    print(f"正在训练 DQN | 环境: {env_name}")

    # 训练只用状态和奖励，不需要渲染。
    with closing(gym.make(env_name)) as env:
        set_seed(cfg.seed)
        agent = DQN(
            state_dim=env.observation_space.shape[0],
            hidden_dim=cfg.hidden_dim,
            action_dim=env.action_space.n,
            lr=cfg.lr,
            gamma=cfg.gamma,
            epsilon=cfg.epsilon_start,
            target_update=cfg.target_update,
            device=cfg.device,
        )
        buffer = ReplayBuffer(cfg.buffer_size)
        trainer = Trainer(agent, env, buffer, cfg)
        train_returns = trainer.train()

    torch.save(agent.q_net.state_dict(), OUTPUT_DIR / "weights.pt")

    plot_learning_curve(
        train_returns, env_name, "DQN",
        OUTPUT_DIR / "training.png",
    )

    # 用单独的环境评估，并在演示回合记录画面。
    with closing(gym.make(env_name, render_mode="rgb_array")) as env:
        evaluator = Evaluator(agent, env, cfg)
        eval_rewards = evaluator.eval()
        frames = evaluator.render_episode()

    save_animation(frames, OUTPUT_DIR / "demo.gif")
    plot_evaluation_returns(eval_rewards, env_name, OUTPUT_DIR / "evaluation.png")
    summary = {
        "algorithm": "DQN", "env_name": env_name,
        "training_episodes": len(train_returns),
        "training_steps": sum(row["episode_length"] for row in trainer.episode_records),
        "fixed_lr": cfg.lr,
        "observed_training_lrs": sorted({row["lr"] for row in trainer.episode_records}),
        "training_return_mean": float(np.mean(train_returns)),
        "training_last_50_return_mean": float(np.mean(train_returns[-50:])),
        "evaluation_episodes": len(eval_rewards),
        "evaluation_return_mean": float(np.mean(eval_rewards)),
        "evaluation_return_std": float(np.std(eval_rewards)),
        "evaluation_return_min": float(np.min(eval_rewards)),
        "evaluation_return_max": float(np.max(eval_rewards)),
        "evaluation_length_mean": float(np.mean([row["episode_length"] for row in evaluator.episode_records])),
        "evaluation_terminated": sum(row["terminated"] for row in evaluator.episode_records),
        "evaluation_truncated": sum(row["truncated"] for row in evaluator.episode_records),
        "training_reward_note": "Training and evaluation use raw environment rewards",
        "elapsed_seconds": perf_counter() - started,
    }
    results = {
        "config": actual_config,
        "summary": summary,
        "training": trainer.episode_records,
        "evaluation": evaluator.episode_records,
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return train_returns, eval_rewards
