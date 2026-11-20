# 说明本模块包含的 PPO 网络、分布与训练组件。
"""PPO 的 Actor/Critic、动作分布、配置、轨迹、GAE、训练与评估。"""

# 导入数据类定义、字段枚举和字典转换工具。
from dataclasses import asdict, dataclass, fields
# 导入迭代器类型，用于标注小批次生成器。
from collections.abc import Iterator
# 导入上下文管理工具，用于关闭训练和评估环境。
from contextlib import ExitStack, closing
# 导入文件路径工具。
from pathlib import Path
# 导入 JSON 配置与结果的读写工具。
import json
# 导入数学常量和基础函数。
import math
# 导入计时工具。
import time

# 导入 Gymnasium 环境接口。
import gymnasium as gym
# 导入 NumPy 数组计算工具。
import numpy as np
# 导入 PyTorch 张量与自动求导工具。
import torch
# 导入张量类型和神经网络模块。
from torch import Tensor, nn
# 导入离散类别分布和连续正态分布。
from torch.distributions import Categorical, Normal
# 导入神经网络函数，用于稳定计算 softplus。
import torch.nn.functional as F

# 导入统一设置随机种子的函数。
from tools.seed import seed_everything
# 导入 JSON 保存函数。
from tools.save import write_json


# ==================== Actor/Critic 与动作分布 ====================

# 定义多层感知机工厂，接收输入、隐藏层和输出维度。
def make_mlp(input_dim: int, hidden_sizes: tuple[int, ...], output_dim: int,
             # 接收输出层的初始化增益，并声明返回顺序网络。
             output_gain: float) -> nn.Sequential:
    # 创建用于依次存放各层的列表。
    layers: list[nn.Module] = []
    # 记录当前层要接收的特征数。
    previous = input_dim
    # 依次建立配置中的各个隐藏层。
    for size in hidden_sizes:
        # 创建从上一层到当前隐藏层的线性映射。
        layer = nn.Linear(previous, size)
        # 用增益为根号二的正交矩阵初始化隐藏层权重。
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        # 将隐藏层偏置初始化为零。
        nn.init.zeros_(layer.bias)
        # 将线性层及其 Tanh 激活加入网络。
        layers.extend([layer, nn.Tanh()])
        # 将当前隐藏层宽度记录为下一层的输入维度。
        previous = size
    # 建立最后的输出线性层。
    output = nn.Linear(previous, output_dim)
    # 按指定增益正交初始化输出层权重。
    nn.init.orthogonal_(output.weight, output_gain)
    # 将输出层偏置初始化为零。
    nn.init.zeros_(output.bias)
    # 把输出层加入网络末尾。
    layers.append(output)
    # 按列表顺序组装并返回完整网络。
    return nn.Sequential(*layers)


# 定义输出离散动作分布的策略网络。
class DiscreteActor(nn.Module):
    # 接收观测维度、动作数量和隐藏层大小。
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        # 初始化 PyTorch 网络模块的内部状态。
        super().__init__()
        # 建立每个动作对应一个 logit 的策略网络。
        self.net = make_mlp(observation_dim, hidden_sizes, action_dim, output_gain=0.01)

    # 定义从观测生成离散动作分布的前向计算。
    def forward(self, observation: Tensor) -> Categorical:
        # 更新处检查损失有限性，避免分布校验在每次 CUDA 推理时同步。
        # 用网络 logits 构造类别分布，关闭逐次参数校验。
        return Categorical(logits=self.net(observation), validate_args=False)


# 定义输出连续动作正态分布的策略网络。
class ContinuousActor(nn.Module):
    # 接收观测维度和动作上下界。
    def __init__(self, observation_dim: int, low: np.ndarray, high: np.ndarray,
                 # 接收隐藏层大小，完成构造函数参数定义。
                 hidden_sizes: tuple[int, ...]):
        # 初始化 PyTorch 网络模块的内部状态。
        super().__init__()
        # 展平动作下界并转换为浮点张量。
        low = torch.as_tensor(np.asarray(low).reshape(-1), dtype=torch.float32)
        # 展平动作上界并转换为浮点张量。
        high = torch.as_tensor(np.asarray(high).reshape(-1), dtype=torch.float32)
        # 检查所有动作边界有限，且上界严格大于下界。
        if not torch.isfinite(low).all() or not torch.isfinite(high).all() or not torch.all(high > low):
            # 动作边界无效时报告配置错误。
            raise ValueError("Continuous actions require finite, strictly increasing bounds")
        # 建立输出各动作分量均值的网络。
        self.mean_net = make_mlp(observation_dim, hidden_sizes, low.numel(), output_gain=0.01)
        # 创建可训练的对数标准差，初始标准差为一。
        self.log_std = nn.Parameter(torch.zeros(low.numel()))
        # 保存动作半区间宽度，使其随模型迁移设备。
        self.register_buffer("action_scale", (high - low) / 2)
        # 保存动作区间中心，使其随模型迁移设备。
        self.register_buffer("action_bias", (high + low) / 2)

    # 定义从观测生成连续动作分布的前向计算。
    def forward(self, observation: Tensor) -> Normal:
        # 计算每个动作分量的均值。
        mean = self.mean_net(observation)
        # 限制对数标准差范围，取指数并扩展到批次形状。
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        # 用均值和标准差构造正态分布，关闭逐次参数校验。
        return Normal(mean, std, validate_args=False)

    # 定义从原始高斯动作到环境动作区间的变换。
    def transform(self, raw_action: Tensor) -> Tensor:
        # 用 Tanh 压缩动作，再缩放和平移到指定区间。
        return raw_action.tanh() * self.action_scale + self.action_bias

    # 定义变换后动作的联合对数概率计算。
    def log_prob(self, distribution: Normal, raw_action: Tensor) -> Tensor:
        # log(1-tanh(z)^2) 的稳定形式，z 很大时也不会 log(0)。
        # 用数值稳定的形式计算 Tanh 变换的对数雅可比。
        log_tanh_jacobian = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
        # 加入区间缩放的对数雅可比修正。
        correction = self.action_scale.log() + log_tanh_jacobian
        # 从高斯对数概率减去修正项，并对动作维度求和。
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)


# 定义预测状态价值的 Critic 网络。
class Critic(nn.Module):
    # 接收观测维度和隐藏层大小。
    def __init__(self, observation_dim: int, hidden_sizes: tuple[int, ...]):
        # 初始化 PyTorch 网络模块的内部状态。
        super().__init__()
        # 建立输出单个状态价值的网络。
        self.net = make_mlp(observation_dim, hidden_sizes, 1, output_gain=1.0)

    # 定义状态价值的前向计算。
    def forward(self, observation: Tensor) -> Tensor:
        # 计算价值并移除最后一个长度为一的维度。
        return self.net(observation).squeeze(-1)


# 定义组合策略网络和价值网络的模型。
class ActorCritic(nn.Module):
    # 说明批次输入形状及连续动作的内部表示。
    """输入 shape [batch, obs_dim]；连续动作始终在网络内部展平。"""

    # 接收观测维度和环境动作空间。
    def __init__(self, observation_dim: int, action_space: gym.Space,
                 # 接收默认两层各六十四单元的隐藏层配置。
                 hidden_sizes: tuple[int, ...] = (64, 64)):
        # 初始化 PyTorch 网络模块的内部状态。
        super().__init__()
        # 记录环境是否使用离散动作空间。
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        # 离散动作环境进入这一分支。
        if self.is_discrete:
            # 为离散动作建立对应动作数量的策略网络。
            self.actor = DiscreteActor(observation_dim, action_space.n, hidden_sizes)
            # 将每条离散动作记录为一个索引。
            self.action_dim = 1
        # 连续 Box 动作环境进入这一分支。
        elif isinstance(action_space, gym.spaces.Box):
            # 用环境动作边界建立连续策略网络。
            self.actor = ContinuousActor(observation_dim, action_space.low, action_space.high, hidden_sizes)
            # 计算展平后的连续动作维度。
            self.action_dim = int(np.prod(action_space.shape))
        # 其余动作空间进入错误处理分支。
        else:
            # 报告当前实现支持的动作空间类型。
            raise TypeError("Only Discrete and finite-bounded Box actions are supported")
        # 为两类动作环境建立同样的状态价值网络。
        self.critic = Critic(observation_dim, hidden_sizes)

    # 定义单独查询状态价值的方法。
    def value(self, observation: Tensor) -> Tensor:
        # 返回 Critic 对观测的价值估计。
        return self.critic(observation)

    # 定义确定性或随机的纯策略动作预测。
    def predict_actions(self, observation: Tensor, deterministic: bool = True) -> Tensor:
        # 说明此方法只执行 Actor 推理。
        """纯 Actor 推理：评估不计算 value、log_prob，也不构造确定性分布。"""
        # 需要确定性动作时进入这一分支。
        if deterministic:
            # 在离散动作环境中选择最大 logit 的动作。
            if self.is_discrete:
                # 返回策略网络评分最高的动作索引。
                return self.actor.net(observation).argmax(dim=-1)
            # 将连续动作均值变换到环境允许的区间。
            return self.actor.transform(self.actor.mean_net(observation))
        # 需要随机动作时，由策略网络构造动作分布。
        distribution = self.actor(observation)
        # 从策略分布采样一个原始动作。
        raw_action = distribution.sample()
        # 离散动作直接返回，连续动作先做区间变换。
        return raw_action if self.is_discrete else self.actor.transform(raw_action)

    # 定义轨迹采样所需的动作及其对数概率计算。
    def sample_actions(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        # 说明返回动作和旧策略概率，价值可另行批量计算。
        """轨迹采样的动作与旧概率；Critic 可在回合结束后批量计算。"""
        # 根据观测构造当前策略分布。
        distribution = self.actor(observation)
        # 离散动作使用类别分布的采样与概率计算。
        if self.is_discrete:
            # 确定性模式选择最大 logit，否则从分布采样。
            raw_action = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
            # 将离散动作索引直接作为环境动作。
            action = raw_action
            # 计算所选离散动作的对数概率。
            log_prob = distribution.log_prob(raw_action)
        # 连续动作使用正态分布及区间变换。
        else:
            # 确定性模式使用均值，否则采样原始高斯动作。
            raw_action = distribution.mean if deterministic else distribution.sample()
            # 将原始高斯动作变换到环境动作区间。
            action = self.actor.transform(raw_action)
            # 计算含变换修正的连续动作对数概率。
            log_prob = self.actor.log_prob(distribution, raw_action)
        # 返回环境动作、原始动作及其对数概率。
        return action, raw_action, log_prob

    # 定义同时返回动作、对数概率和状态价值的方法。
    def act(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # 调用动作采样函数获取动作及其概率。
        action, raw_action, log_prob = self.sample_actions(observation, deterministic)
        # 连同 Critic 价值一起返回采样结果。
        return action, raw_action, log_prob, self.critic(observation)

    # 定义更新时重新评估旧动作概率、熵和价值的方法。
    def evaluate_actions(self, observation: Tensor, raw_action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # 用当前策略参数重建动作分布。
        distribution = self.actor(observation)
        # 离散动作按类别分布计算统计量。
        if self.is_discrete:
            # 将动作转为整数索引并计算当前对数概率。
            log_prob = distribution.log_prob(raw_action.long())
            # 计算离散策略分布的解析熵。
            entropy = distribution.entropy()
        # 连续动作按变换后的高斯策略计算统计量。
        else:
            # 重新计算已存原始动作的修正对数概率。
            log_prob = self.actor.log_prob(distribution, raw_action)
            # 当前 squashed Gaussian 的 entropy；不是未变换高斯的解析 entropy。
            # 重参数化采样一个新高斯动作，用于估计当前策略熵。
            fresh_action = distribution.rsample()
            # 用新样本的负对数概率估计变换后策略的熵。
            entropy = -self.actor.log_prob(distribution, fresh_action)
        # 返回当前对数概率、熵和状态价值。
        return log_prob, entropy, self.critic(observation)


# ==================== 参数与设备 ====================

# 定义配置中的设备名称到 PyTorch 设备的转换。
def resolve_device(name: str) -> torch.device:
    # 自动选择设备时先判断 CUDA 是否可用。
    if name == "auto":
        # 有可用 CUDA 时选显卡，否则选择 CPU。
        name = "cuda" if torch.cuda.is_available() else "cpu"
    # 按设备名称构造 PyTorch 设备对象。
    device = torch.device(name)
    # CUDA 设备未指定编号时补充当前设备编号。
    if device.type == "cuda" and device.index is None:
        # 用当前 CUDA 设备编号构造完整设备对象。
        device = torch.device("cuda", torch.cuda.current_device())
    # 返回实际使用的设备。
    return device


# 让配置类自动生成初始化等数据类方法。
@dataclass
# 定义 PPO 训练与评估的参数集合。
class PPOConfig:
    # 设置默认训练环境。
    env_id: str = "CartPole-v1"
    # 设置默认训练轮数。
    train_rounds: int = 500
    # 设置每次梯度更新的小批次大小。
    batch_size: int = 64
    # 设置每轮对采集数据重复优化的次数。
    update_epochs: int = 10
    # 设置优化器始终使用的固定学习率。
    learning_rate: float = 3e-3
    # 设置进入优势与价值目标计算的奖励缩放系数。
    reward_scale: float = 1.0
    # 设置未来奖励的折扣因子。
    gamma: float = 0.99
    # 设置 GAE 优势递推的衰减参数。
    gae_lambda: float = 0.95
    # 设置 PPO 概率比率的裁剪幅度。
    clip_coef: float = 0.2
    # 设置价值损失在总损失中的权重。
    value_coef: float = 0.5
    # 设置熵奖励在总损失中的权重。
    entropy_coef: float = 0.0
    # 设置梯度裁剪的最大范数。
    max_grad_norm: float = 0.5
    # 设置提前结束策略更新的 KL 阈值。
    target_kl: float = 0.02
    # 设置是否对一轮数据的优势做标准化。
    normalize_advantage: bool = True
    # 设置策略网络与价值网络的隐藏层大小。
    hidden_sizes: tuple[int, ...] = (64, 64)
    # 设置训练过程中固定的三个监测种子。
    monitor_seeds: tuple[int, ...] = (10_000, 10_001, 10_002)
    # 设置独立最终评估的回合数。
    final_eval_episodes: int = 100
    # 设置最终评估回合种子的起点。
    final_eval_seed: int = 30_000
    # 设置评估时批量推理的环境数量。
    eval_batch_size: int = 32
    # 设置训练的基础随机种子。
    seed: int = 42
    # 设置默认计算设备。
    device: str = "cuda"
    # 设置 PyTorch 使用的 CPU 线程数。
    torch_threads: int = 1

    # 定义配置初始化后的类型整理与种子检查。
    def __post_init__(self) -> None:
        # 将隐藏层配置统一转换为元组。
        self.hidden_sizes = tuple(self.hidden_sizes)
        # 将监测种子统一转换为元组。
        self.monitor_seeds = tuple(self.monitor_seeds)
        # 三种子监测与独立终评对应 README 中的固定统计口径。
        # 检查监测种子恰好为三个且彼此不同。
        if len(set(self.monitor_seeds)) != 3 or len(self.monitor_seeds) != 3:
            # 监测种子数量或唯一性不满足要求时报告错误。
            raise ValueError("monitor_seeds must contain exactly three distinct seeds")
        # 检查最终评估种子是否与监测种子发生重叠。
        if set(range(self.final_eval_seed, self.final_eval_seed + self.final_eval_episodes)) & set(self.monitor_seeds):
            # 种子重叠时报告最终评估配置错误。
            raise ValueError("Final evaluation seeds must differ from the three monitoring seeds")

    # 定义将配置转换为可保存字典的方法。
    def to_dict(self) -> dict:
        # 将数据类的所有字段展开为字典。
        result = asdict(self)
        # 将隐藏层元组转换为便于 JSON 保存的列表。
        result["hidden_sizes"] = list(self.hidden_sizes)
        # 将监测种子元组转换为列表。
        result["monitor_seeds"] = list(self.monitor_seeds)
        # 返回整理后的配置字典。
        return result

    # 将下面的方法声明为配置类的类方法。
    @classmethod
    # 定义从 JSON 文件创建配置的方法。
    def from_json(cls, path: str | Path) -> "PPOConfig":
        # 读取并解析 JSON 配置。
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        # 忽略历史实验保存的学习率衰减选项。
        config.pop("anneal_lr", None)
        # 用当前配置字段构造配置。
        return cls(**config)



# ==================== Rollout 与 GAE ====================

# 定义从奖励与价值序列计算 GAE 的函数。
def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray,
                # 接收真实终止与时间截断标记。
                terminated: np.ndarray, truncated: np.ndarray,
                # 接收折扣和 GAE 衰减系数，返回优势与价值目标。
                gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    # 将五个输入序列统一转换为 NumPy 数组。
    arrays = [np.asarray(a) for a in (rewards, values, next_values, terminated, truncated)]
    # 将转换后的数组分别绑定到对应变量。
    rewards, values, next_values, terminated, truncated = arrays
    # 创建与奖励形状一致的浮点优势数组。
    advantages = np.zeros_like(rewards, dtype=np.float32)
    # 初始化逆序递推时下一步的优势为零。
    next_advantage = 0.0
    # 从序列末尾向前逐步计算优势。
    for t in reversed(range(len(rewards))):
        # 真实终止时关闭下一状态价值的 bootstrap。
        bootstrap = 1.0 - float(terminated[t])
        # 真实终止或时间截断时停止跨步优势递推。
        continue_trace = 1.0 - float(bool(terminated[t]) or bool(truncated[t]))
        # 计算当前步的时序差分残差。
        delta = float(rewards[t]) + gamma * bootstrap * float(next_values[t]) - float(values[t])
        # 将当前残差与折扣后的下一步优势相加。
        next_advantage = delta + gamma * gae_lambda * continue_trace * next_advantage
        # 保存当前步的优势估计。
        advantages[t] = next_advantage
    # 把优势加回旧价值，得到价值回归目标。
    returns = advantages + values.astype(np.float32)
    # 返回完整优势序列和价值目标序列。
    return advantages, returns



# 为轨迹批次自动生成字段初始化方法。
@dataclass
# 定义训练使用的张量批次结构。
class RolloutBatch:
    # 保存每一步的观测张量。
    observations: Tensor
    # 保存用于重新计算概率的原始动作。
    raw_actions: Tensor
    # 保存采样时旧策略的动作对数概率。
    old_log_probs: Tensor
    # 保存采样时的旧状态价值。
    old_values: Tensor
    # 保存每一步的优势估计。
    advantages: Tensor
    # 保存用于训练 Critic 的价值目标。
    returns: Tensor

    # 定义随机划分小批次的迭代器。
    def minibatches(self, batch_size: int) -> Iterator["RolloutBatch"]:
        # 在观测所在设备上生成随机排列的样本索引。
        indices = torch.randperm(len(self.observations), device=self.observations.device)
        # 按指定批大小遍历随机索引。
        for start in range(0, len(indices), batch_size):
            # 截取当前小批次对应的索引。
            subset = indices[start:start + batch_size]
            # 对所有字段取同样的样本子集并逐批返回。
            yield RolloutBatch(**{name: value[subset] for name, value in vars(self).items()})



# 定义暂存一个完整轨迹的缓冲区。
class RolloutBuffer:
    # 接收初始容量、观测维度、动作维度和动作类型。
    def __init__(self, capacity: int, observation_dim: int, action_dim: int, is_discrete: bool):
        # 记录当前轨迹缓冲区的容量。
        self.capacity = capacity
        # 为每一步的观测预分配浮点数组。
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        # 按离散索引或连续向量的形状分配原始动作数组。
        self.raw_actions = np.zeros((capacity,) if is_discrete else (capacity, action_dim),
                                    # 离散动作使用整数存储，连续动作使用浮点存储。
                                    dtype=np.int64 if is_discrete else np.float32)
        # 为每一步的奖励分配数组。
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # 为每一步的旧价值分配数组。
        self.values = np.zeros(capacity, dtype=np.float32)
        # 为每一步的下一状态价值分配数组。
        self.next_values = np.zeros(capacity, dtype=np.float32)
        # 为每一步的旧动作对数概率分配数组。
        self.old_log_probs = np.zeros(capacity, dtype=np.float32)
        # 为每一步的真实终止标记分配布尔数组。
        self.terminated = np.zeros(capacity, dtype=bool)
        # 为每一步的时间截断标记分配布尔数组。
        self.truncated = np.zeros(capacity, dtype=bool)
        # 将写入位置和计算结果初始化为空状态。
        self.reset()

    # 定义清空当前轨迹写入状态的方法。
    def reset(self) -> None:
        # 将下一条样本的写入位置重置为零。
        self.position = 0
        # 清除上一条轨迹的优势结果。
        self.advantages = None
        # 清除上一条轨迹的价值目标。
        self.returns = None

    # 定义添加单步轨迹样本的方法。
    def add(self, observation, raw_action, reward: float, value: float, next_value: float,
            # 接收旧动作概率及终止、截断标记。
            old_log_prob: float, terminated: bool, truncated: bool) -> None:
        # 缓冲区已满时先扩容。
        if self.position >= self.capacity:
            # 无固定时间上限的环境也能保存一个完整回合。
            # 依次扩容观测、动作、奖励和价值数组。
            for name in ("observations", "raw_actions", "rewards", "values", "next_values",
                         # 同时扩容旧概率及两个回合结束标记数组。
                         "old_log_probs", "terminated", "truncated"):
                # 取出当前要扩容的数组。
                array = getattr(self, name)
                # 分配容量翻倍且保留其余维度和数据类型的新数组。
                expanded = np.zeros((self.capacity * 2, *array.shape[1:]), dtype=array.dtype)
                # 将原数组内容复制到新数组的前半部分。
                expanded[:self.capacity] = array
                # 将扩容后的数组写回对应字段。
                setattr(self, name, expanded)
            # 将缓冲区容量记录更新为两倍。
            self.capacity *= 2
        # 记录当前样本要写入的位置。
        t = self.position
        # 保存当前观测。
        self.observations[t] = observation
        # 保存当前原始动作。
        self.raw_actions[t] = raw_action
        # 保存当前奖励。
        self.rewards[t] = reward
        # 保存当前状态的旧价值。
        self.values[t] = value
        # 保存下一状态的价值。
        self.next_values[t] = next_value
        # 保存旧策略对当前动作的对数概率。
        self.old_log_probs[t] = old_log_prob
        # 保存真实终止标记。
        self.terminated[t] = terminated
        # 保存时间截断标记。
        self.truncated[t] = truncated
        # 把写入位置移到下一个空位。
        self.position += 1

    # 定义采样结束后的优势与价值目标计算。
    def finish(self, gamma: float, gae_lambda: float) -> None:
        # 建立只覆盖已写入样本的切片。
        sl = slice(0, self.position)
        # 调用 GAE 并保存返回的优势和价值目标。
        self.advantages, self.returns = compute_gae(
            # 传入有效样本的奖励、旧价值和下一状态价值。
            self.rewards[sl], self.values[sl], self.next_values[sl],
            # 传入有效样本的结束标记与两个递推系数。
            self.terminated[sl], self.truncated[sl], gamma, gae_lambda)

    # 定义把轨迹转为指定设备训练张量的方法。
    def as_batch(self, device: torch.device, normalize_advantage: bool = True) -> RolloutBatch:
        # 建立当前有效轨迹范围的切片。
        sl = slice(0, self.position)
        # 复制优势数组并转换为目标设备上的张量。
        advantages = torch.as_tensor(self.advantages.copy(), device=device)
        # 启用标准化且样本数大于一时执行优势标准化。
        if normalize_advantage and len(advantages) > 1:
            # 减去优势均值并除以总体标准差，加小常数避免除零。
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        # 将各项张量打包成训练批次。
        return RolloutBatch(
            # 复制有效观测并转换为目标设备上的张量。
            observations=torch.as_tensor(self.observations[sl].copy(), device=device),
            # 复制有效原始动作并转换为目标设备上的张量。
            raw_actions=torch.as_tensor(self.raw_actions[sl].copy(), device=device),
            # 复制有效旧动作对数概率并转换为张量。
            old_log_probs=torch.as_tensor(self.old_log_probs[sl].copy(), device=device),
            # 复制有效旧价值并转换为目标设备上的张量。
            old_values=torch.as_tensor(self.values[sl].copy(), device=device),
            # 传入已处理的优势张量。
            advantages=advantages,
            # 复制价值目标并转换为目标设备上的张量。
            returns=torch.as_tensor(self.returns.copy(), device=device),
        # 完成轨迹训练批次的构造。
        )



# ==================== PPO 训练与评估 ====================

# 定义 PPO 裁剪策略目标对应的损失函数。
def ppo_policy_loss(new_log_prob: Tensor, old_log_prob: Tensor,
                    # 接收优势和裁剪系数，并声明返回损失张量。
                    advantages: Tensor, clip_coef: float) -> Tensor:
    # 用新旧对数概率之差的指数得到概率比率。
    ratio = (new_log_prob - old_log_prob).exp()
    # 计算未裁剪的概率比率乘优势。
    unclipped = ratio * advantages
    # 将概率比率裁剪到指定范围后乘优势。
    clipped = ratio.clamp(1 - clip_coef, 1 + clip_coef) * advantages
    # 逐样本取两个目标中的较小值，再取负平均用于最小化。
    return -torch.minimum(unclipped, clipped).mean()



# 定义记录环境观测与动作空间信息的函数。
def describe_spaces(env: gym.Env) -> dict:
    # 检查观测空间是否为连续 Box 类型。
    if not isinstance(env.observation_space, gym.spaces.Box):
        # 观测空间类型不支持时报告转换要求。
        raise TypeError("Observations must be a Box space; flatten/encode other spaces first")
    # 记录观测空间原始形状。
    result = {"observation_shape": list(env.observation_space.shape)}
    # 离散动作空间记录动作数量和起始索引。
    if isinstance(env.action_space, gym.spaces.Discrete):
        # 写入离散空间类型及动作数量。
        result["action"] = {"type": "Discrete", "n": int(env.action_space.n),
                            # 写入离散动作索引的起点，完成该空间记录。
                            "start": int(env.action_space.start)}
    # 连续 Box 动作空间记录形状及边界。
    elif isinstance(env.action_space, gym.spaces.Box):
        # 写入连续空间类型和动作形状。
        result["action"] = {"type": "Box", "shape": list(env.action_space.shape),
                            # 把动作下界展平并转换为普通列表。
                            "low": env.action_space.low.reshape(-1).tolist(),
                            # 把动作上界展平并转换为列表，完成该空间记录。
                            "high": env.action_space.high.reshape(-1).tolist()}
    # 其余动作空间进入错误处理分支。
    else:
        # 报告当前支持的动作空间类型。
        raise TypeError("Actions must be Discrete or finite-bounded Box")
    # 返回整理后的环境空间信息。
    return result


# 定义 PPO 智能体，管理网络、训练和评估。
class PPO:
    # 初始化智能体，接收本次实验配置。
    def __init__(self, config: PPOConfig):
        # 保存配置，供训练和评估共用。
        self.config = config
        # 设置 PyTorch 的 CPU 线程数。
        torch.set_num_threads(config.torch_threads)
        # 固定 Python、NumPy 和 PyTorch 随机种子。
        seed_everything(config.seed)
        # 解析配置指定的计算设备。
        self.device = resolve_device(config.device)
        # 临时创建环境以读取空间信息，结束后自动关闭。
        with closing(gym.make(config.env_id)) as probe:
            # 记录观测和动作空间，供加载检查点时核对。
            self.space_info = describe_spaces(probe)
            # 将观测形状换算为展平后的特征维数。
            self.observation_dim = int(np.prod(probe.observation_space.shape))
            # 保存环境动作空间及其范围。
            self.action_space = probe.action_space
            # 创建 Actor/Critic 网络并放到选定设备。
            self.network = ActorCritic(self.observation_dim, probe.action_space, config.hidden_sizes).to(self.device)

        # 为网络参数创建 Adam 优化器。
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate,
                                          # 设置数值稳定常数，在 CUDA 上启用融合 Adam。
                                          eps=1e-5, fused=self.device.type == "cuda")
        # 将累计环境交互步数清零。
        self.global_step = 0
        # 将累计 PPO 更新次数清零。
        self.update_count = 0
        # 初始化最佳监测回报，首次监测后再赋值。
        self.best_mean_return = None

    # 把单个环境观测转换为网络输入。
    def _observation_tensor(self, observation) -> Tensor:
        # 转换为 float32，并整理为一条展平的观测。
        array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        # 将观测张量放到计算设备。
        return torch.as_tensor(array, device=self.device)

    # 把网络动作转换为环境接受的格式。
    def _environment_action(self, action: Tensor):
        # 判断当前环境是否使用离散动作。
        if self.network.is_discrete:
            # 取出动作整数，并加上离散空间的起始编号。
            return int(action.item()) + int(self.action_space.start)
        # 把连续动作恢复为环境要求的形状和数据类型。
        return action.cpu().numpy().reshape(self.action_space.shape).astype(self.action_space.dtype)

    # 单次预测不记录梯度。
    @torch.no_grad()
    # 根据观测预测动作，可选择确定性策略。
    def predict(self, observation, deterministic: bool = True):
        # 转换观测后调用 Actor 预测。
        action = self.network.predict_actions(self._observation_tensor(observation), deterministic)
        # 取出批量中的唯一动作并转为环境格式。
        return self._environment_action(action[0])

    # 对一批固定轨迹执行多轮 PPO 更新。
    def update(self, batch: RolloutBatch) -> dict:
        # 文档说明旧概率、优势和回报在本轮更新中保持固定。
        """old_log_probs / advantages / returns 都是采样时固定的目标。"""
        # 将网络切换到训练模式。
        self.network.train()
        # 定义本轮要累计的五项优化指标。
        metric_names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")
        # 在设备上建立指标累计张量。
        weighted = torch.zeros(len(metric_names), device=self.device)
        # 初始化累计处理的样本数。
        total_samples = 0
        # 初始化实际参数更新次数。
        optimizer_steps = 0
        # 初始化 KL 提前停止标志。
        early_stopped = False
        # 初始化本轮最大近似 KL。
        max_kl = 0.0
        # 按配置重复使用这批轨迹。
        for _ in range(self.config.update_epochs):
            # 每轮打乱轨迹并分成小批量。
            for minibatch in batch.minibatches(self.config.batch_size):
                # 重新计算当前策略下的概率、熵和价值。
                new_log_prob, entropy, value = self.network.evaluate_actions(
                    # 传入小批量观测和采样时保存的原始动作。
                    minibatch.observations, minibatch.raw_actions)
                # 用新旧对数概率之差计算对数概率比。
                log_ratio = new_log_prob - minibatch.old_log_probs
                # 取指数得到新旧策略的概率比。
                ratio = log_ratio.exp()
                # 统计 KL 和裁剪比例时关闭梯度记录。
                with torch.no_grad():
                    # 计算非负估计形式的平均近似 KL。
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    # 统计概率比越过裁剪边界的样本比例。
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_coef).float().mean()

                # 取回 KL 数值，用于判断是否提前停止。
                kl_value = approx_kl.item()
                # 检查 KL 是否为有限数。
                if not math.isfinite(kl_value):
                    # KL 异常时停止，避免继续无效更新。
                    raise FloatingPointError("Non-finite KL; inspect log probabilities and learning rate")
                # 更新本轮出现的最大 KL。
                max_kl = max(max_kl, kl_value)

                # 在当前小批量更新前检查 KL 阈值。
                if self.config.target_kl > 0 and kl_value > 1.5 * self.config.target_kl:
                    # 记录本轮触发了提前停止。
                    early_stopped = True
                    # 退出当前小批量循环。
                    break
                # 开始计算 PPO 裁剪策略损失。
                policy_loss = ppo_policy_loss(new_log_prob, minibatch.old_log_probs,
                                              # 使用固定优势和配置中的裁剪宽度。
                                              minibatch.advantages, self.config.clip_coef)
                # 计算带二分之一系数的价值均方误差。
                value_loss = 0.5 * (value - minibatch.returns).square().mean()
                # 计算当前小批量的平均策略熵。
                mean_entropy = entropy.mean()
                # 合并策略损失、价值损失和熵奖励。
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * mean_entropy
                # 检查总损失是否为有限数。
                if not torch.isfinite(loss):
                    # 损失异常时停止更新。
                    raise FloatingPointError("Non-finite PPO loss")
                # 清空旧梯度，使用 None 减少清零开销。
                self.optimizer.zero_grad(set_to_none=True)
                # 从总损失反向计算参数梯度。
                loss.backward()
                # 按配置限制全部参数梯度的整体范数。
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm,
                                                # 遇到非有限梯度时直接报错。
                                                error_if_nonfinite=True)
                # 执行一次 Adam 参数更新。
                self.optimizer.step()
                # 取得当前小批量的样本数。
                n = len(minibatch.observations)
                # 开始累加脱离计算图的优化指标。
                weighted += torch.stack((policy_loss.detach(), value_loss.detach(),
                                         # 把五项指标按当前样本数加权。
                                         mean_entropy.detach(), approx_kl, clip_fraction)) * n
                # 累计本轮已处理的样本数。
                total_samples += n
                # 记录实际完成了一次参数更新。
                optimizer_steps += 1
            # 检查内层循环是否触发提前停止。
            if early_stopped:
                # 提前停止时也退出外层更新轮次。
                break
        # 检查本轮是否至少执行过一次更新。
        if total_samples == 0:
            # 无有效更新时报告旧概率或轨迹策略不一致。
            raise RuntimeError("No optimizer step was performed; rollout policy and old probabilities disagree")
        # 将样本加权均值转为可记录的指标字典。
        metrics = dict(zip(metric_names, (weighted / total_samples).tolist()))
        # 计算解释方差时关闭梯度记录。
        with torch.no_grad():
            # 计算回报目标的总体方差。
            variance = batch.returns.var(unbiased=False)
            # 仅在目标方差足够大时计算解释方差。
            explained_variance = torch.where(variance > 1e-8,
                # 用回报误差方差衡量旧价值预测的解释能力。
                1 - (batch.returns - batch.old_values).var(unbiased=False) / variance.clamp_min(1e-8),
                # 目标方差过小时记为零，并取出标量。
                torch.zeros_like(variance)).item()
        # 追加解释方差和最大 KL。
        metrics.update(explained_variance=explained_variance, max_approx_kl=max_kl,
                       # 追加更新次数及提前停止标志。
                       optimizer_steps=optimizer_steps, early_stopped=int(early_stopped),
                       # 记录本轮实际使用的学习率。
                       learning_rate=self.optimizer.param_groups[0]["lr"])
        # 累计一次 PPO 更新调用。
        self.update_count += 1
        # 返回本轮的全部优化指标。
        return metrics

    # 运行完整训练并保存实验结果。
    def train(self, output_dir: str | Path) -> dict:
        # 文档说明每轮训练、固定三种子监测和最终评估的顺序。
        """每轮一个训练回合+PPO更新，然后按固定三种子监测；结束后评估最终模型。"""
        # 将输出目录转换为路径对象。
        output_dir = Path(output_dir)
        # 检查目录中是否已有实验文件。
        if output_dir.exists() and any(output_dir.iterdir()):
            # 拒绝覆盖已有结果，要求使用新的运行目录。
            raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new directory")
        # 创建模型目录及所需父目录。
        model_dir = output_dir / "model"
        model_dir.mkdir(parents=True)
        # 用局部名称访问本次配置。
        cfg = self.config
        # 取得设备名称，CPU 直接显示名称。
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        # 打印设备和 PyTorch 版本。
        print(f"Device: {self.device} ({device_name}); PyTorch {torch.__version__}", flush=True)
        # 记录训练开始时间。
        started = time.perf_counter()
        # 保存运行前的累计交互和更新计数。
        starting_step, starting_update = self.global_step, self.update_count
        # 将本次累计梯度更新次数清零。
        gradient_step = 0
        # 重置本次训练的最佳监测回报。
        self.best_mean_return = None
        # 收集每轮训练、监测和优化记录，结束时写入同一个 JSON。
        training, episodes, monitor_episodes, updates = [], [], [], []
        with closing(gym.make(cfg.env_id)) as env:
            # 用环境时间上限确定轨迹缓存初始容量。
            capacity = getattr(env.spec, "max_episode_steps", None) or 1024
            # 开始创建保存完整回合的轨迹缓存。
            buffer = RolloutBuffer(capacity, self.observation_dim,
                                   # 传入动作维数和是否为离散动作。
                                   self.network.action_dim, self.network.is_discrete)
            # 固定环境动作空间的随机种子。
            env.action_space.seed(cfg.seed)
            # 依次执行配置规定的训练轮次。
            for round_number in range(1, cfg.train_rounds + 1):
                # 每轮采样前恢复训练模式。
                self.network.train()
                # 第一轮指定环境种子，后续沿用环境随机流。
                observation, _ = env.reset(seed=cfg.seed if round_number == 1 else None)
                # 清空上一轮的轨迹缓存。
                buffer.reset()
                # 初始化当前回合的回报和长度。
                episode_return, episode_length = 0.0, 0
                # 准备保存状态转移、原始动作和旧对数概率。
                transitions, raw_actions, log_probs = [], [], []
                # 持续交互到本回合结束。
                while True:
                    # 采样轨迹时不记录梯度。
                    with torch.no_grad():
                        # 开始根据当前策略采样动作。
                        action, raw_action, log_prob = self.network.sample_actions(
                            # 将当前观测转换成设备上的输入张量。
                            self._observation_tensor(observation))
                    # 执行环境动作，取得下一状态及终止标志。
                    next_observation, reward, terminated, truncated, _ = env.step(self._environment_action(action[0]))
                    # 检查环境奖励是否为有限数。
                    if not np.isfinite(reward):
                        # 奖励异常时停止本次训练。
                        raise FloatingPointError("Environment returned non-finite reward")
                    # 保存当前状态副本、奖励及两种结束标志。
                    transitions.append((np.asarray(observation).reshape(-1).copy(), reward, terminated, truncated))
                    # 保存采样动作变换前的值。
                    raw_actions.append(raw_action[0])
                    # 保存采样策略的对数概率。
                    log_probs.append(log_prob[0])
                    # 累计一次环境交互。
                    self.global_step += 1
                    # 将当前奖励加入本回合回报。
                    episode_return += float(reward)
                    # 增加当前回合的交互长度。
                    episode_length += 1
                    # 将下一观测作为后续输入。
                    observation = next_observation
                    # 检查任务终止或时间截断。
                    if terminated or truncated:
                        # 回合结束时退出交互循环。
                        break

                # 将当前回合的状态堆叠成观测矩阵。
                observations = np.stack([transition[0] for transition in transitions])
                # 回合结束后批量估计价值，不记录梯度。
                with torch.no_grad():
                    # 开始计算全部已采样状态的价值。
                    values = self.network.value(torch.as_tensor(
                        # 在设备上批量推理后，把价值结果取回 CPU。
                        observations, dtype=torch.float32, device=self.device)).cpu().numpy()
                    # 真正终止时尾部价值为零，否则估计最终观测价值。
                    last_value = 0.0 if terminated else self.network.value(
                        # 将结束前的观测转换为张量后取出其价值。
                        self._observation_tensor(observation)).item()
                    # 把采样的原始动作统一取回 CPU。
                    sampled_raw = torch.stack(raw_actions).cpu().numpy()
                    # 把采样的旧对数概率统一取回 CPU。
                    sampled_log_probs = torch.stack(log_probs).cpu().numpy()
                # 逐个处理本回合保存的状态转移。
                for t, (state, reward, terminal, timeout) in enumerate(transitions):
                    # 中间状态取下一项价值，末项使用尾部价值。
                    next_value = values[t + 1] if t + 1 < len(values) else last_value

                    # 开始写入轨迹，仅训练目标使用奖励缩放。
                    buffer.add(state, sampled_raw[t], reward * cfg.reward_scale, values[t], next_value,
                               # 同时保存旧概率和终止、截断标志。
                               sampled_log_probs[t], terminal, timeout)
                # 开始记录本轮实际交互信息。
                episodes.append(dict(round=round_number, global_step=self.global_step,
                                        # 写入未经缩放的回报和回合长度。
                                        episode_return=episode_return, episode_length=episode_length))
                # 计算本回合的 GAE 优势与回报目标。
                buffer.finish(cfg.gamma, cfg.gae_lambda)
                # 转为张量批次，按配置标准化优势并执行 PPO 更新。
                metrics = self.update(buffer.as_batch(self.device, cfg.normalize_advantage))
                # 累计本次实际执行的梯度更新次数。
                gradient_step += metrics["optimizer_steps"]
                # 开始记录本轮优化日志。
                updates.append(dict(round=round_number, global_step=self.global_step,
                    # 写入累计更新计数及优化指标。
                    update=self.update_count, gradient_step=gradient_step, **metrics))

                # 用固定三个种子评估当前策略，不参与训练更新。
                monitor = self.evaluate(episodes=3, seeds=cfg.monitor_seeds)
                # 开始汇总本轮训练回报。
                row = dict(round=round_number, training_episode_return=episode_return,
                           # 记录本轮训练回合长度。
                           training_episode_length=episode_length,
                           # 记录三个监测回合的回报均值和标准差。
                           mean_return=monitor["mean_return"], std_return=monitor["std_return"])
                # 依次处理三个固定监测种子。
                for i, seed in enumerate(cfg.monitor_seeds, 1):
                    # 保存当前监测种子。
                    row[f"seed_{i}"] = seed
                    # 保存该种子对应的回合回报。
                    row[f"return_{i}"] = monitor["returns"][i - 1]
                    # 开始记录单个监测回合。
                    monitor_episodes.append(dict(round=round_number, seed=seed,
                        # 写入该监测回合的真实回报和长度。
                        episode_return=monitor["returns"][i - 1], episode_length=monitor["lengths"][i - 1]))
                # 保存本轮监测汇总。
                training.append(row)
                # 比较当前监测均值与历史最佳值。
                if self.best_mean_return is None or monitor["mean_return"] >= self.best_mean_return:
                    # 更新最佳监测回报。
                    self.best_mean_return = monitor["mean_return"]
                    # 保存当前最佳模型检查点。
                    self.save(model_dir / "best.pt")
                # 在首轮、每 25 轮和末轮输出训练进度。
                if round_number == 1 or round_number % 25 == 0 or round_number == cfg.train_rounds:
                    # 计算已经用时。
                    elapsed = time.perf_counter() - started
                    # 开始打印当前轮次与总轮数。
                    print(f"round={round_number:>3}/{cfg.train_rounds} "
                          # 追加三个监测回合的均值和标准差。
                          f"return={monitor['mean_return']:.1f} +/-{monitor['std_return']:.1f} "
                          # 追加累计交互步数与耗时，并立即刷新。
                          f"steps={self.global_step} elapsed={elapsed:.1f}s", flush=True)
            # 训练完成后保存最终模型。
            self.save(model_dir / "final.pt")

        # 使用独立终评种子评估当前最终模型。
        final_eval = self.evaluate(episodes=cfg.final_eval_episodes, seed=cfg.final_eval_seed)
        # 开始构造本次训练的汇总信息。
        summary = {
            # 记录环境、训练种子和设备。
            "env_id": cfg.env_id, "seed": cfg.seed, "device": str(self.device),
            # 记录固定三种子逐轮监测的实验协议。
            "protocol": "episode_rounds_fixed_three_seeds",
            # 记录实际配置的训练轮数。
            "training_rounds": cfg.train_rounds,
            # 记录监测种子及累计监测回合数。
            "monitor_seeds": list(cfg.monitor_seeds), "monitor_episodes": 3 * cfg.train_rounds,
            # 计算本次新增的训练交互步数。
            "trained_steps_this_run": self.global_step - starting_step,
            # 记录累计交互步数和本次 PPO 更新次数。
            "global_step": self.global_step, "updates_this_run": self.update_count - starting_update,
            # 记录梯度更新次数与本次耗时。
            "gradient_steps": gradient_step, "elapsed_seconds": time.perf_counter() - started,
            # 记录本次最佳监测回报。
            "best_mean_return": self.best_mean_return,
        # 结束实验汇总字典。
        }
        # 将实验配置、汇总和原始记录集中保存。
        write_json(output_dir / "results.json", {
            "config": cfg.to_dict(),
            "summary": summary,
            "training": training,
            "episodes": episodes,
            "monitor": monitor_episodes,
            "updates": updates,
            "evaluation": {**final_eval, "checkpoint": "final.pt"},
        })
        # 开始打印最终评估回合数。
        print(f"Final evaluation: {cfg.final_eval_episodes} episodes, "
              # 追加最终回报均值和标准差，并立即刷新。
              f"return={final_eval['mean_return']:.1f} +/-{final_eval['std_return']:.1f}", flush=True)
        # 返回训练汇总及最终评估。
        return {**summary, "final_evaluation": final_eval, "final_evaluation_checkpoint": "final.pt"}

    # 评估策略时关闭梯度记录。
    @torch.no_grad()
    # 定义评估接口，设置回合数、种子和策略模式。
    def evaluate(self, episodes: int = 100, seed: int = 30_000, deterministic: bool = True,
                 # 支持渲染以及直接指定每个回合的种子。
                 render: bool = False, seeds: tuple[int, ...] | None = None) -> dict:
        # 使用指定种子列表，或生成连续评估种子。
        episode_seeds = list(seeds) if seeds is not None else list(range(seed, seed + episodes))
        # 把首个回合种子用于独立的策略随机流。
        seed = episode_seeds[0]
        # 保存评估前的网络模式。
        was_training = self.network.training
        # 切换到评估模式。
        self.network.eval()
        # 为各回合初始化回报和交互长度。
        rewards, lengths = [0.0] * episodes, [0] * episodes
        # 确定需要暂存随机状态的 CUDA 设备。
        devices = [self.device.index] if self.device.type == "cuda" else []
        # 确保评估结束后恢复网络模式。
        try:

            # 暂存随机状态并统一管理评估环境的关闭。
            with torch.random.fork_rng(devices=devices), ExitStack() as stack:
                # 设置评估使用的 CPU 随机种子。
                torch.random.default_generator.manual_seed(seed)
                # CUDA 推理还需设置对应设备的随机流。
                if self.device.type == "cuda":
                    # 切换到当前智能体使用的 CUDA 设备。
                    with torch.cuda.device(self.device):
                        # 固定该设备的评估采样种子。
                        torch.cuda.manual_seed(seed)

                # 渲染时串行评估，其余情况按配置合批。
                batch_size = 1 if render else min(episodes, self.config.eval_batch_size)
                # 开始创建这批独立评估环境。
                environments = [stack.enter_context(closing(gym.make(
                    # 按需打开环境渲染窗口。
                    self.config.env_id, render_mode="human" if render else None)))
                    # 创建足够数量的环境供批量评估使用。
                    for _ in range(batch_size)]
                # 按环境批量依次评估所有回合。
                for start in range(0, episodes, batch_size):
                    # 取得本批尚未结束的回合编号。
                    active = list(range(start, min(start + batch_size, episodes)))
                    # 分别按各自种子重置环境并取得初始观测。
                    observations = [environments[i - start].reset(seed=episode_seeds[i])[0] for i in active]
                    # 只要本批仍有活跃回合就继续交互。
                    while active:
                        # 将活跃回合的观测堆叠并展平为 float32。
                        array = np.asarray(observations, dtype=np.float32).reshape(len(active), -1)
                        # 开始批量预测当前策略动作。
                        actions = self.network.predict_actions(
                            # 使用所选策略模式推理，并将动作取回 CPU。
                            torch.as_tensor(array, device=self.device), deterministic).cpu().numpy()
                        # 准备保存下一步仍活跃的回合和观测。
                        next_active, next_observations = [], []
                        # 逐个将批量动作交给对应环境。
                        for i, action in zip(active, actions):
                            # 判断当前动作是否为离散动作。
                            if self.network.is_discrete:
                                # 将网络动作编号换算为环境动作编号。
                                env_action = int(action) + int(self.action_space.start)
                            # 其余情况按连续动作处理。
                            else:
                                # 恢复连续动作的原始形状及数据类型。
                                env_action = action.reshape(self.action_space.shape).astype(self.action_space.dtype)
                            # 执行当前回合的一步环境交互。
                            observation, reward, terminated, truncated, _ = environments[i - start].step(env_action)
                            # 检查评估奖励是否为有限数。
                            if not np.isfinite(reward):
                                # 奖励异常时停止评估。
                                raise FloatingPointError("Evaluation reward is NaN/Inf")
                            # 累计当前回合的原始奖励。
                            rewards[i] += float(reward)
                            # 累计当前回合的长度。
                            lengths[i] += 1
                            # 只保留尚未终止或截断的回合。
                            if not (terminated or truncated):
                                # 保存仍需继续交互的回合编号。
                                next_active.append(i)
                                # 保存该回合的下一观测。
                                next_observations.append(observation)
                        # 用仍活跃的回合更新下一步推理批次。
                        active, observations = next_active, next_observations
        # 无论正常结束还是异常退出都执行恢复。
        finally:
            # 恢复评估前的训练或评估模式。
            self.network.train(was_training)
        # 返回环境、回合种子及策略模式。
        return dict(env_id=self.config.env_id, episodes=episodes, seed=seed, seeds=episode_seeds, deterministic=deterministic,
                    # 记录逐回合回报的均值和标准差。
                    mean_return=float(np.mean(rewards)), std_return=float(np.std(rewards)),
                    # 记录最小和最大回合回报。
                    min_return=float(np.min(rewards)), max_return=float(np.max(rewards)),
                    # 记录平均长度及完整逐回合回报和长度。
                    mean_length=float(np.mean(lengths)), returns=rewards, lengths=lengths)

    # 保存模型参数、优化器状态和训练计数。
    def save(self, path: str | Path) -> None:
        # 把检查点路径转换为路径对象。
        path = Path(path)
        # 创建检查点所需的父目录。
        path.parent.mkdir(parents=True, exist_ok=True)
        # 开始保存格式版本、配置和环境空间信息。
        torch.save({"format_version": 1, "config": self.config.to_dict(), "spaces": self.space_info,
                    # 保存网络参数与优化器状态。
                    "network": self.network.state_dict(), "optimizer": self.optimizer.state_dict(),
                    # 保存累计交互步数和 PPO 更新次数。
                    "global_step": self.global_step, "update_count": self.update_count,
                    # 保存最佳监测回报并写入指定路径。
                    "best_mean_return": self.best_mean_return}, path)

    # 将加载方法定义为类方法。
    @classmethod
    # 加载已有检查点，可按需恢复优化器。
    def load(cls, path: str | Path, device: str = "cuda", load_optimizer: bool = False) -> "PPO":
        # 先将检查点字典读入 CPU。
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # 核对检查点格式版本。
        if checkpoint.get("format_version") != 1:
            # 拒绝不支持的检查点格式。
            raise ValueError("Unsupported checkpoint format")
        # 取得当前配置类支持的字段名。
        config_fields = {field.name for field in fields(PPOConfig)}
        # 开始从检查点配置中构造当前配置。
        config = PPOConfig(**{**{key: value for key, value in checkpoint["config"].items()
                                # 保留支持的字段，并采用指定的加载设备。
                                if key in config_fields}, "device": device})
        # 按恢复的配置初始化智能体。
        agent = cls(config)
        # 比较当前环境空间与保存时是否一致。
        if checkpoint["spaces"] != agent.space_info:
            # 拒绝不匹配的环境检查点。
            raise ValueError("Current environment spaces differ from the checkpoint")
        # 加载保存的网络参数。
        agent.network.load_state_dict(checkpoint["network"])
        # 判断是否同时恢复训练用优化器。
        if load_optimizer:
            # 恢复 Adam 的内部状态。
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
            # 遍历优化器参数组。
            for group in agent.optimizer.param_groups:
                # 恢复训练时使用配置的固定学习率。
                group["lr"] = config.learning_rate
                # 按当前设备设置是否使用融合 Adam。
                group["fused"] = agent.device.type == "cuda"

            # 遍历已恢复的优化器状态。
            for state in agent.optimizer.state.values():
                # 逐项检查状态中的数据。
                for name, value in state.items():
                    # 识别需要迁移设备的张量状态。
                    if isinstance(value, Tensor):
                        # 把状态张量移到智能体设备。
                        state[name] = value.to(agent.device)
        # 恢复累计环境交互步数。
        agent.global_step = checkpoint["global_step"]
        # 恢复累计 PPO 更新次数。
        agent.update_count = checkpoint["update_count"]
        # 恢复最佳监测回报。
        agent.best_mean_return = checkpoint["best_mean_return"]
        # 返回已加载的智能体。
        return agent
