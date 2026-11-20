# 导入 NumPy，用于随机探索和选择随机动作。
import numpy as np
# 导入 PyTorch，完成张量计算和梯度更新。
import torch
# 导入网络层和 Adam 优化器。
from torch import nn, optim
# 导入 ReLU 激活和均方误差函数。
from torch.nn import functional as F


# 定义将状态映射到各动作 Q 值的网络。
class Qnet(nn.Module):
    # 按状态维度、隐藏维度和动作数量创建网络。
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 初始化父类，注册网络参数与子层。
        super(Qnet, self).__init__()
        # 将状态特征映射到第一层隐藏表示。
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # 通过第二层线性映射提取隐藏特征。
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出各动作的 Q 值，不加激活以保留负值。
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    # 对单个状态或一批状态计算各动作 Q 值。
    def forward(self, x):
        # 对第一层输出应用 ReLU。
        x = F.relu(self.fc1(x))
        # 对第二层输出应用 ReLU。
        x = F.relu(self.fc2(x))
        # 返回批大小×动作数的 Q 值张量。
        return self.fc3(x)


# 定义分开选择动作和估计目标 Q 值的 Double DQN。
class DDQN:
    # 保存学习参数，并创建在线网络、目标网络和优化器。
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device):
        # 保存动作数量，用于随机探索。
        self.action_dim = action_dim
        # 保存未来奖励的折扣因子。
        self.gamma = gamma
        # 保存训练时随机探索的概率。
        self.epsilon = epsilon
        # 保存目标网络的同步间隔。
        self.target_update = target_update
        # 从零开始累计网络更新次数。
        self.count = 0
        # 保存张量和网络使用的计算设备。
        self.device = device

        # 创建用于选动作和梯度更新的在线 Q 网络。
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 创建用于计算 TD 目标的独立目标网络。
        self.target_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

        # 将在线网络参数完整复制到目标网络。
        self.target_net.load_state_dict(self.q_net.state_dict())
        # 将目标网络设置为评估模式。
        self.target_net.eval()

        # 用指定学习率的 Adam 更新在线网络参数。
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    # 训练时进行随机探索，否则选择当前 Q 值最大的动作。
    def take_action(self, state, is_training=True):
        # 说明动作选择接口及训练时的探索方式。
        """
        根据当前状态选择动作
        :param state: 当前状态
        :param is_training: 是否处于训练模式 (训练模式下使用 epsilon-greedy)
        """
        # 仅在训练阶段以 epsilon 概率进入随机探索。
        if is_training and np.random.random() < self.epsilon:
            # 在所有动作索引中均匀随机选择一个。
            return np.random.randint(self.action_dim)

        # 将状态转到目标设备，并补成 1×状态维度的输入。
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        # 关闭下面计算的梯度记录。
        with torch.no_grad():
            # 选择在线网络给出的最大 Q 值动作，并返回整数索引。
            return self.q_net(state).argmax().item()


    # 从一批回放样本计算 TD 损失并更新在线网络。
    def update(self, transition_dict):
        # 说明本函数负责损失计算与网络更新。
        """
        核心训练函数：计算 Loss 并更新网络
        此处已修改为 Double DQN 逻辑
        """
        # 将当前状态转为批大小×状态维度的浮点张量。
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        # 将动作转为批大小×1的整数索引列。
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        # 将即时奖励转为批大小×1的浮点列。
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        # 将下一状态转为批大小×状态维度的浮点张量。
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        # 将环境的真正终止标记转为批大小×1的零一浮点列。
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 从在线网络输出中取出每个样本实际动作的 Q 值。
        q_values = self.q_net(states).gather(1, actions)

        # 关闭下面计算的梯度记录。
        with torch.no_grad():

            # 用在线网络选择下一状态的最优动作，保留列维度。
            max_next_action = self.q_net(next_states).argmax(dim=1, keepdim=True)

            # 用目标网络评估在线网络选中的动作。
            max_next_q_values = self.target_net(next_states).gather(1, max_next_action)


            # 计算 TD 目标，真正终止时将后续价值置零。
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算预测 Q 值与 TD 目标之间的平均平方误差。
        loss = F.mse_loss(q_values, q_targets)

        # 清除上一批次留下的在线网络梯度。
        self.optimizer.zero_grad()
        # 对 TD 损失反向传播，计算参数梯度。
        loss.backward()
        # 按当前梯度更新在线网络参数。
        self.optimizer.step()

        # 当更新计数为同步间隔的整数倍时复制目标网络。
        if self.count % self.target_update == 0:
            # 将在线网络参数完整复制到目标网络。
            self.target_net.load_state_dict(self.q_net.state_dict())
        # 累加网络更新次数，供下一次同步判断使用。
        self.count += 1
