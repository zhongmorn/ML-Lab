# 用 NumPy 存储 Q 表。
import numpy as np

# 取网格尺寸和动作表。
from environment.gridworld import MOVES, SIZE


# 定义表格型 Q-learning。
class QLearning:
    # 初始化学习参数和 Q 表。
    def __init__(self, learning_rate, discount):
        # 学习率控制更新幅度。
        self.learning_rate = learning_rate
        # 折扣因子调整未来奖励的权重。
        self.discount = discount
        # 初始 Q 值全为零。
        self.q_table = np.zeros((SIZE, SIZE, len(MOVES)))

    # 贪婪选择。
    def select_action(self, state):
        # 取最大 Q 值的动作，并列取第一个。
        return int(np.argmax(self.q_table[state]))

    # 更新本次动作的 Q 值。
    def update(self, state, action, reward, next_state, terminated):
        # 取行、列、动作索引。
        index = (*state, action)
        # TD 目标从即时奖励开始。
        target = reward
        # 终止后没有后续价值。
        if not terminated:
            # 加上折扣后的最大 Q 值。
            target += self.discount * np.max(self.q_table[next_state])
        # 按学习率乘 TD 误差更新。
        self.q_table[index] += self.learning_rate * (target - self.q_table[index])
