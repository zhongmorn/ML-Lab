import random

import numpy as np
import torch


def set_seed(seed=42):
    # 固定模型初始化、动作探索和经验回放采样用到的随机数。
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
