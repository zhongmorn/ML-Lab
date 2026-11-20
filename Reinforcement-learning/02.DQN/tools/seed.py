import random

import numpy as np
import torch


def set_seed(seed=42):
    """固定原有训练流程使用的随机数生成器。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
