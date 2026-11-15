"""随机种子设置。"""

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 相同硬件/软件下提高可复现性；跨设备仍可能有数值差异。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

