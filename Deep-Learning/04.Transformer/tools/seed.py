import torch

from run.configs import CPU_THREADS


# 固定随机种子与 CPU 线程数，便于复现实验。
def set_seed(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(CPU_THREADS)
