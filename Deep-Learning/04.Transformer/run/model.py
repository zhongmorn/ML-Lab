import math

import torch
from torch import nn

from .config import MODEL_CONFIG


class CausalSelfAttention(nn.Module):

    def __init__(self, width, num_heads):
        super().__init__()
        if num_heads < 1 or width % num_heads != 0:
            raise ValueError("width must be divisible by a positive num_heads")
        # 将模型宽度均分到各注意力头，Q、K、V 通过一次线性投影共同生成。
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, x):
        # Q/K/V 拆成 [批大小, 头数, 序列长度, 每头维度]，便于各头独立计算。
        batch_size, length, width = x.shape
        qkv = self.qkv(x).reshape(batch_size, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # 缩放点积 QK^T / sqrt(d)；上三角遮罩禁止当前位置读取未来词元。
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        weights = scores.softmax(dim=-1)
        # 按注意力权重汇总 V，再合并各头并投影回模型宽度。
        context = weights @ value
        context = context.transpose(1, 2).reshape(batch_size, length, width)
        return self.projection(context)


class TransformerBlock(nn.Module):

    def __init__(self, width, num_heads):
        super().__init__()
        # 注意力与前馈网络均采用先归一化、后残差连接的结构。
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, num_heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


# 固定位置编码：偶数维用 sin，奇数维用 cos；截断可兼容奇数宽度。
def sinusoidal_positions(length, width):
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    even_dims = torch.arange(0, width, 2, dtype=torch.float32)
    angles = positions / (10000 ** (even_dims / width))
    pairs = torch.stack((angles.sin(), angles.cos()), dim=-1)
    return pairs.flatten(1)[:, :width]


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        width=MODEL_CONFIG["width"],
        num_heads=MODEL_CONFIG["num_heads"],
        max_length=MODEL_CONFIG["max_length"],
    ):
        super().__init__()
        # 词元嵌入叠加固定位置编码；位置缓冲区随设备迁移，但不写入检查点。
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.register_buffer("position_encoding", sinusoidal_positions(max_length, width), persistent=False)
        self.block = TransformerBlock(width, num_heads)
        self.output_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size)

    def embed(self, tokens):
        positions = self.position_encoding[:tokens.size(1)]
        return self.token_embedding(tokens) + positions

    def forward(self, tokens):
        # 经过 Transformer 块和归一化，为每个位置输出下一词元的词表分数。
        x = self.embed(tokens)
        x = self.block(x)
        x = self.output_norm(x)
        return self.lm_head(x)
