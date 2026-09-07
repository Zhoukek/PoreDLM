from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_linear(layer: nn.Linear, *, zero: bool = False, std: Optional[float] = None) -> nn.Linear:
    if zero:
        nn.init.zeros_(layer.weight)
    elif std is not None:
        nn.init.normal_(layer.weight, std=std)
    else:
        nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int, prefix_length: int = 0, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.prefix_length = prefix_length
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[-2]
        content_length = max(sequence_length - self.prefix_length, 0)
        frequencies = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim)
        )
        positions = torch.arange(content_length, device=x.device, dtype=torch.float32)
        angles = torch.einsum("n,d->nd", positions, frequencies).repeat_interleave(2, dim=-1)
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        if self.prefix_length:
            cosine = torch.cat(
                [torch.ones(self.prefix_length, self.dim, device=x.device), cosine], dim=0
            )
            sine = torch.cat(
                [torch.zeros(self.prefix_length, self.dim, device=x.device), sine], dim=0
            )
        while cosine.ndim < x.ndim:
            cosine = cosine.unsqueeze(0)
            sine = sine.unsqueeze(0)
        return x * cosine.to(x.dtype) + rotate_half(x) * sine.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, size: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (self.weight * normalized).to(x.dtype)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        self.frequency_size = frequency_size
        self.input_projection = init_linear(nn.Linear(frequency_size, hidden_size), std=0.02)
        self.output_projection = init_linear(nn.Linear(hidden_size, hidden_size), std=0.02)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half, 1)
        )
        angles = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.frequency_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.output_projection(F.silu(self.input_projection(embedding)))


class BottleneckProjection(nn.Module):
    def __init__(self, input_size: int, output_size: int, bottleneck_size: int):
        super().__init__()
        self.down = init_linear(nn.Linear(input_size, bottleneck_size, bias=False))
        self.up = init_linear(nn.Linear(bottleneck_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float, projection_dropout: float):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qkv = init_linear(nn.Linear(hidden_size, hidden_size * 3))
        self.output = init_linear(nn.Linear(hidden_size, hidden_size))
        self.q_norm = RMSNorm(self.head_size)
        self.k_norm = RMSNorm(self.head_size)
        self.attention_dropout = attention_dropout
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_size)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = rotary(self.q_norm(query))
        key = rotary(self.k_norm(key))
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.projection_dropout(self.output(attended))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, dropout: float):
        super().__init__()
        inner_size = int(hidden_size * expansion * 2 / 3)
        self.gate_and_value = init_linear(nn.Linear(hidden_size, inner_size * 2))
        self.output = init_linear(nn.Linear(inner_size, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_and_value(x).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * value))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        attention_dropout: float,
        projection_dropout: float,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = Attention(hidden_size, num_heads, attention_dropout, projection_dropout)
        self.feed_forward_norm = RMSNorm(hidden_size)
        self.feed_forward = SwiGLU(hidden_size, expansion, projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rotary, attention_mask)
        return x + self.feed_forward(self.feed_forward_norm(x))

