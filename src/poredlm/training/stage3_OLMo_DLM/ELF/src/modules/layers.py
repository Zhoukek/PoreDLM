import math
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange, repeat


# Init defaults (matching PyTorch initialize_weights):
# - Dense kernels: xavier_uniform; biases: 0
# - TimestepEmbedder MLPs and learned tokens: normal(0.02)
# - final_layer.linear: 0 (zero init)
DEFAULT_KERNEL_INIT = nn.initializers.xavier_uniform()
DEFAULT_BIAS_INIT = nn.initializers.constant(0.0)
ZERO_INIT = nn.initializers.constant(0.0)
NORMAL_INIT_002 = nn.initializers.normal(stddev=0.02)


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x1 = x1.squeeze(-1)
    x2 = x2.squeeze(-1)
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d r -> ... (d r)')


class TextRotaryEmbeddingFast(nn.Module):
    """1D Rotary Position Embedding for text/sequence models in JAX/Flax."""
    dim: int
    pt_seq_len: int = 512
    ft_seq_len: Optional[int] = None
    theta: float = 10000
    num_empty_token: int = 0

    @nn.compact
    def __call__(self, t):
        dim = self.dim
        pt_seq_len = self.pt_seq_len
        ft_seq_len = self.ft_seq_len if self.ft_seq_len is not None else pt_seq_len

        # Compute frequencies
        freqs = 1. / (self.theta ** (jnp.arange(0, dim, 2)[:dim // 2].astype(jnp.float32) / dim))

        pos = jnp.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # 1D: position × frequency (no 2D grid like vision)
        freqs_main = jnp.einsum('..., f -> ... f', pos, freqs)
        freqs_main = repeat(freqs_main, '... n -> ... (n r)', r=2)

        D = freqs_main.shape[-1]
        cos_parts = []
        sin_parts = []

        # 1. Empty tokens (no rotation): cos=1, sin=0
        if self.num_empty_token > 0:
            cos_parts.append(jnp.ones((self.num_empty_token, D), dtype=freqs.dtype))
            sin_parts.append(jnp.zeros((self.num_empty_token, D), dtype=freqs.dtype))

        # 2. Main tokens (RoPE positions 0 to pt_seq_len-1)
        cos_parts.append(jnp.cos(freqs_main))
        sin_parts.append(jnp.sin(freqs_main))

        freqs_cos = jnp.concatenate(cos_parts, axis=0) if len(cos_parts) > 1 else cos_parts[0]
        freqs_sin = jnp.concatenate(sin_parts, axis=0) if len(sin_parts) > 1 else sin_parts[0]

        return t * freqs_cos + rotate_half(t) * freqs_sin


class RMSNorm(nn.Module):
    """RMS Normalization layer for JAX/Flax."""
    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, hidden_states):
        weight = self.param('weight', nn.initializers.ones, (self.hidden_size,))

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(hidden_states ** 2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
        return (weight * hidden_states).astype(input_dtype)


class BottleneckTextProj(nn.Module):
    """Text projection with bottleneck."""
    text_encoder_dim: int
    hidden_size: int
    bottleneck_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.bottleneck_dim, use_bias=False, kernel_init=DEFAULT_KERNEL_INIT, name='proj1')(x)
        return nn.Dense(
            self.hidden_size, use_bias=True,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT, name='proj2',
        )(x)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        dense = partial(
            nn.Dense, self.hidden_size, use_bias=True,
            kernel_init=NORMAL_INIT_002, bias_init=DEFAULT_BIAS_INIT,
        )
        t_emb = dense(name='mlp_0')(self.timestep_embedding(t, self.frequency_embedding_size))
        return dense(name='mlp_2')(nn.silu(t_emb))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Sinusoidal timestep embeddings: (N,) ints -> (N, dim) floats."""
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


def scaled_dot_product_attention(query, key, value, attn_mask=None):
    """Scaled dot-product attention.

    query/key/value: (B, num_heads, L|S, head_dim).
    attn_mask: optional int mask (B, S) or (B, L, S); 1=valid, 0=masked.
    Returns: (B, num_heads, L, head_dim).
    """
    scale_factor = 1 / math.sqrt(query.shape[-1])
    attn_weight = jnp.einsum(
        'bhld,bhsd->bhls', query.astype(jnp.float32), key.astype(jnp.float32),
    ) * scale_factor
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            mask = attn_mask[:, None, None, :]
        elif attn_mask.ndim == 3:
            mask = attn_mask[:, None, :, :]
        else:
            mask = attn_mask
        attn_weight = jnp.where(mask == 0, -1e9, attn_weight)
    attn_weight = jax.nn.softmax(attn_weight, axis=-1)
    return jnp.einsum('bhls,bhsd->bhld', attn_weight, value)


class Attention(nn.Module):
    """Multi-head self-attention."""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    qk_norm: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, rope_fn, attention_mask=None, deterministic=True):
        """x: (B, N, C). attention_mask: optional int mask (B, N), 1=valid, 0=padded."""
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        bias_init = DEFAULT_BIAS_INIT if self.qkv_bias else None
        qkv = nn.Dense(
            self.dim * 3, use_bias=self.qkv_bias,
            kernel_init=DEFAULT_KERNEL_INIT, bias_init=bias_init, name='qkv',
        )(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = RMSNorm(head_dim, name='q_norm')(q)
            k = RMSNorm(head_dim, name='k_norm')(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        x = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim, kernel_init=DEFAULT_KERNEL_INIT, bias_init=DEFAULT_BIAS_INIT, name='proj')(x)
        return nn.Dropout(rate=self.proj_drop, deterministic=deterministic)(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    dim: int
    hidden_dim: int
    drop: float = 0.0
    bias: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True):
        hidden_dim = int(self.hidden_dim * 2 / 3)
        bias_init = DEFAULT_BIAS_INIT if self.bias else None
        dense = partial(nn.Dense, use_bias=self.bias, kernel_init=DEFAULT_KERNEL_INIT, bias_init=bias_init)
        x12 = dense(2 * hidden_dim, name='w12')(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nn.Dropout(rate=self.drop, deterministic=deterministic)(nn.silu(x1) * x2)
        return dense(self.dim, name='w3')(hidden)


class FinalLayer(nn.Module):
    """The final layer of ELF."""
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = RMSNorm(self.hidden_size, name='norm_final')(x)
        return nn.Dense(
            self.patch_size * self.patch_size * self.out_channels, use_bias=True,
            kernel_init=ZERO_INIT, bias_init=ZERO_INIT, name='linear',
        )(x)
