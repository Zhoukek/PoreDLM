#!/usr/bin/env python
"""JAX/Flax T5 encoder used as a frozen text embedder."""

import logging

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from typing import Optional, Tuple, Any, Dict
from utils.logging_utils import log_for_0

# Type aliases
Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey


class T5LayerNorm(nn.Module):
    """T5-style layer normalization (RMSNorm without bias)."""

    epsilon: float = 1e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.epsilon)
        weight = self.param("weight", initializers.ones, (hidden_states.shape[-1],))
        return weight.astype(self.dtype) * hidden_states.astype(self.dtype)


class T5RelativePositionBias(nn.Module):
    """Compute relative position bias for T5 attention."""

    num_heads: int
    num_buckets: int = 32
    max_distance: int = 128
    bidirectional: bool = True
    d_model: int = 512
    embedding_init: Any = None  # will default to normal(stddev=d_model^{-0.5})

    @nn.compact
    def __call__(self, query_length: int, key_length: int):
        """Compute relative position bias.

        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence

        Returns:
            Relative position bias of shape [1, num_heads, query_length, key_length]
        """
        relative_position = self._compute_relative_position(query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(relative_position)

        # Shape: [num_buckets, num_heads]
        _init = self.embedding_init or initializers.normal(stddev=self.d_model ** -0.5)
        relative_attention_bias = self.param(
            "rel_embedding", _init, (self.num_buckets, self.num_heads)
        )

        # Shape: [query_length, key_length, num_heads]
        values = relative_attention_bias[relative_position_bucket]
        # Shape: [1, num_heads, query_length, key_length]
        values = jnp.transpose(values, (2, 0, 1))[None, ...]
        return values

    def _compute_relative_position(self, query_length: int, key_length: int):
        """Compute relative position matrix."""
        context_position = jnp.arange(query_length)[:, None]
        memory_position = jnp.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        return relative_position

    def _relative_position_bucket(self, relative_position):
        """Compute relative position bucket."""
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        relative_buckets = 0
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.minimum(relative_position, 0)

        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half use logarithmically bigger bins
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact + 1e-6)
            / jnp.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        relative_position_if_large = jnp.minimum(
            relative_position_if_large, num_buckets - 1
        )

        relative_buckets += jnp.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets.astype(jnp.int32)


class T5Attention(nn.Module):
    """T5 self-attention layer."""

    d_model: int
    d_kv: int
    num_heads: int
    dropout_rate: float = 0.0
    has_relative_attention_bias: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, 1, 1, seq_len]
            position_bias: [1, num_heads, seq_len, seq_len]
            deterministic: Whether to apply dropout

        Returns:
            (output, position_bias)
        """
        batch_size, seq_length, _ = hidden_states.shape

        # Linear projections (original T5 init)
        q = nn.Dense(
            self.num_heads * self.d_kv, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=(self.d_model * self.d_kv) ** -0.5),
            name="q",
        )(hidden_states)
        k = nn.Dense(
            self.num_heads * self.d_kv, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_model ** -0.5),
            name="k",
        )(hidden_states)
        v = nn.Dense(
            self.num_heads * self.d_kv, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_model ** -0.5),
            name="v",
        )(hidden_states)

        # Reshape to [batch, num_heads, seq_len, d_kv]
        q = q.reshape(batch_size, seq_length, self.num_heads, self.d_kv).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(batch_size, seq_length, self.num_heads, self.d_kv).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch_size, seq_length, self.num_heads, self.d_kv).transpose(
            0, 2, 1, 3
        )

        # Compute attention scores
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)

        # Compute position bias if needed
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = T5RelativePositionBias(
                num_heads=self.num_heads,
                d_model=self.d_model,
                bidirectional=True,
                name="relative_attention_bias",
            )(seq_length, seq_length)

        if position_bias is not None:
            scores = scores + position_bias

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and dropout
        attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
            self.dtype
        )
        attn_weights = nn.Dropout(rate=self.dropout_rate)(
            attn_weights, deterministic=deterministic
        )

        # Compute output
        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Reshape back to [batch, seq_len, d_model]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, -1
        )

        # Output projection (original T5 init)
        attn_output = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=(self.num_heads * self.d_kv) ** -0.5),
            name="o",
        )(attn_output)

        return attn_output, position_bias


class T5LayerSelfAttention(nn.Module):
    """T5 self-attention layer with layer norm and residual."""

    d_model: int
    d_kv: int
    num_heads: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    has_relative_attention_bias: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        # Pre-layer norm
        normed_hidden_states = T5LayerNorm(
            epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="layer_norm"
        )(hidden_states)

        # Self-attention
        attention_output, position_bias = T5Attention(
            d_model=self.d_model,
            d_kv=self.d_kv,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            has_relative_attention_bias=self.has_relative_attention_bias,
            dtype=self.dtype,
            name="SelfAttention",
        )(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            deterministic=deterministic,
        )

        # Dropout and residual
        attention_output = nn.Dropout(rate=self.dropout_rate)(
            attention_output, deterministic=deterministic
        )
        hidden_states = hidden_states + attention_output

        return hidden_states, position_bias


class T5DenseGatedActDense(nn.Module):
    """T5 feed-forward layer with gated activation (for T5 v1.1+)."""

    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
        # Gated linear unit (original T5 init)
        hidden_gelu = nn.Dense(
            self.d_ff, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_model ** -0.5),
            name="wi_0",
        )(hidden_states)
        # Use gelu_new (tanh approximation) to match PyTorch transformers
        hidden_gelu = nn.gelu(hidden_gelu, approximate=True)

        hidden_linear = nn.Dense(
            self.d_ff, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_model ** -0.5),
            name="wi_1",
        )(hidden_states)

        hidden_states = hidden_gelu * hidden_linear
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            hidden_states, deterministic=deterministic
        )

        # Down projection (original T5 init)
        hidden_states = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_ff ** -0.5),
            name="wo",
        )(hidden_states)

        return hidden_states


class T5DenseActDense(nn.Module):
    """T5 feed-forward layer (original T5)."""

    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
        # Up projection with ReLU (original T5 init)
        hidden_states = nn.Dense(
            self.d_ff, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_model ** -0.5),
            name="wi",
        )(hidden_states)
        hidden_states = nn.relu(hidden_states)
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            hidden_states, deterministic=deterministic
        )

        # Down projection (original T5 init)
        hidden_states = nn.Dense(
            self.d_model, use_bias=False, dtype=self.dtype,
            kernel_init=initializers.normal(stddev=self.d_ff ** -0.5),
            name="wo",
        )(hidden_states)

        return hidden_states


class T5LayerFF(nn.Module):
    """T5 feed-forward layer with layer norm and residual."""

    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True  # True for T5 v1.1+, False for original T5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
        # Pre-layer norm
        normed_hidden_states = T5LayerNorm(
            epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="layer_norm"
        )(hidden_states)

        # Feed-forward
        if self.is_gated_act:
            ff_output = T5DenseGatedActDense(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name="DenseReluDense",
            )(normed_hidden_states, deterministic=deterministic)
        else:
            ff_output = T5DenseActDense(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                name="DenseReluDense",
            )(normed_hidden_states, deterministic=deterministic)

        # Dropout and residual
        ff_output = nn.Dropout(rate=self.dropout_rate)(
            ff_output, deterministic=deterministic
        )
        hidden_states = hidden_states + ff_output

        return hidden_states


class T5EncoderOnlyBlock(nn.Module):
    """
    T5 block with only self-attention and feed-forward (no cross-attention).
    This is identical to the encoder block structure.
    """

    d_model: int
    d_kv: int
    d_ff: int
    num_heads: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    has_relative_attention_bias: bool = False
    is_gated_act: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states: Array,
        attention_mask: Optional[Array] = None,
        position_bias: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Tuple[Array, Optional[Array]]:
        # Self-attention
        hidden_states, position_bias = T5LayerSelfAttention(
            d_model=self.d_model,
            d_kv=self.d_kv,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            has_relative_attention_bias=self.has_relative_attention_bias,
            dtype=self.dtype,
            name="layer_0",
        )(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            deterministic=deterministic,
        )

        # Feed-forward
        hidden_states = T5LayerFF(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            is_gated_act=self.is_gated_act,
            dtype=self.dtype,
            name="layer_1",
        )(hidden_states, deterministic=deterministic)

        return hidden_states, position_bias


class T5EncoderLikeStack(nn.Module):
    """
    A T5 stack with encoder-like architecture (no cross-attention, no causal masking).
    Can be used as a decoder that mirrors the encoder structure.
    """

    num_layers: int
    d_model: int
    d_kv: int
    d_ff: int
    num_heads: int
    vocab_size: int
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    is_gated_act: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        inputs_embeds: Array,
        attention_mask: Optional[Array] = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            inputs_embeds: Input embeddings [batch, seq_len, d_model]
            attention_mask: Attention mask [batch, seq_len]
            deterministic: Whether to apply dropout

        Returns:
            Dictionary with 'last_hidden_state' and optionally 'hidden_states'
        """
        # Create extended attention mask if provided
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # 1D mask (B, L) -> (B, 1, 1, L)
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim == 3:
                # 2D mask (B, L, L) -> (B, 1, L, L)
                extended_attention_mask = attention_mask[:, None, :, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(
                self.dtype
            ).min
        else:
            extended_attention_mask = None

        # Dropout on input
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            inputs_embeds, deterministic=deterministic
        )

        # Process through blocks
        position_bias = None
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, position_bias = T5EncoderOnlyBlock(
                d_model=self.d_model,
                d_kv=self.d_kv,
                d_ff=self.d_ff,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                layer_norm_epsilon=self.layer_norm_epsilon,
                has_relative_attention_bias=(i == 0),
                is_gated_act=self.is_gated_act,
                dtype=self.dtype,
                name=f"block_{i}",
            )(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                deterministic=deterministic,
            )

        # Final layer norm
        hidden_states = T5LayerNorm(
            epsilon=self.layer_norm_epsilon, dtype=self.dtype, name="final_layer_norm"
        )(hidden_states)

        # Final dropout
        hidden_states = nn.Dropout(rate=self.dropout_rate)(
            hidden_states, deterministic=deterministic
        )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


class T5EncoderConfig:
    """Configuration class for T5Encoder."""

    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        is_gated_act: bool = True,
        dtype: Any = jnp.float32,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.is_gated_act = is_gated_act
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, model_name: str):
        """Create config from pretrained model name."""
        # Common T5 configurations
        configs = {
            "t5-small": {
                "vocab_size": 32128,
                "d_model": 512,
                "d_kv": 64,
                "d_ff": 2048,
                "num_layers": 6,
                "num_decoder_layers": 6,
                "num_heads": 8,
                "is_gated_act": False,
            },
            "t5-base": {
                "vocab_size": 32128,
                "d_model": 768,
                "d_kv": 64,
                "d_ff": 3072,
                "num_layers": 12,
                "num_decoder_layers": 12,
                "num_heads": 12,
                "is_gated_act": False,
            },
            "t5-large": {
                "vocab_size": 32128,
                "d_model": 1024,
                "d_kv": 64,
                "d_ff": 4096,
                "num_layers": 24,
                "num_decoder_layers": 24,
                "num_heads": 16,
                "is_gated_act": False,
            },
        }

        if model_name in configs:
            return cls(**configs[model_name])
        else:
            # Default to t5-small config
            log_for_0(f"Warning: Unknown model {model_name}, using t5-small config", level=logging.WARNING)
            return cls(**configs["t5-small"])


class T5Encoder(nn.Module):
    """JAX/Flax T5 encoder used as a frozen text embedder."""

    config: T5EncoderConfig

    def setup(self):
        self.shared = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            dtype=self.config.dtype,
            name="shared",
        )

    @nn.compact
    def __call__(
        self,
        input_ids: Array,
        attention_mask: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Array:
        """Encoder forward pass: returns last hidden state."""
        inputs_embeds = self.shared(input_ids)

        encoder_outputs = T5EncoderLikeStack(
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            d_kv=self.config.d_kv,
            d_ff=self.config.d_ff,
            num_heads=self.config.num_heads,
            vocab_size=self.config.vocab_size,
            dropout_rate=self.config.dropout_rate,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            is_gated_act=self.config.is_gated_act,
            dtype=self.config.dtype,
            name="encoder",
        )(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )
        return encoder_outputs["last_hidden_state"]


def init_t5_encoder(
    model: T5Encoder,
    rng: PRNGKey,
    max_seq_length: int = 128,
    batch_size: int = 1,
):
    """
    Initialize T5Encoder parameters.

    Args:
        model: T5Encoder instance
        rng: Random key for initialization
        max_seq_length: Maximum sequence length
        batch_size: Batch size for initialization

    Returns:
        Initialized parameters
    """
    # Create dummy inputs
    dummy_input_ids = jnp.ones((batch_size, max_seq_length), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((batch_size, max_seq_length), dtype=jnp.float32)

    # Initialize parameters
    params = model.init(
        rng,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        deterministic=True,
    )

    return params


def get_encoder(model_name: str, dtype):
    """Get encoder config and model."""

    log_for_0(f"Loading T5 Encoder: {model_name}...")
    config = T5EncoderConfig.from_pretrained(model_name)
    config.dtype = dtype
    model = T5Encoder(config=config)
    return config, model, init_t5_encoder


