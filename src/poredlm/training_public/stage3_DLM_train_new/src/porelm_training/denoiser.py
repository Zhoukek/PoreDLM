from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    BottleneckProjection,
    RMSNorm,
    RotaryEmbedding,
    TimestepEmbedding,
    TransformerBlock,
    init_linear,
)


@dataclass(frozen=True)
class DenoiserSize:
    hidden_size: int
    depth: int
    num_heads: int


MODEL_SIZES = {
    "base": DenoiserSize(hidden_size=768, depth=12, num_heads=12),
    "medium": DenoiserSize(hidden_size=1056, depth=24, num_heads=16),
    "large": DenoiserSize(hidden_size=1280, depth=32, num_heads=16),
}


@dataclass
class DenoiserOutput:
    prediction: torch.Tensor
    logits: Optional[torch.Tensor]
    representation: Optional[torch.Tensor] = None


class PoreLMDenoiser(nn.Module):
    """Transformer denoiser conditioned by prefix timestep tokens."""

    def __init__(
        self,
        *,
        latent_size: int,
        vocab_size: int,
        max_length: int,
        size: str = "base",
        bottleneck_size: int = 128,
        num_time_tokens: int = 4,
        num_self_condition_tokens: int = 4,
        num_mode_tokens: int = 4,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        dimensions = MODEL_SIZES[size]
        self.latent_size = latent_size
        self.hidden_size = dimensions.hidden_size
        self.num_heads = dimensions.num_heads
        self.max_length = max_length
        self.num_time_tokens = num_time_tokens
        self.num_self_condition_tokens = num_self_condition_tokens
        self.num_mode_tokens = num_mode_tokens

        self.self_condition_projection = init_linear(nn.Linear(latent_size * 2, latent_size))
        self.input_projection = BottleneckProjection(latent_size, self.hidden_size, bottleneck_size)
        self.timestep_embedding = TimestepEmbedding(self.hidden_size)
        self.guidance_embedding = (
            TimestepEmbedding(self.hidden_size) if num_self_condition_tokens > 0 else None
        )
        self.time_tokens = nn.Parameter(torch.randn(1, num_time_tokens, self.hidden_size) * 0.02)
        self.self_condition_tokens = nn.Parameter(
            torch.randn(1, num_self_condition_tokens, self.hidden_size) * 0.02
        )
        self.mode_tokens = nn.Parameter(torch.randn(1, num_mode_tokens, self.hidden_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_size,
                    dimensions.num_heads,
                    4.0,
                    attention_dropout if dimensions.depth // 4 <= i < dimensions.depth * 3 // 4 else 0.0,
                    projection_dropout if dimensions.depth // 4 <= i < dimensions.depth * 3 // 4 else 0.0,
                )
                for i in range(dimensions.depth)
            ]
        )
        self.output_norm = RMSNorm(self.hidden_size)
        self.output_projection = init_linear(nn.Linear(self.hidden_size, latent_size), zero=True)
        self.decoder_projection = init_linear(nn.Linear(self.hidden_size, latent_size))
        self.token_head = init_linear(nn.Linear(latent_size, vocab_size))

    def _prefix(
        self,
        timestep: torch.Tensor,
        guidance_scale: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch = timestep.shape[0]
        if self.num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive")
        pieces = [
            self.time_tokens.expand(batch, -1, -1) + self.timestep_embedding(timestep).unsqueeze(1)
        ]
        if guidance_scale is not None and self.num_self_condition_tokens:
            assert self.guidance_embedding is not None
            pieces.append(
                self.self_condition_tokens.expand(batch, -1, -1)
                + self.guidance_embedding(guidance_scale).unsqueeze(1)
            )
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
        decoder_active: bool | torch.Tensor = False,
        representation_layer: Optional[int] = None,
    ) -> DenoiserOutput:
        if latent.shape[-1] == self.latent_size * 2:
            latent = self.self_condition_projection(latent)
        x = self.input_projection(latent)
        batch = x.shape[0]

        active = bool(decoder_active.item()) if torch.is_tensor(decoder_active) else bool(decoder_active)
        mode_length = self.num_mode_tokens
        if mode_length:
            mode = self.mode_tokens.expand(batch, -1, -1) * float(active)
            x = torch.cat([mode, x], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask.new_ones(batch, mode_length), attention_mask], dim=1
                )

        prefix = self._prefix(timestep, guidance_scale)
        prefix_length = prefix.shape[1]
        x = torch.cat([prefix, x], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask.new_ones(batch, prefix_length), attention_mask], dim=1
            )
        rotary = RotaryEmbedding(
            self.hidden_size // self.num_heads,
            self.max_length,
            prefix_length=prefix_length + mode_length,
        )
        representation = None
        if representation_layer is not None and not -len(self.blocks) <= representation_layer < len(self.blocks):
            raise ValueError(f"representation_layer must be in [{-len(self.blocks)}, {len(self.blocks) - 1}]")
        selected_layer = representation_layer % len(self.blocks) if representation_layer is not None else None
        for layer_index, block in enumerate(self.blocks):
            x = block(x, rotary, attention_mask)
            if layer_index == selected_layer:
                representation = x[:, prefix_length + mode_length :]
        x = x[:, prefix_length + mode_length :]

        clean_latent = self.output_projection(self.output_norm(x))
        logits = self.token_head(F.gelu(self.decoder_projection(x))) if active else None
        return DenoiserOutput(clean_latent, logits, representation)
