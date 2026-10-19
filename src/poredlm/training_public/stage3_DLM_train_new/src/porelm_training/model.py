from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig
from .context_encoder import load_context_encoder
from .denoiser import DenoiserOutput, PoreLMDenoiser


class PoreLM(nn.Module):
    """Stage-3 model: a frozen signal encoder followed by a generative denoiser."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.context_encoder = load_context_encoder(config.context_encoder_path)
        self.freeze_context_encoder = config.freeze_context_encoder
        if self.freeze_context_encoder:
            self.context_encoder.requires_grad_(False)
            self.context_encoder.eval()
        latent_size = int(self.context_encoder.config.hidden_size)
        self.denoiser = PoreLMDenoiser(
            latent_size=latent_size,
            vocab_size=config.vocab_size,
            max_length=config.max_length,
            size=config.size,
            bottleneck_size=config.bottleneck_dim,
            num_time_tokens=config.num_time_tokens,
            num_self_condition_tokens=config.num_self_condition_tokens,
            num_mode_tokens=config.num_mode_tokens,
            attention_dropout=config.attention_dropout,
            projection_dropout=config.projection_dropout,
        )

    def train(self, mode: bool = True) -> "PoreLM":
        super().train(mode)
        if self.freeze_context_encoder:
            self.context_encoder.eval()
        return self

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_context_encoder else torch.enable_grad()
        with context:
            output = self.context_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return output.last_hidden_state.to(next(self.denoiser.parameters()).dtype)

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
        return self.denoiser(
            latent,
            timestep,
            attention_mask=attention_mask,
            guidance_scale=guidance_scale,
            decoder_active=decoder_active,
            representation_layer=representation_layer,
        )
