"""Diffusion and flow matching modules for continuous signal embeddings."""

from __future__ import annotations

import torch
from torch import nn


class DiffusionLanguageModel(nn.Module):
    """Backbone scaffold for Stage 2 continuous embedding denoising."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, embeddings: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        del timesteps
        return self.proj(embeddings)
