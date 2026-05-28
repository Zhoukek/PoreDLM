"""Signal encoders used in Stage 1."""

from __future__ import annotations

import torch
from torch import nn
from vector_quantize_pytorch import VectorQuantize


class SignalCNNEncoder(nn.Module):
    """1D-CNN scaffold for raw nanopore current signals."""

    def __init__(self, in_channels: int = 1, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
        )


        

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.net(signal)
