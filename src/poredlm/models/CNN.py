"""1D-CNN encoder blocks for nanopore raw signal."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple


class SignalCNN(nn.Module):
    """Nanopore 信号重建用纯卷积自编码器（无 VQ）。"""

    def __init__(self, cnn_type: Literal[0, 1] = 1) -> None:
        super().__init__()

        if cnn_type not in (0, 1):
            raise ValueError(f"`cnn_type` must be 0 or 1, got {cnn_type}.")

        self.cnn_type: int = cnn_type
        if cnn_type == 0:
            self._build_cnn_type0()
            self.out_channels = 768
            self.stride = 5
            self.receptive_field = 33
            self.RF = 33
        elif cnn_type == 1:
            pass
    
    def _build_cnn_type0(self) -> None:
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.SiLU(),

            nn.Conv1d(4, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.SiLU(),

            nn.Conv1d(16, 768, kernel_size=19, stride=5, padding=9, bias=False),
        )
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 768 → 16
            nn.ConvTranspose1d(
                in_channels=768,
                out_channels=16,
                kernel_size=19,
                stride=5,
                padding=9,
                output_padding=1,
                bias=False,
            ),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 16 → 4
            nn.Conv1d(16, 4, kernel_size=5, padding=2, bias=False),
            nn.SiLU(),
            
            # Inverse of encoder Layer 1: 4 → 1
            nn.Conv1d(4, 1, kernel_size=5, padding=2, bias=True)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signal to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to signal."""
        return self.decoder(z)
