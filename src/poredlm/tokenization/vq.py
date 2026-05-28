"""Vector quantization modules for stride-level signal representations."""

from __future__ import annotations

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """Minimal VQ codebook scaffold."""

    def __init__(self, num_codes: int, dim: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("VectorQuantizer forward pass is not implemented yet.")
