"""Decoders for sequence and modification prediction."""

from __future__ import annotations

import torch
from torch import nn


class BaseTokenDecoder(nn.Module):
    """Simple base-token decoder scaffold."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(hidden_states)
