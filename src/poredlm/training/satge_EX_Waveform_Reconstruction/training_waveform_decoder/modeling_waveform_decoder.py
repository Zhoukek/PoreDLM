"""Waveform decoder trained on frozen PoreDLM hidden states."""

from __future__ import annotations

import torch
from torch import nn


class WaveformDecoder(nn.Module):
    """The same convolutional decoder architecture used by Stage 1 ``SignalCNN``."""

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                self.hidden_size,
                16,
                kernel_size=19,
                stride=5,
                padding=9,
                output_padding=1,
                bias=False,
            ),
            nn.SiLU(),
            nn.Conv1d(16, 4, kernel_size=5, padding=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(4, 1, kernel_size=5, padding=2, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, sequence, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        return self.decoder(hidden_states.transpose(1, 2))

    def initialize_from_stage1(self, stage1_model: nn.Module) -> None:
        source = getattr(getattr(stage1_model, "cnn_model", None), "decoder", None)
        if source is None:
            raise ValueError("Stage 1 model does not expose cnn_model.decoder.")
        self.decoder.load_state_dict(source.state_dict(), strict=True)

