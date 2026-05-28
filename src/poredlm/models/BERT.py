"""BERT-style contextual encoder for stride-level signal embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BERTConfig:
    """Configuration for the Stage 1 contextual encoder."""

    hidden_size: int = 256
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    dropout: float = 0.1
    max_position_embeddings: int = 4096


class BERTEncoder(nn.Module):
    """BERT-style encoder over continuous nanopore signal embeddings."""

    def __init__(self, config: BERTConfig | None = None) -> None:
        super().__init__()
        self.config = config or BERTConfig()
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings,
            self.config.hidden_size,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_size,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.intermediate_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode signal embeddings.

        Args:
            embeddings: Tensor with shape ``[batch, length, hidden_size]``.
            attention_mask: Optional tensor with shape ``[batch, length]`` where
                1 marks valid positions and 0 marks padding.
        """

        batch_size, seq_len, _ = embeddings.shape
        positions = torch.arange(seq_len, device=embeddings.device).expand(batch_size, seq_len)
        hidden_states = embeddings + self.position_embeddings(positions)
        hidden_states = self.dropout(self.layer_norm(hidden_states))

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        return self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
