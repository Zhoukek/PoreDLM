"""BERT-style contextual encoder for continuous signal embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ContinuousBertOutput:
    loss: Optional[torch.Tensor]
    predictions: torch.Tensor
    last_hidden_state: torch.Tensor


class ContinuousBertConfig:
    def __init__(self, feature_dim: int = 512, hidden_size: int = 512,
                 num_hidden_layers: int = 12, num_attention_heads: int = 8,
                 intermediate_size: int = 2048, dropout: float = 0.1,
                 max_position_embeddings: int = 1536, mask_probability: float = 0.15):
        self.feature_dim = int(feature_dim); self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers); self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size); self.dropout = float(dropout)
        self.max_position_embeddings = int(max_position_embeddings); self.mask_probability = float(mask_probability)


class ContinuousBert(nn.Module):
    def __init__(self, config: ContinuousBertConfig | None = None):
        super().__init__()
        self.config = config or ContinuousBertConfig()
        c = self.config
        self.input_projection = nn.Linear(c.feature_dim, c.hidden_size) if c.feature_dim != c.hidden_size else nn.Identity()
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.hidden_size)
        self.mask_embedding = nn.Parameter(torch.zeros(c.hidden_size))
        self.input_norm = nn.LayerNorm(c.hidden_size)
        self.dropout = nn.Dropout(c.dropout)
        layer = nn.TransformerEncoderLayer(d_model=c.hidden_size, nhead=c.num_attention_heads,
                                           dim_feedforward=c.intermediate_size, dropout=c.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=c.num_hidden_layers)
        self.final_norm = nn.LayerNorm(c.hidden_size)
        self.prediction_head = nn.Sequential(nn.Linear(c.hidden_size, c.hidden_size), nn.GELU(), nn.LayerNorm(c.hidden_size), nn.Linear(c.hidden_size, c.feature_dim))
        self.post_init()

    def post_init(self) -> None:
        nn.init.normal_(self.mask_embedding, std=0.02)

    def encode(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None,
               mask_positions: torch.Tensor | None = None) -> torch.Tensor:
        if embeddings.ndim != 3: raise ValueError("embeddings must have shape [batch, length, feature_dim].")
        b, length, _ = embeddings.shape
        if length > self.config.max_position_embeddings: raise ValueError("Sequence exceeds max_position_embeddings.")
        hidden = self.input_projection(embeddings)
        if mask_positions is not None:
            hidden = hidden.clone(); hidden[mask_positions] = self.mask_embedding
        positions = torch.arange(length, device=embeddings.device).unsqueeze(0)
        hidden = self.dropout(self.input_norm(hidden + self.position_embeddings(positions)))
        padding = attention_mask == 0 if attention_mask is not None else None
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        return self.final_norm(hidden)

    def forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor,
                target_embeddings: torch.Tensor | None = None,
                mask_positions: torch.Tensor | None = None) -> ContinuousBertOutput:
        hidden = self.encode(embeddings, attention_mask, mask_positions)
        predictions = self.prediction_head(hidden)
        loss = None
        if target_embeddings is not None and mask_positions is not None and mask_positions.any():
            pred = predictions[mask_positions]; target = target_embeddings[mask_positions]
            smooth_l1 = F.smooth_l1_loss(pred, target)
            cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss = smooth_l1 + 0.1 * cosine
        return ContinuousBertOutput(loss=loss, predictions=predictions, last_hidden_state=hidden)
