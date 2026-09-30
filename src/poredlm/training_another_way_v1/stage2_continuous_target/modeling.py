"""Continuous-input masked contextual model with VQ embedding targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ContinuousTargetOutput:
    loss: Optional[torch.Tensor]
    smooth_l1_loss: Optional[torch.Tensor]
    cosine_loss: Optional[torch.Tensor]
    predictions: torch.Tensor
    last_hidden_state: torch.Tensor


class ContinuousTargetConfig:
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 1536,
        target_cosine_weight: float = 0.1,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.dropout = float(dropout)
        self.max_position_embeddings = int(max_position_embeddings)
        self.target_cosine_weight = float(target_cosine_weight)


class ContinuousTargetBert(nn.Module):
    """BERT-style encoder that predicts quantized target embeddings.

    The input remains continuous.  The target is supplied by the frozen Stage
    1 VQ codec and is detached by the training script, so this module cannot
    move the Stage 1 target space during the first experiment.
    """

    def __init__(self, config: ContinuousTargetConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = (
            nn.Linear(config.feature_dim, config.hidden_size)
            if config.feature_dim != config.hidden_size
            else nn.Identity()
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.feature_dim),
        )
        nn.init.normal_(self.mask_embedding, std=0.02)

    def encode(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, length, feature_dim].")
        batch, length, _ = embeddings.shape
        if length > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {length} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}."
            )
        hidden = self.input_projection(embeddings)
        if mask_positions is not None:
            hidden = hidden.clone()
            hidden[mask_positions] = self.mask_embedding
        positions = torch.arange(length, device=embeddings.device).unsqueeze(0)
        hidden = self.dropout(self.input_norm(hidden + self.position_embeddings(positions)))
        padding = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=padding)
        return self.final_norm(hidden)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        target_embeddings: torch.Tensor | None = None,
        mask_positions: torch.Tensor | None = None,
    ) -> ContinuousTargetOutput:
        hidden = self.encode(embeddings, attention_mask, mask_positions)
        predictions = self.prediction_head(hidden)
        loss = None
        smooth_l1 = None
        cosine = None
        if target_embeddings is not None and mask_positions is not None and mask_positions.any():
            pred = predictions[mask_positions]
            target = target_embeddings[mask_positions].detach()
            smooth_l1 = F.smooth_l1_loss(pred, target)
            cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            loss = smooth_l1 + self.config.target_cosine_weight * cosine
        return ContinuousTargetOutput(
            loss=loss,
            smooth_l1_loss=smooth_l1,
            cosine_loss=cosine,
            predictions=predictions,
            last_hidden_state=hidden,
        )

