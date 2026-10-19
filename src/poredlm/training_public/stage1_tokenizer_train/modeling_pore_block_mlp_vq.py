"""Strictly block-local MLP VQ tokenizer.

The encoder reshapes [B, 1, 6000] to [B, 1200, 5] and applies the same MLP
independently to every block. There is no convolution, padding, pooling, or
cross-block operation, so token i only depends on signal[5*i:5*i+5].
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from vector_quantize_pytorch import VectorQuantize

MODEL_TYPE = "pore_block_mlp_vq"


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    use_layer_norm: bool = True,
) -> nn.Sequential:
    dims = [int(input_dim), *(int(x) for x in hidden_dims), int(output_dim)]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if out_dim != dims[-1]:
            layers.append(nn.SiLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class BlockMLPEncoder(nn.Module):
    """Shared MLP applied independently to every five-sample block."""

    def __init__(
        self,
        block_size=5,
        hidden_dims=(128, 256, 512),
        latent_dim=768,
        use_layer_norm=True,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.mlp = _make_mlp(
            self.block_size, hidden_dims, latent_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim == 3:
            if signal.shape[1] != 1:
                raise ValueError(f"Expected one signal channel, got {tuple(signal.shape)}")
            signal = signal[:, 0, :]
        if signal.ndim != 2:
            raise ValueError(f"Expected [B, 1, T] or [B, T], got {tuple(signal.shape)}")
        batch, length = signal.shape
        if length % self.block_size != 0:
            raise ValueError(f"Signal length {length} is not divisible by block_size={self.block_size}.")
        blocks = signal.reshape(batch, length // self.block_size, self.block_size)
        return self.mlp(blocks)


class BlockMLPDecoder(nn.Module):
    """Shared MLP applied independently to every quantized token."""

    def __init__(
        self,
        block_size=5,
        hidden_dims=(512, 256, 128),
        latent_dim=768,
        use_layer_norm=True,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.mlp = _make_mlp(
            latent_dim, hidden_dims, block_size,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        blocks = self.mlp(latent)
        return blocks.reshape(blocks.shape[0], 1, -1)


class BlockMLPVQCodecConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self, codebook_size=8192, codebook_decay=0.99, codebook_emadc=2,
        commitment_weight=1.0, orthogonal_reg_weight=1.0,
        codebook_diversity_loss_weight=1.0, block_size=5, chunk_size=6000,
        latent_dim=768, encoder_hidden_dims=(128, 256, 512),
        decoder_hidden_dims=(512, 256, 128), learnable_codebook=True,
        use_layer_norm=True, **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = int(codebook_size)
        self.codebook_decay = float(codebook_decay)
        self.codebook_emadc = int(codebook_emadc)
        self.commitment_weight = float(commitment_weight)
        self.orthogonal_reg_weight = float(orthogonal_reg_weight)
        self.codebook_diversity_loss_weight = float(codebook_diversity_loss_weight)
        self.block_size = int(block_size)
        self.chunk_size = int(chunk_size)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = [int(x) for x in encoder_hidden_dims]
        self.decoder_hidden_dims = [int(x) for x in decoder_hidden_dims]
        self.learnable_codebook = bool(learnable_codebook)
        self.use_layer_norm = bool(use_layer_norm)


class BlockMLPVQCodec(PreTrainedModel):
    config_class = BlockMLPVQCodecConfig
    _no_split_modules = ["BlockMLPEncoder", "BlockMLPDecoder", "VectorQuantize"]

    def __init__(self, config: BlockMLPVQCodecConfig):
        self.all_tied_weights_keys = {}
        self._tied_weights_keys = set()
        super().__init__(config)
        if config.chunk_size % config.block_size != 0:
            raise ValueError("chunk_size must be divisible by block_size")
        self.block_size = config.block_size
        self.chunk_size = config.chunk_size
        self.tokens_per_chunk = config.chunk_size // config.block_size
        self.latent_dim = config.latent_dim
        self.codebook_size = config.codebook_size
        self.encoder = BlockMLPEncoder(
            config.block_size, config.encoder_hidden_dims, config.latent_dim,
            use_layer_norm=config.use_layer_norm,
        )
        self.decoder = BlockMLPDecoder(
            config.block_size, config.decoder_hidden_dims, config.latent_dim,
            use_layer_norm=config.use_layer_norm,
        )
        self.vq = VectorQuantize(
            dim=config.latent_dim, codebook_size=config.codebook_size,
            kmeans_init=True, kmeans_iters=10, decay=config.codebook_decay,
            threshold_ema_dead_code=config.codebook_emadc,
            commitment_weight=config.commitment_weight,
            codebook_diversity_loss_weight=config.codebook_diversity_loss_weight,
            orthogonal_reg_weight=config.orthogonal_reg_weight,
            orthogonal_reg_max_codes=256, orthogonal_reg_active_codes_only=True,
            learnable_codebook=config.learnable_codebook,
            ema_update=not config.learnable_codebook,
        )

    @property
    def cnn_stride(self):
        return self.block_size

    @property
    def RF(self):
        return self.block_size

    def forward(self, signal: torch.Tensor):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError(f"Expected signal shape [B, 1, T], got {tuple(signal.shape)}")
        if signal.shape[-1] != self.chunk_size:
            raise ValueError(f"Expected chunk length {self.chunk_size}, got {signal.shape[-1]}")
        z = self.encoder(signal)
        z_quantized, indices, vq_loss, breakdown = self.vq(z, return_loss_breakdown=True)
        recon = self.decoder(z_quantized)
        distill_loss = torch.zeros((), device=signal.device, dtype=signal.dtype)
        return recon, indices.to(dtype=torch.long, device=signal.device), vq_loss, breakdown, distill_loss

    @torch.no_grad()
    def encode_signal(self, signal: torch.Tensor):
        self.eval()
        return self.forward(signal)[1]

    @torch.no_grad()
    def decode_token(self, token_ids: torch.Tensor):
        self.eval()
        token_ids = token_ids.to(next(self.parameters()).device).long()
        return self.decoder(self.vq.get_output_from_indices(token_ids))


AutoConfig.register(MODEL_TYPE, BlockMLPVQCodecConfig)
AutoModel.register(BlockMLPVQCodecConfig, BlockMLPVQCodec)
