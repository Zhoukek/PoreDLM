from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN, TextRotaryEmbeddingFast, TimestepEmbedder, init_linear


class FlowMatchingBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=True, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[TextRotaryEmbeddingFast] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_fn, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class FeatureProjector(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = init_linear(nn.Linear(hidden_size, hidden_size))
        self.norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.silu(self.proj(x)))


class FlowMatchingDenoiser(nn.Module):
    def __init__(self, text_encoder_dim: int, max_length: int, hidden_size: int = 1024, depth: int = 24, num_heads: int = 16, mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0, bottleneck_dim: int = 128, num_time_tokens: int = 4, num_self_cond_cfg_tokens: int = 4, num_model_mode_tokens: int = 0, vocab_size: int = 0):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size

        self.self_cond_proj = init_linear(nn.Linear(text_encoder_dim * 2, text_encoder_dim))
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.randn(1, num_time_tokens, hidden_size) * 0.02)
        self.self_cond_cfg_tokens = nn.Parameter(torch.randn(1, num_self_cond_cfg_tokens, hidden_size) * 0.02)
        self.mode_tokens = nn.Parameter(torch.randn(1, num_model_mode_tokens, hidden_size) * 0.02)

        q1, q3 = depth // 4, depth // 4 * 3
        blocks = []
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            blocks.append(FlowMatchingBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop if in_drop_range else 0.0, proj_drop=proj_drop if in_drop_range else 0.0))
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = FinalLayer(hidden_size, 1, text_encoder_dim)
        self.proj = init_linear(nn.Linear(hidden_size, text_encoder_dim))
        self.unembed = init_linear(nn.Linear(text_encoder_dim, vocab_size))
        self.feature_projector = FeatureProjector(hidden_size)

    def build_context(self, t: torch.Tensor, self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        prefix_tokens = []
        batch = t.shape[0]
        if self.num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        prefix_t = t.mean(dim=1) if t.ndim == 2 else t
        time_emb = self.t_embedder(prefix_t)
        prefix_tokens.append(self.t_emb_tokens.expand(batch, -1, -1) + time_emb.unsqueeze(1))
        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(self.self_cond_cfg_tokens.expand(batch, -1, -1) + sc_emb.unsqueeze(1))
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[torch.Tensor | bool] = None,
        return_features: Optional[int] = None,
        return_raw_features: Optional[int] = None,
        use_per_token_time: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]] | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        head_dim = self.hidden_size // self.num_heads
        batch = x.shape[0]
        seq_len = x.shape[1]
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)
        x = self.text_proj(x)
        if t.ndim == 2:
            if t.shape != (batch, seq_len):
                raise ValueError(f"per-token timestep shape must be {(batch, seq_len)}, got {tuple(t.shape)}")
            token_time_emb = self.t_embedder(t.reshape(-1)).reshape(batch, seq_len, self.hidden_size)
            x = x + token_time_emb
        elif use_per_token_time:
            token_time_emb = self.t_embedder(t).unsqueeze(1)
            x = x + token_time_emb

        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(batch, -1, -1)
            if decoder_step_active is None:
                active_gate = torch.tensor(False, device=x.device)
            else:
                active_gate = decoder_step_active if torch.is_tensor(decoder_step_active) else torch.tensor(decoder_step_active, device=x.device)
            mode_tokens = mode_tokens * active_gate.to(dtype=mode_tokens.dtype)
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones((batch, self.num_model_mode_tokens), dtype=attention_mask.dtype, device=x.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale=self_cond_cfg_scale)
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones((batch, prefix_len), dtype=attention_mask.dtype, device=x.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        feat_rope = TextRotaryEmbeddingFast(dim=head_dim, pt_seq_len=self.max_length, num_empty_token=prefix_len + model_mode_offset)
        requested_layer = return_features if return_features is not None else return_raw_features
        captured_features = None
        for layer_idx, block in enumerate(self.blocks, start=1):
            x = block(x, rope_fn=feat_rope, attention_mask=attention_mask)
            if requested_layer is not None and layer_idx == requested_layer:
                token_features = x[:, prefix_len + model_mode_offset :]
                if return_features is not None:
                    captured_features = self.feature_projector(token_features)
                else:
                    captured_features = token_features
        x = x[:, prefix_len + model_mode_offset :]

        decoder_logits = None
        if decoder_step_active is not None:
            active = bool(decoder_step_active.detach().to(dtype=torch.bool).item()) if torch.is_tensor(decoder_step_active) else bool(decoder_step_active)
            if active:
                decoder_logits = self.unembed(F.gelu(self.proj(x)))
            else:
                decoder_logits = torch.zeros((*x.shape[:2], self.vocab_size), dtype=x.dtype, device=x.device)

        output = self.final_layer(x)
        if requested_layer is not None:
            return output, decoder_logits, captured_features
        return output, decoder_logits


def flow_matching_b(**kwargs) -> FlowMatchingDenoiser:
    return FlowMatchingDenoiser(depth=12, hidden_size=768, num_heads=12, **kwargs)


def flow_matching_m(**kwargs) -> FlowMatchingDenoiser:
    return FlowMatchingDenoiser(depth=24, hidden_size=1056, num_heads=16, **kwargs)


def flow_matching_l(**kwargs) -> FlowMatchingDenoiser:
    return FlowMatchingDenoiser(depth=32, hidden_size=1280, num_heads=16, **kwargs)


FLOW_MATCHING_DENOISERS = {
    "flow_matching_b": flow_matching_b,
    "flow_matching_m": flow_matching_m,
    "flow_matching_l": flow_matching_l,
}
