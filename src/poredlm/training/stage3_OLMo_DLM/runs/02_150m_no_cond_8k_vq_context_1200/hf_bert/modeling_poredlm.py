from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from types import SimpleNamespace
from transformers import BertConfig, BertModel, PretrainedConfig, PreTrainedModel


class MaskedSignalContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d_model)
        self.num_attention_heads = heads
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        if attention_mask.dim() == 3:
            attn_mask = ~attention_mask.to(device=input_ids.device, dtype=torch.bool)
            attn_mask = attn_mask.repeat_interleave(self.num_attention_heads, dim=0)
            x = self.encoder(x, mask=attn_mask)
        else:
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.view(batch, -1)
            x = self.encoder(x, src_key_padding_mask=~attention_mask.to(device=input_ids.device).bool())
        hidden = self.norm(x)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden)
        return (hidden,)


class PoreDLMConfig(PretrainedConfig):
    model_type = "poredlm_dlm"

    def __init__(
        self,
        context_encoder_type: str = "bert",
        context_encoder_config: Optional[dict[str, Any]] = None,
        dlm_config: Optional[dict[str, Any]] = None,
        model_config: Optional[dict[str, Any]] = None,
        elf_src_path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.context_encoder_type = context_encoder_type
        self.context_encoder_config = context_encoder_config or {}
        self.dlm_config = dlm_config or {}
        self.model_config = model_config or {}
        self.elf_src_path = elf_src_path


class PoreDLMForDiffusion(PreTrainedModel):
    config_class = PoreDLMConfig
    base_model_prefix = "poredlm"
    main_input_name = "input_ids"

    def __init__(self, config: PoreDLMConfig):
        super().__init__(config)
        if config.elf_src_path and Path(config.elf_src_path).is_dir() and config.elf_src_path not in sys.path:
            sys.path.insert(0, config.elf_src_path)
        try:
            from torch_elf.model import ELF_models
        except ImportError as exc:
            raise ImportError(
                "Cannot import torch_elf. Add stage3_OLMo_DLM/ELF-pytorch-port/src to PYTHONPATH "
                "before loading this HF model."
            ) from exc

        if config.context_encoder_type == "masked_signal":
            self.context_encoder = MaskedSignalContextEncoder(**config.context_encoder_config)
        elif config.context_encoder_type == "bert":
            self.context_encoder = BertModel(
                BertConfig.from_dict(config.context_encoder_config),
                add_pooling_layer=False,
            )
        else:
            raise ValueError(f"Unsupported context_encoder_type={config.context_encoder_type!r}")
        self.context_hidden_size = self.context_encoder.config.hidden_size

        dlm = config.dlm_config
        model_cfg = config.model_config
        model_name = dlm.get("model", "ELF-B")
        if model_name not in ELF_models:
            raise ValueError(f"Unknown ELF model {model_name!r}; expected one of {sorted(ELF_models.keys())}")
        self.elf_denoiser = ELF_models[model_name](
            text_encoder_dim=self.context_hidden_size,
            max_length=int(dlm.get("max_length") or model_cfg.get("max_sequence_length") or 1024),
            attn_drop=float(dlm.get("attn_dropout", 0.0)),
            proj_drop=float(dlm.get("proj_dropout", 0.0)),
            num_time_tokens=int(dlm.get("num_time_tokens", 4)),
            num_self_cond_cfg_tokens=int(dlm.get("num_self_cond_cfg_tokens", 4)),
            vocab_size=int(model_cfg.get("vocab_size", 50257)),
            num_model_mode_tokens=int(dlm.get("num_model_mode_tokens", 0)),
            bottleneck_dim=int(dlm.get("bottleneck_dim", 128)),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cond_seq_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[torch.Tensor] = None,
        return_context: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if encoder_attention_mask is None:
            encoder_attention_mask = attention_mask
        if encoder_attention_mask is None:
            encoder_attention_mask = input_ids.new_ones(input_ids.shape)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        context_dtype = next(self.elf_denoiser.parameters()).dtype
        context = self.context_encoder(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state.to(dtype=context_dtype)

        if cond_seq_mask is not None:
            cond_seq_mask = cond_seq_mask.to(device=context.device, dtype=context.dtype).unsqueeze(-1)
        if t is None:
            t = torch.ones(input_ids.shape[0], device=context.device, dtype=context.dtype)
        if self_cond is not None:
            model_input = torch.cat([context, self_cond.to(context)], dim=-1)
        else:
            model_input = context

        pred, decoder_logits = self.elf_denoiser(
            model_input,
            t.to(device=context.device, dtype=context.dtype),
            attention_mask=attention_mask,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,
        )
        if cond_seq_mask is not None:
            pred = cond_seq_mask * context + (1.0 - cond_seq_mask) * pred
        output = {"last_hidden_state": pred}
        if decoder_logits is not None:
            output["logits"] = decoder_logits
        if return_context:
            output["context_hidden_state"] = context
        return output


__all__ = ["PoreDLMConfig", "PoreDLMForDiffusion"]
