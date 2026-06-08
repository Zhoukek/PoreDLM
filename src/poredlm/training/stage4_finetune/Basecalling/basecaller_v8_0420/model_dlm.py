# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model import (
    BiLSTMPreHead,
    IdentityPreHead,
    LinearCRFEncoder,
    LinearCTCEncoder,
    TCNPreHead,
    TransformerPreHead,
    show_layer_trainable_status,
)
from .utils import ID2BASE, NUM_CLASSES


class BasecallModel(nn.Module):
    """Basecaller adapter for the Stage 3 PoreDLM HF wrapper."""

    def __init__(
        self,
        model_path: str,
        num_classes: int = NUM_CLASSES,
        hidden_layer: int = -1,
        learnable_fuse_last_n_layers: int = 0,
        feature_source: str = "hidden",
        vq_device: str = "cuda",
        vq_token_batch_size: int = 100,
        freeze_backbone: bool = False,
        reset_backbone_weights: bool = False,
        unfreeze_last_n_layers: int = 0,
        unfreeze_layer_start: int | None = None,
        unfreeze_layer_end: int | None = None,
        head_output_activation: str | None = None,
        head_output_scale: float | None = None,
        head_crf_blank_score: float | None = None,
        head_crf_n_base: int | None = None,
        head_crf_state_len: int | None = None,
        head_crf_expand_blanks: bool = True,
        pre_head_type: str = "none",
        pre_head_transformer_nhead: int = 8,
        head_type: str = "ctc_crf",
        backbone_chunk_size: int = 600,
    ):
        del vq_device, vq_token_batch_size
        super().__init__()
        self.hidden_layer = hidden_layer
        self.learnable_fuse_last_n_layers = max(0, int(learnable_fuse_last_n_layers))
        self.feature_source = feature_source
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self.unfreeze_layer_start = unfreeze_layer_start
        self.unfreeze_layer_end = unfreeze_layer_end
        self.backbone_chunk_size = max(0, int(backbone_chunk_size))
        self.tokenizer = None
        self.backbone = None
        self.vq_embedding = None

        if self.feature_source != "hidden":
            raise ValueError("model_dlm.BasecallModel only supports feature_source='hidden'.")
        if self.learnable_fuse_last_n_layers > 0:
            raise ValueError("PoreDLM HF wrapper does not expose per-layer hidden_states; use hidden_layer=-1.")
        if self.hidden_layer not in {-1, 0}:
            raise ValueError("PoreDLM HF wrapper exposes one sequence feature. Use hidden_layer=-1 or 0.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if reset_backbone_weights:
            backbone_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.backbone = AutoModel.from_config(backbone_config, trust_remote_code=True)
        else:
            self.backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        if (
            self.freeze_backbone
            or self.unfreeze_last_n_layers > 0
            or unfreeze_layer_start is not None
            or unfreeze_layer_end is not None
        ):
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.freeze_backbone and self.unfreeze_last_n_layers == 0:
                self.backbone.eval()

        if self.unfreeze_last_n_layers > 0 or unfreeze_layer_start is not None or unfreeze_layer_end is not None:
            layers = self._get_transformer_layers()
            n_layers = len(layers)
            if unfreeze_layer_start is not None or unfreeze_layer_end is not None:
                start = 0 if unfreeze_layer_start is None else int(unfreeze_layer_start)
                end = n_layers if unfreeze_layer_end is None else int(unfreeze_layer_end)
                if start < 0:
                    start = n_layers + start
                if end < 0:
                    end = n_layers + end
                if not 0 <= start <= end <= n_layers:
                    raise ValueError(f"Invalid unfreeze layer range: [{start}, {end}) with {n_layers} layers.")
                target_layers = layers[start:end]
            else:
                n_unfreeze = min(self.unfreeze_last_n_layers, n_layers)
                target_layers = layers[-n_unfreeze:]
            for layer in target_layers:
                for param in layer.parameters():
                    param.requires_grad = True

        show_layer_trainable_status(self.backbone)

        hidden_size = self._infer_hidden_size()
        self.head_type = head_type
        self.pre_head = self._build_pre_head(
            pre_head_type=pre_head_type,
            hidden_size=hidden_size,
            transformer_nhead=pre_head_transformer_nhead,
        )

        if self.head_type == "ctc_crf":
            n_base = head_crf_n_base if head_crf_n_base is not None else (len(ID2BASE) - 1)
            if head_crf_state_len is None:
                if n_base <= 1:
                    raise ValueError("Cannot infer head_crf_state_len with n_base <= 1.")
                base = num_classes / (n_base + 1)
                state_len = math.log(base, n_base) - 1
                if not math.isclose(state_len, round(state_len)):
                    raise ValueError("Unable to infer head_crf_state_len from num_classes and n_base.")
                head_crf_state_len = int(round(state_len))
            self.base_head = LinearCRFEncoder(
                insize=self.pre_head.output_dim,
                n_base=n_base,
                state_len=head_crf_state_len,
                bias=True,
                scale=head_output_scale,
                activation=head_output_activation,
                blank_score=head_crf_blank_score,
                expand_blanks=head_crf_expand_blanks,
            )
        elif self.head_type == "ctc":
            self.base_head = LinearCTCEncoder(
                insize=self.pre_head.output_dim,
                num_classes=num_classes,
                bias=True,
                scale=head_output_scale,
                activation=head_output_activation,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    def _infer_hidden_size(self) -> int:
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(self.backbone.config, attr, None)
            if value is not None:
                return int(value)
        context_cfg = getattr(self.backbone.config, "context_encoder_config", None) or {}
        if context_cfg.get("hidden_size") is not None:
            return int(context_cfg["hidden_size"])
        model_cfg = getattr(self.backbone.config, "model_config", None) or {}
        if model_cfg.get("d_model") is not None:
            return int(model_cfg["d_model"])
        if hasattr(self.backbone, "context_hidden_size"):
            return int(self.backbone.context_hidden_size)
        raise ValueError("Cannot infer hidden_size from PoreDLM backbone config.")

    @staticmethod
    def _build_pre_head(
        pre_head_type: str,
        hidden_size: int,
        transformer_nhead: int,
    ) -> nn.Module:
        if pre_head_type == "none":
            return IdentityPreHead(hidden_size)
        if pre_head_type == "bilstm":
            return BiLSTMPreHead(input_dim=hidden_size, hidden_dim=128)
        if pre_head_type == "transformer":
            if hidden_size % transformer_nhead != 0:
                raise ValueError(
                    f"hidden_size={hidden_size} must be divisible by transformer nhead={transformer_nhead}."
                )
            return TransformerPreHead(model_dim=hidden_size, nhead=transformer_nhead)
        if pre_head_type == "tcn":
            return TCNPreHead(model_dim=hidden_size)
        raise ValueError(f"Unsupported pre_head_type: {pre_head_type}")

    def _get_transformer_layers(self) -> nn.ModuleList:
        candidates = (
            ("elf_denoiser", "blocks"),
            ("context_encoder", "encoder", "layer"),
            ("encoder", "layer"),
            ("model", "layers"),
            ("layers",),
            ("blocks",),
        )
        for path in candidates:
            obj = self.backbone
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if obj is not None and isinstance(obj, (nn.ModuleList, list, tuple)):
                print(f"[PoreDLM] partial unfreeze layers from: {'.'.join(path)}")
                return nn.ModuleList(list(obj))
        raise ValueError("Cannot locate PoreDLM transformer layers for partial unfreezing.")

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            self.freeze_backbone
            and self.unfreeze_last_n_layers == 0
            and self.unfreeze_layer_start is None
            and self.unfreeze_layer_end is None
            and self.backbone is not None
        ):
            self.backbone.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backbone_chunk_size > 0 and input_ids.shape[1] > self.backbone_chunk_size:
            hidden_parts = []
            for start in range(0, input_ids.shape[1], self.backbone_chunk_size):
                end = min(start + self.backbone_chunk_size, input_ids.shape[1])
                chunk_attention_mask = (
                    attention_mask[:, start:end]
                    if attention_mask is not None
                    else None
                )
                hidden_parts.append(
                    self._forward_backbone_hidden(
                        input_ids=input_ids[:, start:end],
                        attention_mask=chunk_attention_mask,
                    )
                )
            hidden = torch.cat(hidden_parts, dim=1)
        else:
            hidden = self._forward_backbone_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = self.pre_head(hidden)
        logits_btc = self.base_head(hidden)
        return logits_btc

    def _forward_backbone_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
        else:
            hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("PoreDLM backbone output does not contain last_hidden_state.")
        return hidden
