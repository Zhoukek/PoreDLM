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
        unfreeze_target: str = "auto",
        unfreeze_context_last_n_layers: int = 0,
        unfreeze_elf_last_n_layers: int = 0,
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
        elf_ode_steps: int = 4,
        elf_ode_start_t: float = 0.85,
        elf_self_cond_cfg_scale: float = 1.0,
    ):
        del vq_device, vq_token_batch_size
        super().__init__()
        self.hidden_layer = hidden_layer
        self.learnable_fuse_last_n_layers = max(0, int(learnable_fuse_last_n_layers))
        self.feature_source = feature_source
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self.unfreeze_target = str(unfreeze_target)
        self.unfreeze_context_last_n_layers = max(0, int(unfreeze_context_last_n_layers))
        self.unfreeze_elf_last_n_layers = max(0, int(unfreeze_elf_last_n_layers))
        self.unfreeze_layer_start = unfreeze_layer_start
        self.unfreeze_layer_end = unfreeze_layer_end
        self.backbone_chunk_size = max(0, int(backbone_chunk_size))
        self.elf_ode_steps = max(1, int(elf_ode_steps))
        self.elf_ode_start_t = float(elf_ode_start_t)
        self.elf_self_cond_cfg_scale = float(elf_self_cond_cfg_scale)
        self.tokenizer = None
        self.backbone = None
        self.vq_embedding = None

        allowed_feature_sources = {"hidden", "denoised_hidden", "context_hidden", "ode_hidden"}
        if self.feature_source not in allowed_feature_sources:
            raise ValueError(
                "model_dlm.BasecallModel supports feature_source in "
                f"{sorted(allowed_feature_sources)}."
            )
        if self.learnable_fuse_last_n_layers > 0:
            raise ValueError("PoreDLM HF wrapper does not expose per-layer hidden_states; use hidden_layer=-1.")
        if self.hidden_layer not in {-1, 0}:
            raise ValueError("PoreDLM HF wrapper exposes one sequence feature. Use hidden_layer=-1 or 0.")
        if not 0.0 < self.elf_ode_start_t <= 1.0:
            raise ValueError("--elf_ode_start_t must be in (0, 1].")

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
            or self.unfreeze_context_last_n_layers > 0
            or self.unfreeze_elf_last_n_layers > 0
            or unfreeze_layer_start is not None
            or unfreeze_layer_end is not None
        ):
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.freeze_backbone and not self._has_partial_backbone_unfreeze():
                self.backbone.eval()

        if self.unfreeze_context_last_n_layers > 0:
            self._unfreeze_last_layers("context_encoder", self.unfreeze_context_last_n_layers)
        if self.unfreeze_elf_last_n_layers > 0:
            self._unfreeze_last_layers("elf_denoiser", self.unfreeze_elf_last_n_layers)

        if self.unfreeze_last_n_layers > 0 or unfreeze_layer_start is not None or unfreeze_layer_end is not None:
            layers = self._get_transformer_layers(self.unfreeze_target)
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
            if target_layers:
                self._unfreeze_context_embeddings()

        self._set_frozen_backbone_submodules_eval()
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

    def _get_transformer_layers(self, target: str = "auto") -> nn.ModuleList:
        target = str(target)
        candidate_groups = {
            "auto": (
                ("elf_denoiser", "blocks"),
                ("context_encoder", "encoder", "layers"),
                ("context_encoder", "encoder", "layer"),
                ("encoder", "layer"),
                ("model", "layers"),
                ("layers",),
                ("blocks",),
            ),
            "elf_denoiser": (
                ("elf_denoiser", "blocks"),
            ),
            "context_encoder": (
                ("context_encoder", "encoder", "layers"),
                ("context_encoder", "encoder", "layer"),
            ),
        }
        if target not in candidate_groups:
            raise ValueError(
                f"Unsupported unfreeze_target={target!r}; choose from {sorted(candidate_groups)}."
            )
        candidates = candidate_groups[target]
        for path in candidates:
            obj = self.backbone
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if obj is not None and isinstance(obj, (nn.ModuleList, list, tuple)):
                print(f"[PoreDLM] partial unfreeze target={target} layers from: {'.'.join(path)}")
                return nn.ModuleList(list(obj))
        raise ValueError(f"Cannot locate PoreDLM transformer layers for partial unfreezing target={target!r}.")

    def _unfreeze_last_layers(self, target: str, n_layers: int) -> None:
        layers = self._get_transformer_layers(target)
        n_unfreeze = min(max(0, int(n_layers)), len(layers))
        if n_unfreeze <= 0:
            return
        print(f"[PoreDLM] unfreeze last {n_unfreeze}/{len(layers)} layers for target={target}")
        for layer in layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        if target == "context_encoder":
            self._unfreeze_context_embeddings()

    def _unfreeze_context_embeddings(self) -> None:
        context_encoder = getattr(self.backbone, "context_encoder", None)
        if context_encoder is None:
            raise ValueError("Cannot unfreeze context embeddings: backbone has no context_encoder.")

        embeddings = []
        if hasattr(context_encoder, "token_embedding"):
            embeddings.append(("token_embedding", context_encoder.token_embedding))
        if hasattr(context_encoder, "position_embedding"):
            embeddings.append(("position_embedding", context_encoder.position_embedding))

        hf_embeddings = getattr(context_encoder, "embeddings", None)
        if hf_embeddings is not None:
            if hasattr(hf_embeddings, "word_embeddings"):
                embeddings.append(("embeddings.word_embeddings", hf_embeddings.word_embeddings))
            if hasattr(hf_embeddings, "position_embeddings"):
                embeddings.append(("embeddings.position_embeddings", hf_embeddings.position_embeddings))

        if not embeddings:
            raise ValueError("Cannot unfreeze context embeddings: no known embedding modules found.")

        for _, embedding in embeddings:
            for param in embedding.parameters():
                param.requires_grad = True
        names = ", ".join(name for name, _ in embeddings)
        print(f"[PoreDLM] unfreeze context_encoder embeddings: {names}")

    def _set_frozen_backbone_submodules_eval(self) -> None:
        if self.backbone is None:
            return
        for name in ("context_encoder", "elf_denoiser"):
            module = getattr(self.backbone, name, None)
            if module is None:
                continue
            if not any(param.requires_grad for param in module.parameters()):
                module.eval()

    def _has_partial_backbone_unfreeze(self) -> bool:
        return (
            self.unfreeze_last_n_layers > 0
            or self.unfreeze_context_last_n_layers > 0
            or self.unfreeze_elf_last_n_layers > 0
            or self.unfreeze_layer_start is not None
            or self.unfreeze_layer_end is not None
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            self.freeze_backbone
            and not self._has_partial_backbone_unfreeze()
            and self.backbone is not None
        ):
            self.backbone.eval()
        elif mode:
            self._set_frozen_backbone_submodules_eval()
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
        if self.feature_source in {"context_hidden", "ode_hidden"} and hasattr(self.backbone, "context_encoder"):
            outputs = self.backbone.context_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, dict):
                hidden = outputs.get("last_hidden_state")
            if hidden is None:
                raise ValueError("PoreDLM context_encoder output does not contain last_hidden_state.")
            if self.feature_source == "ode_hidden":
                return self._ode_from_context_hidden(hidden, attention_mask=attention_mask)
            return hidden

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_context=(self.feature_source == "context_hidden"),
            return_dict=True,
        )
        output_key = "context_hidden_state" if self.feature_source == "context_hidden" else "last_hidden_state"
        if isinstance(outputs, dict):
            hidden = outputs.get(output_key)
        else:
            hidden = getattr(outputs, output_key, None)
        if hidden is None:
            raise ValueError(f"PoreDLM backbone output does not contain {output_key}.")
        return hidden

    def _get_dlm_config_value(self, key: str, default):
        dlm_cfg = getattr(self.backbone.config, "dlm_config", None) or {}
        return dlm_cfg.get(key, default)

    def _elf_t_eps(self) -> float:
        return float(self._get_dlm_config_value("t_eps", 0.05))

    @staticmethod
    def _elf_net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor, t_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(net_out, tuple):
            net_out = net_out[0]
        t_reshaped = t.view(-1, 1, 1)
        x_pred = net_out
        v_pred = (x_pred - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
        return v_pred, x_pred

    def _elf_forward_ode_sample(
        self,
        z: torch.Tensor,
        t_batch: torch.Tensor,
        x_pred_prev: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        elf = getattr(self.backbone, "elf_denoiser", None)
        if elf is None:
            raise ValueError("feature_source='ode_hidden' requires backbone.elf_denoiser.")

        num_self_cond_cfg_tokens = int(getattr(elf, "num_self_cond_cfg_tokens", 0))
        if num_self_cond_cfg_tokens > 0:
            if x_pred_prev is None:
                x_pred_prev = torch.zeros_like(z)
            model_input = torch.cat([z, x_pred_prev], dim=-1)
            self_cond_cfg_scale = torch.full(
                (z.shape[0],),
                self.elf_self_cond_cfg_scale,
                device=z.device,
                dtype=z.dtype,
            )
            net_out = elf(
                model_input,
                t_batch,
                attention_mask=attention_mask,
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=False,
            )
        else:
            net_out = elf(
                z,
                t_batch,
                attention_mask=attention_mask,
                decoder_step_active=False,
            )
        return self._elf_net_out_to_v_x(net_out, z, t_batch, self._elf_t_eps())

    def _ode_from_context_hidden(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_dtype = next(self.backbone.elf_denoiser.parameters()).dtype
        context = context.to(dtype=context_dtype)
        z = context
        x_pred = torch.zeros_like(z)
        t_steps = torch.linspace(
            self.elf_ode_start_t,
            1.0,
            self.elf_ode_steps + 1,
            device=context.device,
            dtype=context.dtype,
        )

        for idx in range(self.elf_ode_steps):
            t = t_steps[idx]
            t_next = t_steps[idx + 1]
            t_batch = torch.full((z.shape[0],), float(t.detach().item()), device=z.device, dtype=z.dtype)
            v_pred, x_pred = self._elf_forward_ode_sample(
                z,
                t_batch,
                x_pred,
                attention_mask=attention_mask,
            )
            z = z + (t_next - t) * v_pred
            if attention_mask is not None:
                valid_mask = attention_mask.to(device=context.device, dtype=torch.bool).unsqueeze(-1)
                z = torch.where(valid_mask, z, context)
                x_pred = torch.where(valid_mask, x_pred, context)

        return z
