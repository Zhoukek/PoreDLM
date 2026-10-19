# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .signal_tokenizer import BwavTokenizer

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


class CodebookFeatureFusion(nn.Module):
    """Fuse sequence hidden states with token-id codebook/embedding features."""

    def __init__(
        self,
        hidden_size: int,
        codebook_dim: int,
        mode: str,
        dropout: float = 0.1,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.code_proj = (
            nn.Identity()
            if int(codebook_dim) == int(hidden_size)
            else nn.Linear(int(codebook_dim), int(hidden_size))
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_size))

        if self.mode == "only":
            self.fuse = None
        elif self.mode == "add":
            self.fuse = None
        elif self.mode == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(int(hidden_size) * 2, int(hidden_size)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_size)),
            )
        elif self.mode == "gate":
            self.fuse = nn.Linear(int(hidden_size) * 2, int(hidden_size))
            nn.init.constant_(self.fuse.bias, float(gate_bias))
        else:
            raise ValueError(f"Unsupported codebook fusion mode: {self.mode!r}.")

    def forward(self, hidden: torch.Tensor | None, code_features: torch.Tensor) -> torch.Tensor:
        code = self.dropout(self.code_proj(code_features))
        if self.mode == "only":
            return self.norm(code)
        if hidden is None:
            raise ValueError("hidden cannot be None unless codebook fusion mode is 'only'.")
        if self.mode == "add":
            return self.norm(hidden + code)
        if self.mode == "concat":
            return self.fuse(torch.cat([hidden, code], dim=-1))
        if self.mode == "gate":
            gate = torch.sigmoid(self.fuse(torch.cat([hidden, code], dim=-1)))
            return self.norm(hidden + gate * code)
        raise RuntimeError(f"Unexpected codebook fusion mode: {self.mode!r}.")


class ExternalCodebookLookup(nn.Module):
    """Lookup raw tokenizer codebook rows with a bwav token offset."""

    def __init__(self, codebook: torch.Tensor, token_offset: int, freeze: bool = True) -> None:
        super().__init__()
        if codebook.ndim != 2:
            raise ValueError(f"External codebook must have shape [K, D], got {tuple(codebook.shape)}.")
        self.embedding = nn.Embedding.from_pretrained(codebook.to(dtype=torch.float32), freeze=bool(freeze))
        self.token_offset = int(token_offset)
        self.embedding_dim = int(codebook.shape[1])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        code_ids = input_ids.to(dtype=torch.long) - self.token_offset
        valid = (code_ids >= 0) & (code_ids < self.embedding.num_embeddings)
        safe_ids = torch.where(valid, code_ids, torch.zeros_like(code_ids))
        features = self.embedding(safe_ids)
        return torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))


class BasecallModel(nn.Module):
    """Basecaller adapter for the Stage 3 PoreDLM HF wrapper."""

    def __init__(
        self,
        model_path: str,
        backbone_type: str = "auto",
        tokenizer_path: str | None = None,
        tokenizer_type: str = "bwav",
        tokenizer_token_offset: int = 128,
        num_classes: int = NUM_CLASSES,
        hidden_layer: int = -1,
        learnable_fuse_last_n_layers: int = 0,
        feature_source: str = "hidden",
        feature_l2_normalize: bool = False,
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
        codebook_fusion: str = "none",
        codebook_fusion_dropout: float = 0.1,
        codebook_fusion_gate_bias: float = -2.0,
        codebook_feature_path: str | None = None,
        codebook_feature_token_offset: int | None = None,
        codebook_feature_trainable: bool = False,
        skip_backbone_for_codebook_only: bool = False,
        backbone_chunk_size: int = 600,
        elf_ode_steps: int = 4,
        elf_ode_start_t: float = 0.85,
        elf_self_cond_cfg_scale: float = 1.0,
        elf_sde_steps: int = 4,
        elf_sde_start_t: float = 0.85,
        elf_sde_gamma: float = 0.1,
        elf_sde_self_cond_cfg_scale: float = 1.0,
        elf_sde_seed: int | None = None,
    ):
        del vq_device, vq_token_batch_size
        super().__init__()
        self.hidden_layer = hidden_layer
        self.backbone_type = str(backbone_type).lower()
        self.learnable_fuse_last_n_layers = max(0, int(learnable_fuse_last_n_layers))
        self.feature_source = feature_source
        self.feature_l2_normalize = bool(feature_l2_normalize)
        self.codebook_fusion = str(codebook_fusion).lower()
        self.codebook_fusion_dropout = float(codebook_fusion_dropout)
        self.codebook_fusion_gate_bias = float(codebook_fusion_gate_bias)
        self.codebook_feature_path = codebook_feature_path
        self.codebook_feature_token_offset = (
            int(tokenizer_token_offset)
            if codebook_feature_token_offset is None
            else int(codebook_feature_token_offset)
        )
        self.codebook_feature_trainable = bool(codebook_feature_trainable)
        self.skip_backbone_for_codebook_only = bool(skip_backbone_for_codebook_only)
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
        self.elf_sde_steps = max(1, int(elf_sde_steps))
        self.elf_sde_start_t = float(elf_sde_start_t)
        self.elf_sde_gamma = float(elf_sde_gamma)
        self.elf_sde_self_cond_cfg_scale = float(elf_sde_self_cond_cfg_scale)
        self.elf_sde_seed = elf_sde_seed
        self.tokenizer = None
        self.backbone = None
        self.vq_embedding = None
        self.layer_fuse_logits = (
            nn.Parameter(torch.zeros(self.learnable_fuse_last_n_layers))
            if self.learnable_fuse_last_n_layers > 0
            else None
        )

        if self.backbone_type not in {"auto", "dlm", "bert"}:
            raise ValueError("backbone_type must be one of: auto, dlm, bert.")
        allowed_feature_sources = {"hidden", "denoised_hidden", "context_hidden", "ode_hidden", "sde_hidden"}
        if self.feature_source not in allowed_feature_sources:
            raise ValueError(
                "model_dlm.BasecallModel supports feature_source in "
                f"{sorted(allowed_feature_sources)}."
            )
        allowed_codebook_fusions = {"none", "only", "add", "concat", "gate"}
        if self.codebook_fusion not in allowed_codebook_fusions:
            raise ValueError(f"codebook_fusion must be one of {sorted(allowed_codebook_fusions)}.")
        if self.skip_backbone_for_codebook_only and self.codebook_fusion != "only":
            raise ValueError("--skip_backbone_for_codebook_only requires --codebook_fusion only.")
        if not 0.0 < self.elf_ode_start_t <= 1.0:
            raise ValueError("--elf_ode_start_t must be in (0, 1].")
        if not 0.0 < self.elf_sde_start_t <= 1.0:
            raise ValueError("--elf_sde_start_t must be in (0, 1].")
        if self.elf_sde_gamma < 0.0:
            raise ValueError("--elf_sde_gamma must be >= 0.")

        if reset_backbone_weights:
            backbone_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            self.backbone = AutoModel.from_config(backbone_config, trust_remote_code=True)
        else:
            self.backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if self.backbone_type == "auto":
            config = self.backbone.config
            is_dlm = any(
                getattr(config, name, None) is not None
                for name in ("dlm_config", "context_encoder_config", "model_config")
            ) or any(hasattr(self.backbone, name) for name in ("context_encoder", "elf_denoiser"))
            self.backbone_type = "dlm" if is_dlm else "bert"
        print(f"[Backbone] detected type={self.backbone_type} class={type(self.backbone).__name__}")

        max_positions = getattr(self.backbone.config, "max_position_embeddings", None)
        if max_positions and (self.backbone_chunk_size == 0 or self.backbone_chunk_size > int(max_positions)):
            old_chunk_size = self.backbone_chunk_size
            self.backbone_chunk_size = int(max_positions)
            print(
                f"[Backbone] adjusted chunk_size={old_chunk_size} -> {self.backbone_chunk_size} "
                "to fit max_position_embeddings"
            )

        if self.backbone_type == "dlm":
            if self.learnable_fuse_last_n_layers > 0:
                raise ValueError("PoreDLM HF wrapper does not expose per-layer hidden_states; use hidden_layer=-1.")
            if self.hidden_layer not in {-1, 0}:
                raise ValueError("PoreDLM HF wrapper exposes one sequence feature. Use hidden_layer=-1 or 0.")
        elif self.feature_source != "hidden":
            raise ValueError(
                f"BERT backbone only supports feature_source='hidden', got {self.feature_source!r}."
            )

        if tokenizer_type == "auto":
            tokenizer_path = tokenizer_path or model_path
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        elif tokenizer_type == "bwav":
            context_cfg = getattr(self.backbone.config, "context_encoder_config", None) or {}
            def token_id(name: str, default: int) -> int:
                value = context_cfg.get(name, getattr(self.backbone.config, name, default))
                return default if value is None else int(value)

            vocab_size = int(context_cfg.get("vocab_size", getattr(self.backbone.config, "vocab_size", 65664)))
            self.tokenizer = BwavTokenizer(
                vocab_size=vocab_size,
                token_offset=tokenizer_token_offset,
                pad_token_id=token_id("pad_token_id", 1),
                bos_token_id=token_id("bos_token_id", 2),
                eos_token_id=token_id("eos_token_id", 3),
                mask_token_id=token_id("mask_token_id", 4),
            )
        else:
            raise ValueError(f"Unsupported tokenizer_type={tokenizer_type!r}.")

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

        self._set_frozen_backbone_submodules_eval()
        show_layer_trainable_status(self.backbone)

        hidden_size = self._infer_hidden_size()
        self.codebook_feature_embedding = (
            self._build_codebook_feature_embedding() if self.codebook_fusion != "none" else None
        )
        if self.codebook_feature_embedding is not None:
            codebook_dim = int(self.codebook_feature_embedding.embedding_dim)
        else:
            codebook_dim = hidden_size
        self.codebook_feature_fuser = self._build_codebook_feature_fuser(
            mode=self.codebook_fusion,
            hidden_size=hidden_size,
            codebook_dim=codebook_dim,
            dropout=self.codebook_fusion_dropout,
            gate_bias=self.codebook_fusion_gate_bias,
        )
        if self.codebook_fusion != "none":
            print(
                f"[CodebookFusion] mode={self.codebook_fusion} "
                f"codebook_dim={codebook_dim} hidden_size={hidden_size} "
                f"dropout={self.codebook_fusion_dropout} gate_bias={self.codebook_fusion_gate_bias} "
                f"skip_backbone_for_codebook_only={self.skip_backbone_for_codebook_only}"
            )
        self.head_type = head_type
        self.pre_head = self._build_pre_head(
            pre_head_type=pre_head_type,
            hidden_size=hidden_size,
            transformer_nhead=pre_head_transformer_nhead,
        )
        self.token_prediction_head = nn.Linear(self.pre_head.output_dim, len(self.tokenizer))

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

    def _build_codebook_feature_embedding(self) -> nn.Module:
        if self.codebook_feature_path:
            codebook = self._load_external_codebook(self.codebook_feature_path)
            print(
                f"[CodebookFusion] using external codebook path={self.codebook_feature_path} "
                f"shape={tuple(codebook.shape)} token_offset={self.codebook_feature_token_offset} "
                f"trainable={self.codebook_feature_trainable}"
            )
            return ExternalCodebookLookup(
                codebook=codebook,
                token_offset=self.codebook_feature_token_offset,
                freeze=not self.codebook_feature_trainable,
            )
        return self._find_token_embedding_module()

    @staticmethod
    def _load_external_codebook(path: str) -> torch.Tensor:
        import os
        import sys
        from pathlib import Path

        if os.path.isdir(path):
            direct_codebook = BasecallModel._try_load_codebook_from_checkpoint_dir(path)
            if direct_codebook is not None:
                return direct_codebook

            VQETokenizer = None
            try:
                from poregpt.tokenizers import VQETokenizer as PoregptVQETokenizer

                VQETokenizer = PoregptVQETokenizer
            except ModuleNotFoundError:
                poredlm_root = Path(__file__).resolve().parents[4]
                tokenizer_dir = poredlm_root / "data" / "stage2_BERT_Encoder"
                for candidate in (str(poredlm_root), str(tokenizer_dir)):
                    if candidate not in sys.path:
                        sys.path.insert(0, candidate)
                try:
                    from vqe_tokenizer import VQETokenizer as LocalVQETokenizer

                    VQETokenizer = LocalVQETokenizer
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "--codebook_feature_path points to a tokenizer checkpoint directory, "
                        "but neither `poregpt.tokenizers.VQETokenizer` nor the local "
                        "stage2_BERT_Encoder/vqe_tokenizer.py loader could be imported."
                    ) from exc

            vq_tokenizer = VQETokenizer(model_ckpt=path, device="cpu")
            codebook = vq_tokenizer._get_codebook_embed()
            return torch.as_tensor(codebook.detach().cpu() if hasattr(codebook, "detach") else codebook)

        if path.endswith(".npy"):
            import numpy as np

            return torch.as_tensor(np.load(path), dtype=torch.float32)

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype=torch.float32)
        if isinstance(obj, dict):
            for key in ("codebook", "codebook_embed", "embedding", "embeddings", "weight"):
                value = obj.get(key)
                if isinstance(value, torch.Tensor):
                    return value.to(dtype=torch.float32)
            state = obj.get("state_dict")
            if isinstance(state, dict):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 2 and "codebook" in key:
                        return value.to(dtype=torch.float32)
        raise ValueError(f"Cannot find a [K, D] codebook tensor in {path!r}.")

    @staticmethod
    def _try_load_codebook_from_checkpoint_dir(path: str) -> torch.Tensor | None:
        import os

        candidates = [
            os.path.join(path, "model.safetensors"),
            os.path.join(path, "pytorch_model.bin"),
        ]
        for filename in os.listdir(path):
            if filename in {"model.safetensors", "pytorch_model.bin"}:
                continue
            if filename.endswith((".safetensors", ".bin", ".pt", ".pth")) and "model" in filename:
                candidates.append(os.path.join(path, filename))

        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            if candidate.endswith(".safetensors"):
                from safetensors.torch import load_file

                state = load_file(candidate, device="cpu")
            else:
                state = torch.load(candidate, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    state = state.get("model_state_dict", state.get("state_dict", state))
            if not isinstance(state, dict):
                continue
            codebook = BasecallModel._extract_codebook_tensor_from_state_dict(state)
            if codebook is not None:
                print(
                    f"[CodebookFusion] loaded codebook tensor directly from {candidate} "
                    f"shape={tuple(codebook.shape)}"
                )
                return codebook
        return None

    @staticmethod
    def _extract_codebook_tensor_from_state_dict(state: dict) -> torch.Tensor | None:
        exact_suffixes = (
            "vq._codebook.embed",
            "quantizer.codebooks",
            "vq.codebooks",
            "codebooks",
        )
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim not in {2, 3}:
                continue
            if any(str(key).endswith(suffix) for suffix in exact_suffixes):
                return BasecallModel._normalize_codebook_tensor(value)

        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim not in {2, 3}:
                continue
            lowered = str(key).lower()
            if "codebook" in lowered and ("embed" in lowered or "codebooks" in lowered):
                return BasecallModel._normalize_codebook_tensor(value)
        return None

    @staticmethod
    def _normalize_codebook_tensor(value: torch.Tensor) -> torch.Tensor:
        codebook = value.detach().cpu().to(dtype=torch.float32)
        if codebook.ndim == 3:
            if codebook.shape[0] == 1:
                codebook = codebook[0]
            else:
                codebook = codebook.reshape(codebook.shape[0] * codebook.shape[1], codebook.shape[2])
        if codebook.ndim != 2:
            raise ValueError(f"Unexpected codebook shape after normalization: {tuple(codebook.shape)}.")
        return codebook.contiguous()

    def _find_token_embedding_module(self) -> nn.Embedding:
        """Return the token-id embedding table used by the context encoder."""
        context_encoder = getattr(self.backbone, "context_encoder", None)
        if context_encoder is None:
            raise ValueError("--codebook_fusion requires a DLM backbone with context_encoder.")

        candidates = [
            ("context_encoder.token_embedding", context_encoder),
            ("context_encoder.token_embeddings", context_encoder),
        ]
        hf_embeddings = getattr(context_encoder, "embeddings", None)
        if hf_embeddings is not None:
            candidates.append(("context_encoder.embeddings.word_embeddings", hf_embeddings))
        wrapped_context = getattr(context_encoder, "model", None)
        if wrapped_context is not None:
            candidates.extend(
                [
                    ("context_encoder.model.token_embeddings", wrapped_context),
                    ("context_encoder.model.embeddings.word_embeddings", getattr(wrapped_context, "embeddings", None)),
                ]
            )

        for name, owner in candidates:
            if owner is None:
                continue
            attr_name = name.rsplit(".", 1)[-1]
            embedding = getattr(owner, attr_name, None)
            if isinstance(embedding, nn.Embedding):
                print(f"[CodebookFusion] using token features from {name}")
                return embedding

        raise ValueError(
            "--codebook_fusion was enabled, but no token embedding table was found on the context encoder."
        )

    @staticmethod
    def _build_codebook_feature_fuser(
        mode: str,
        hidden_size: int,
        codebook_dim: int,
        dropout: float,
        gate_bias: float,
    ) -> nn.Module | None:
        if mode == "none":
            return None
        return CodebookFeatureFusion(
            hidden_size=hidden_size,
            codebook_dim=codebook_dim,
            mode=mode,
            dropout=dropout,
            gate_bias=gate_bias,
        )

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
                ("encoder", "layers"),
                ("bert", "encoder", "layer"),
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
                ("context_encoder", "model", "encoder", "layers"),
                ("context_encoder", "model", "encoder", "layer"),
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

        wrapped_context = getattr(context_encoder, "model", None)
        if wrapped_context is not None:
            for attr in ("token_embeddings", "position_embeddings"):
                embedding = getattr(wrapped_context, attr, None)
                if embedding is not None:
                    embeddings.append((f"model.{attr}", embedding))

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
        return_token_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.skip_backbone_for_codebook_only:
            hidden = self._fuse_codebook_features(
                hidden=None,
                input_ids=input_ids,
            )
            hidden = self.pre_head(hidden)
            logits_btc = self.base_head(hidden)
            if return_token_logits:
                token_logits = self.token_prediction_head(hidden)
                return logits_btc, token_logits
            return logits_btc

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

        if self.feature_l2_normalize:
            hidden = F.normalize(hidden, p=2, dim=-1)
        hidden = self._fuse_codebook_features(hidden, input_ids)
        hidden = self.pre_head(hidden)
        logits_btc = self.base_head(hidden)
        if return_token_logits:
            token_logits = self.token_prediction_head(hidden)
            return logits_btc, token_logits
        return logits_btc

    def _lookup_codebook_features(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.codebook_feature_embedding is None:
            raise ValueError("codebook_feature_embedding is not initialized.")
        code_features = self.codebook_feature_embedding(input_ids)
        return code_features.to(dtype=dtype)

    def _fuse_codebook_features(self, hidden: torch.Tensor | None, input_ids: torch.Tensor) -> torch.Tensor:
        if self.codebook_feature_fuser is None:
            if hidden is None:
                raise ValueError("hidden cannot be None when codebook fusion is disabled.")
            return hidden
        dtype = hidden.dtype if hidden is not None else next(self.codebook_feature_fuser.parameters()).dtype
        code_features = self._lookup_codebook_features(input_ids, dtype=dtype)
        if hidden is not None and code_features.shape[:2] != hidden.shape[:2]:
            raise ValueError(
                "Codebook feature shape does not match hidden sequence shape: "
                f"codebook={tuple(code_features.shape)} hidden={tuple(hidden.shape)}."
            )
        return self.codebook_feature_fuser(hidden, code_features)

    def _forward_backbone_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backbone_type == "bert":
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise ValueError("BERT backbone did not return hidden_states.")
            if self.learnable_fuse_last_n_layers > 0:
                n = self.learnable_fuse_last_n_layers
                if n > len(hidden_states) - 1:
                    raise ValueError(f"Cannot fuse last {n} layers; BERT only has {len(hidden_states) - 1} layers.")
                selected = torch.stack(hidden_states[-n:], dim=0)
                weights = torch.softmax(self.layer_fuse_logits, dim=0).to(selected.dtype)
                return torch.sum(selected * weights.view(-1, 1, 1, 1), dim=0)
            try:
                return hidden_states[self.hidden_layer]
            except IndexError as exc:
                raise ValueError(
                    f"hidden_layer={self.hidden_layer} out of range for {len(hidden_states)} BERT hidden states."
                ) from exc

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_context=(self.feature_source == "context_hidden"),
            return_ode_hidden=(self.feature_source == "ode_hidden"),
            ode_steps=self.elf_ode_steps,
            ode_start_t=self.elf_ode_start_t,
            ode_self_cond_cfg_scale=self.elf_self_cond_cfg_scale,
            return_sde_hidden=(self.feature_source == "sde_hidden"),
            sde_steps=self.elf_sde_steps,
            sde_start_t=self.elf_sde_start_t,
            sde_gamma=self.elf_sde_gamma,
            sde_self_cond_cfg_scale=self.elf_sde_self_cond_cfg_scale,
            sde_seed=self.elf_sde_seed,
        )
        output_key = {
            "context_hidden": "context_hidden_state",
            "ode_hidden": "ode_hidden_state",
            "sde_hidden": "sde_hidden_state",
        }.get(self.feature_source, "last_hidden_state")
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
