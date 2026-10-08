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


class Stage2BertConfig(PretrainedConfig):
    model_type = "poredlm_stage2_bert"

    def __init__(
        self,
        vocab_size: int = 65664,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1280,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        cls_token_id: int = 4,
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.attention_probs_dropout_prob = float(attention_probs_dropout_prob)
        self.max_position_embeddings = int(max_position_embeddings)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.cls_token_id = int(cls_token_id)
        self.layer_norm_eps = float(layer_norm_eps)


class Stage2MaskedSignalLM(nn.Module):
    def __init__(self, config: Stage2BertConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}.")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}."
            )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden = self.embedding_layer_norm(hidden)
        hidden = self.dropout(hidden)
        if attention_mask is not None and attention_mask.dim() == 3:
            if attention_mask.shape != (input_ids.shape[0], seq_len, seq_len):
                raise ValueError(
                    "3D attention_mask must have shape [batch, seq_len, seq_len], "
                    f"got {tuple(attention_mask.shape)}."
                )
            # nn.TransformerEncoder uses True for disallowed query/key pairs
            # and expects a separate mask for every attention head.
            attn_mask = ~attention_mask.to(device=input_ids.device, dtype=torch.bool)
            attn_mask = attn_mask.repeat_interleave(self.config.num_attention_heads, dim=0)
            hidden = self.encoder(hidden, mask=attn_mask)
        else:
            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must have shape [batch, seq_len] or "
                    "[batch, seq_len, seq_len]."
                )
            key_padding_mask = attention_mask == 0 if attention_mask is not None else None
            hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.final_layer_norm(hidden)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden)
        return (hidden,)


class HFContextEncoderAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, dict):
            hidden = output.get("last_hidden_state")
        if hidden is None:
            raise ValueError(
                f"HF context encoder {self.model.__class__.__name__} did not return last_hidden_state."
            )
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
        model_config = model_config or {}
        kwargs.setdefault("pad_token_id", model_config.get("pad_token_id"))
        kwargs.setdefault("bos_token_id", model_config.get("bos_token_id"))
        kwargs.setdefault("eos_token_id", model_config.get("eos_token_id"))
        super().__init__(**kwargs)
        self.context_encoder_type = context_encoder_type
        self.context_encoder_config = context_encoder_config or {}
        self.dlm_config = dlm_config or {}
        self.model_config = model_config
        self.elf_src_path = elf_src_path


class PoreDLMForDiffusion(PreTrainedModel):
    config_class = PoreDLMConfig
    base_model_prefix = "poredlm"
    main_input_name = "input_ids"

    @staticmethod
    def _add_elf_src_to_path(config: PoreDLMConfig) -> None:
        candidate_paths = []
        model_dir = getattr(config, "name_or_path", None) or getattr(config, "_name_or_path", None)
        if model_dir:
            candidate_paths.append(Path(model_dir) / "ELF-pytorch-port" / "src")
        candidate_paths.append(Path(__file__).resolve().parent / "ELF-pytorch-port" / "src")
        if config.elf_src_path:
            candidate_paths.append(Path(config.elf_src_path))

        for path in candidate_paths:
            path_str = str(path)
            if path.is_dir() and path_str not in sys.path:
                sys.path.insert(0, path_str)

    def __init__(self, config: PoreDLMConfig):
        super().__init__(config)
        self._add_elf_src_to_path(config)
        try:
            from torch_elf.model import ELF_models
        except ImportError as exc:
            raise ImportError(
                "Cannot import torch_elf. Put ELF-pytorch-port under the HF model directory "
                "or add ELF-pytorch-port/src to PYTHONPATH before loading this HF model."
            ) from exc

        if config.context_encoder_type == "masked_signal":
            self.context_encoder = MaskedSignalContextEncoder(**config.context_encoder_config)
        elif config.context_encoder_type == "hf_auto":
            if config.context_encoder_config.get("model_type") == Stage2BertConfig.model_type:
                context_config = Stage2BertConfig(**config.context_encoder_config)
                self.context_encoder = HFContextEncoderAdapter(Stage2MaskedSignalLM(context_config))
            else:
                raise ValueError(
                    "context_encoder_type='hf_auto' currently supports only "
                    f"{Stage2BertConfig.model_type!r}; got "
                    f"{config.context_encoder_config.get('model_type')!r}."
                )
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

    def _elf_t_eps(self) -> float:
        return float(self.config.dlm_config.get("t_eps", 0.05))

    @staticmethod
    def _elf_net_out_to_v_x(
        net_out: Any,
        z: torch.Tensor,
        t: torch.Tensor,
        t_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        x_pred_prev: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        self_cond_cfg_scale_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_self_cond_cfg_tokens = int(getattr(self.elf_denoiser, "num_self_cond_cfg_tokens", 0))
        if num_self_cond_cfg_tokens > 0:
            if x_pred_prev is None:
                x_pred_prev = torch.zeros_like(z)
            model_input = torch.cat([z, x_pred_prev], dim=-1)
            self_cond_cfg_scale = torch.full(
                (z.shape[0],),
                float(self_cond_cfg_scale_value),
                device=z.device,
                dtype=z.dtype,
            )
            net_out = self.elf_denoiser(
                model_input,
                t_batch,
                attention_mask=attention_mask,
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=False,
            )
        else:
            net_out = self.elf_denoiser(
                z,
                t_batch,
                attention_mask=attention_mask,
                decoder_step_active=False,
            )
        return self._elf_net_out_to_v_x(net_out, z, t_batch, self._elf_t_eps())

    @staticmethod
    def _restore_condition(
        value: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(condition_mask.unsqueeze(-1), condition, value)

    def _conditional_sample_step(
        self,
        z: torch.Tensor,
        t_batch: torch.Tensor,
        x_pred_prev: Optional[torch.Tensor],
        condition: torch.Tensor,
        condition_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        cfg_scale: float,
        self_cond_cfg_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_cond, x_cond = self._elf_forward_ode_sample(
            z,
            t_batch,
            x_pred_prev,
            attention_mask=attention_mask,
            self_cond_cfg_scale_value=self_cond_cfg_scale,
        )
        v_cond = self._restore_condition(v_cond, torch.zeros_like(condition), condition_mask)
        x_cond = self._restore_condition(x_cond, condition, condition_mask)
        if cfg_scale == 1.0:
            return v_cond, x_cond

        # Classifier-free guidance for a sequence condition: remove the fixed
        # prefix from both the current latent and the previous self-condition.
        z_uncond = self._restore_condition(z, torch.zeros_like(condition), condition_mask)
        x_prev_uncond = None
        if x_pred_prev is not None:
            x_prev_uncond = self._restore_condition(
                x_pred_prev, torch.zeros_like(condition), condition_mask
            )
        v_uncond, x_uncond = self._elf_forward_ode_sample(
            z_uncond,
            t_batch,
            x_prev_uncond,
            attention_mask=attention_mask,
            self_cond_cfg_scale_value=self_cond_cfg_scale,
        )
        v = v_uncond + float(cfg_scale) * (v_cond - v_uncond)
        x = x_uncond + float(cfg_scale) * (x_cond - x_uncond)
        v = self._restore_condition(v, torch.zeros_like(condition), condition_mask)
        x = self._restore_condition(x, condition, condition_mask)
        return v, x

    @torch.no_grad()
    def generate(
        self,
        condition_input_ids: torch.LongTensor,
        condition_attention_mask: Optional[torch.Tensor] = None,
        condition_token_mask: Optional[torch.Tensor] = None,
        *,
        max_length: Optional[int] = None,
        num_steps: int = 50,
        sampling_method: str = "ode",
        sde_gamma: float = 0.1,
        cfg_scale: float = 1.0,
        self_cond_cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        return_dict: bool = False,
        return_latents: bool = False,
    ) -> Any:
        """Generate token sequences while keeping selected token positions fixed.

        ``max_length`` is the total length (condition plus generated tokens),
        not the number of new tokens. Without ``condition_token_mask``, valid
        inputs are compacted into a prefix. With an explicit boolean mask,
        positions are preserved, enabling BERT-style infill.
        """
        if condition_input_ids.ndim != 2:
            raise ValueError("condition_input_ids must have shape [batch, condition_length].")
        device = next(self.parameters()).device
        condition_input_ids = condition_input_ids.to(device=device, dtype=torch.long)
        if condition_attention_mask is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is None:
                condition_attention_mask = torch.ones_like(condition_input_ids)
            else:
                condition_attention_mask = condition_input_ids.ne(int(pad_token_id))
        condition_attention_mask = condition_attention_mask.to(device=device, dtype=torch.bool)
        if condition_attention_mask.shape != condition_input_ids.shape:
            raise ValueError("condition_attention_mask must have the same shape as condition_input_ids.")
        if condition_token_mask is not None:
            condition_token_mask = condition_token_mask.to(device=device, dtype=torch.bool)
            if condition_token_mask.shape != condition_input_ids.shape:
                raise ValueError("condition_token_mask must have the same shape as condition_input_ids.")

        num_steps = int(num_steps)
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        if sampling_method not in {"ode", "sde"}:
            raise ValueError("sampling_method must be either 'ode' or 'sde'.")
        if sde_gamma < 0:
            raise ValueError("sde_gamma must be >= 0.")
        model_max_length = int(
            self.config.dlm_config.get("max_length")
            or self.config.model_config.get("max_sequence_length")
            or self.elf_denoiser.max_length
        )
        total_length = model_max_length if max_length is None else int(max_length)
        condition_lengths = condition_attention_mask.sum(dim=1) if condition_token_mask is None else condition_token_mask.sum(dim=1)
        if total_length < 1 or total_length > model_max_length:
            raise ValueError(f"max_length must be in [1, {model_max_length}].")
        if bool((condition_lengths > total_length).any()):
            raise ValueError("A condition is longer than max_length.")
        if bool((condition_lengths < 1).any()):
            raise ValueError("Every sequence must contain at least one condition token.")

        context_dtype = next(self.elf_denoiser.parameters()).dtype
        encoder_attention_mask = condition_attention_mask
        if condition_token_mask is not None:
            # Match Stage-3 conditional training exactly. Known queries may
            # attend only known keys, so their fixed context representations
            # cannot absorb information from the unknown/generated span.
            # Unknown queries may attend every valid key. This structural mask
            # is only for the context encoder; the ELF denoiser below still
            # receives the ordinary valid-token mask.
            is_cond = condition_token_mask & condition_attention_mask
            is_valid = condition_attention_mask
            encoder_attention_mask = (
                (is_cond[:, :, None] & is_cond[:, None, :])
                | (~is_cond[:, :, None] & is_valid[:, None, :])
            )

        encoded = self.context_encoder(
            input_ids=condition_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state.to(dtype=context_dtype)
        batch_size = condition_input_ids.shape[0]
        hidden_size = encoded.shape[-1]
        if condition_token_mask is not None:
            if condition_input_ids.shape[1] != total_length:
                raise ValueError("With condition_token_mask, condition_input_ids length must equal max_length.")
            condition = encoded
            condition_mask = condition_token_mask & condition_attention_mask
            sequences = condition_input_ids.clone()
        else:
            condition = torch.zeros(batch_size, total_length, hidden_size, device=device, dtype=context_dtype)
            condition_mask = torch.zeros(batch_size, total_length, device=device, dtype=torch.bool)
            sequences = torch.full((batch_size, total_length), int(getattr(self.config, "pad_token_id", 0) or 0), device=device, dtype=torch.long)
            for row in range(batch_size):
                valid = condition_attention_mask[row]
                length = int(condition_lengths[row].item())
                condition[row, :length] = encoded[row, valid]
                condition_mask[row, :length] = True
                sequences[row, :length] = condition_input_ids[row, valid]

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        noise_scale = float(self.config.dlm_config.get("denoiser_noise_scale", 1.0))
        z = torch.randn(
            condition.shape, device=device, dtype=context_dtype, generator=generator
        ) * noise_scale
        z = self._restore_condition(z, condition, condition_mask)
        x_pred = self._restore_condition(torch.zeros_like(z), condition, condition_mask)
        attention_mask = torch.ones(batch_size, total_length, device=device, dtype=torch.bool)
        time_schedule = self.config.dlm_config.get("time_schedule", "logit_normal")
        if time_schedule == "uniform" or num_steps == 1:
            t_steps = torch.linspace(
                0.0, 1.0, num_steps + 1, device=device, dtype=context_dtype
            )
        elif time_schedule == "logit_normal":
            p_mean = float(self.config.dlm_config.get("denoiser_p_mean", -0.8))
            p_std = float(self.config.dlm_config.get("denoiser_p_std", 0.8))
            middle = torch.randn(
                num_steps - 1, device=device, dtype=context_dtype, generator=generator
            )
            middle = torch.sort(torch.sigmoid(middle * p_std + p_mean)).values
            t_steps = torch.cat(
                [middle.new_zeros(1), middle, middle.new_ones(1)], dim=0
            )
        else:
            raise ValueError(f"Unsupported time_schedule={time_schedule!r}.")

        for index in range(num_steps):
            t = t_steps[index]
            t_next = t_steps[index + 1]
            if sampling_method == "sde":
                h = t_next - t
                alpha = torch.clamp(1.0 - float(sde_gamma) * h, min=0.0, max=1.0)
                t_model = alpha * t
                eps = torch.randn(
                    z.shape, device=device, dtype=context_dtype, generator=generator
                ) * noise_scale
                z = alpha * z + (1.0 - alpha) * eps
                z = self._restore_condition(z, condition, condition_mask)
            else:
                t_model = t
            t_batch = torch.full(
                (batch_size,), float(t_model.item()), device=device, dtype=context_dtype
            )
            v_pred, x_pred = self._conditional_sample_step(
                z,
                t_batch,
                x_pred,
                condition,
                condition_mask,
                attention_mask,
                cfg_scale,
                self_cond_cfg_scale,
            )
            z = z + (t_next - t_model) * v_pred
            z = self._restore_condition(z, condition, condition_mask)

        decoder_input = z
        if int(getattr(self.elf_denoiser, "num_self_cond_cfg_tokens", 0)) > 0:
            decoder_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
        decoder_scale = torch.full(
            (batch_size,), float(self_cond_cfg_scale), device=device, dtype=context_dtype
        )
        _, logits = self.elf_denoiser(
            decoder_input,
            torch.ones(batch_size, device=device, dtype=context_dtype),
            attention_mask=attention_mask,
            self_cond_cfg_scale=decoder_scale,
            decoder_step_active=True,
        )
        if logits is None:
            raise RuntimeError("ELF decoder did not return logits.")
        generated_ids = torch.argmax(logits, dim=-1)
        sequences = torch.where(condition_mask, sequences, generated_ids)
        if not return_dict:
            return sequences
        output = {"sequences": sequences, "logits": logits}
        if return_latents:
            output["latents"] = z
            output["condition_mask"] = condition_mask
        return output

    def ode_from_context_hidden(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        ode_steps: int = 4,
        ode_start_t: float = 0.85,
        self_cond_cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        ode_steps = max(1, int(ode_steps))
        ode_start_t = float(ode_start_t)
        if not 0.0 < ode_start_t <= 1.0:
            raise ValueError("ode_start_t must be in (0, 1].")

        context_dtype = next(self.elf_denoiser.parameters()).dtype
        context = context.to(dtype=context_dtype)
        z = context
        x_pred = torch.zeros_like(z)
        t_steps = torch.linspace(
            ode_start_t,
            1.0,
            ode_steps + 1,
            device=context.device,
            dtype=context.dtype,
        )

        for idx in range(ode_steps):
            t = t_steps[idx]
            t_next = t_steps[idx + 1]
            t_batch = torch.full((z.shape[0],), float(t.detach().item()), device=z.device, dtype=z.dtype)
            v_pred, x_pred = self._elf_forward_ode_sample(
                z,
                t_batch,
                x_pred,
                attention_mask=attention_mask,
                self_cond_cfg_scale_value=self_cond_cfg_scale,
            )
            z = z + (t_next - t) * v_pred
            if attention_mask is not None:
                valid_mask = attention_mask.to(device=context.device, dtype=torch.bool).unsqueeze(-1)
                z = torch.where(valid_mask, z, context)
                x_pred = torch.where(valid_mask, x_pred, context)

        return z

    def sde_from_context_hidden(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        sde_steps: int = 4,
        sde_start_t: float = 0.85,
        sde_gamma: float = 0.1,
        self_cond_cfg_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        sde_steps = max(1, int(sde_steps))
        sde_start_t = float(sde_start_t)
        sde_gamma = float(sde_gamma)
        if not 0.0 < sde_start_t <= 1.0:
            raise ValueError("sde_start_t must be in (0, 1].")
        if sde_gamma < 0.0:
            raise ValueError("sde_gamma must be >= 0.")

        context_dtype = next(self.elf_denoiser.parameters()).dtype
        context = context.to(dtype=context_dtype)
        z = context
        x_pred = torch.zeros_like(z)
        t_steps = torch.linspace(
            sde_start_t,
            1.0,
            sde_steps + 1,
            device=context.device,
            dtype=context.dtype,
        )
        noise_scale = float(self.config.dlm_config.get("denoiser_noise_scale", 1.0))
        generator = None
        if seed is not None:
            generator = torch.Generator(device=context.device)
            generator.manual_seed(int(seed))

        for idx in range(sde_steps):
            t = t_steps[idx]
            t_next = t_steps[idx + 1]
            h = t_next - t
            alpha = torch.clamp(1.0 - sde_gamma * h, min=0.0, max=1.0)
            t_back = alpha * t
            eps = torch.randn(
                z.shape,
                device=z.device,
                dtype=z.dtype,
                generator=generator,
            ) * noise_scale
            z_back = alpha * z + (1.0 - alpha) * eps

            if attention_mask is not None:
                valid_mask = attention_mask.to(device=context.device, dtype=torch.bool).unsqueeze(-1)
                z_back = torch.where(valid_mask, z_back, context)

            t_batch = torch.full((z.shape[0],), float(t_back.detach().item()), device=z.device, dtype=z.dtype)
            v_pred, x_pred = self._elf_forward_ode_sample(
                z_back,
                t_batch,
                x_pred,
                attention_mask=attention_mask,
                self_cond_cfg_scale_value=self_cond_cfg_scale,
            )
            z = z_back + (t_next - t_back) * v_pred
            if attention_mask is not None:
                z = torch.where(valid_mask, z, context)
                x_pred = torch.where(valid_mask, x_pred, context)

        return z

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
        return_ode_hidden: bool = False,
        ode_steps: int = 4,
        ode_start_t: float = 0.85,
        ode_self_cond_cfg_scale: float = 1.0,
        return_sde_hidden: bool = False,
        sde_steps: int = 4,
        sde_start_t: float = 0.85,
        sde_gamma: float = 0.1,
        sde_self_cond_cfg_scale: float = 1.0,
        sde_seed: Optional[int] = None,
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
        if return_ode_hidden:
            output["ode_hidden_state"] = self.ode_from_context_hidden(
                context,
                attention_mask=attention_mask,
                ode_steps=ode_steps,
                ode_start_t=ode_start_t,
                self_cond_cfg_scale=ode_self_cond_cfg_scale,
            )
        if return_sde_hidden:
            output["sde_hidden_state"] = self.sde_from_context_hidden(
                context,
                attention_mask=attention_mask,
                sde_steps=sde_steps,
                sde_start_t=sde_start_t,
                sde_gamma=sde_gamma,
                self_cond_cfg_scale=sde_self_cond_cfg_scale,
                seed=sde_seed,
            )
        return output


__all__ = ["PoreDLMConfig", "PoreDLMForDiffusion"]
