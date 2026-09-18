"""HF-compatible Stage 2 masked language model for PoreDLM tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


@dataclass
class Stage2MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


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
        **kwargs,
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


class Stage2MaskedSignalLM(PreTrainedModel):
    config_class = Stage2BertConfig
    base_model_prefix = "stage2_bert"
    supports_gradient_checkpointing = False
    _tied_weights_keys = ["lm_head.weight"]


    def __init__(self, config: Stage2BertConfig) -> None:
        super().__init__(config)
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
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embeddings = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def _encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.final_layer_norm(hidden)

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._encode(input_ids=input_ids, attention_mask=attention_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Stage2MaskedLMOutput | Tuple[torch.Tensor, ...]:
        if input_ids is None and "data" in kwargs:
            input_ids = kwargs.pop("data")
        if input_ids is None:
            raise ValueError("Stage2MaskedSignalLM.forward requires input_ids or data.")
        del kwargs
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_hidden_states = bool(output_hidden_states)

        hidden = self._encode(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + ((hidden,),)
            return ((loss,) + output) if loss is not None else output

        return Stage2MaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
        )


AutoConfig.register(Stage2BertConfig.model_type, Stage2BertConfig)
AutoModel.register(Stage2BertConfig, Stage2MaskedSignalLM)
AutoModelForMaskedLM.register(Stage2BertConfig, Stage2MaskedSignalLM)
