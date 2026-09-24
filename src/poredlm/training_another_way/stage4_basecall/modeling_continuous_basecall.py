"""Basecalling head on top of the frozen/finetunable continuous route."""

from __future__ import annotations

import torch
from torch import nn

from poredlm.training_another_way.stage1_cnn_train.modeling_continuous_cnn import ContinuousSignalCNN
from poredlm.training_another_way.stage2_bert_train.modeling_continuous_bert import ContinuousBert
from poredlm.training_public.stage4_basecall.Basecalling.basecaller_v8_0420.model import (
    BiLSTMPreHead,
    IdentityPreHead,
    LinearCRFEncoder,
    LinearCTCEncoder,
    TCNPreHead,
    TransformerPreHead,
)


class ContinuousBasecallModel(nn.Module):
    def __init__(
        self,
        cnn_checkpoint: str,
        bert_checkpoint: str,
        num_classes: int = 5,
        freeze_cnn: bool = True,
        freeze_bert: bool = True,
        head_type: str = "ctc",
        ctc_crf_state_len: int = 5,
        ctc_crf_blank_score: float = 2.0,
        pre_head_type: str = "none",
        pre_head_transformer_nhead: int = 8,
        head_output_activation: str | None = None,
        head_output_scale: float | None = None,
    ):
        super().__init__()
        self.cnn = ContinuousSignalCNN.from_pretrained(cnn_checkpoint)
        self.bert = ContinuousBert.from_pretrained(bert_checkpoint)
        self.num_classes = int(num_classes)
        self.cnn_stride = int(self.cnn.stride)
        hidden_size = int(self.bert.config.hidden_size)
        if int(self.cnn.hidden_size) != int(self.bert.config.feature_dim):
            raise ValueError(
                f"CNN output dim {self.cnn.hidden_size} does not match "
                f"BERT feature_dim {self.bert.config.feature_dim}."
            )
        self.head_type = head_type
        if pre_head_type == "none":
            self.pre_head = IdentityPreHead(hidden_size)
        elif pre_head_type == "bilstm":
            self.pre_head = BiLSTMPreHead(hidden_size, hidden_dim=128)
        elif pre_head_type == "transformer":
            self.pre_head = TransformerPreHead(hidden_size, nhead=pre_head_transformer_nhead)
        elif pre_head_type == "tcn":
            self.pre_head = TCNPreHead(hidden_size)
        else:
            raise ValueError(f"Unsupported pre_head_type: {pre_head_type}")
        if head_type == "ctc":
            self.basecall_head = LinearCTCEncoder(
                self.pre_head.output_dim,
                self.num_classes,
                activation=head_output_activation,
                scale=head_output_scale,
            )
        elif head_type == "ctc_crf":
            if int(ctc_crf_state_len) < 1:
                raise ValueError("ctc_crf_state_len must be >= 1")
            n_base = self.num_classes - 1
            crf_classes = (n_base + 1) * (n_base ** int(ctc_crf_state_len))
            self.basecall_head = LinearCRFEncoder(
                self.pre_head.output_dim,
                n_base=n_base,
                state_len=int(ctc_crf_state_len),
                activation=head_output_activation,
                scale=head_output_scale,
                blank_score=ctc_crf_blank_score,
            )
            self.num_classes = crf_classes
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.freeze_cnn = bool(freeze_cnn)
        self.freeze_bert = bool(freeze_bert)
        if self.freeze_cnn:
            for parameter in self.cnn.parameters():
                parameter.requires_grad_(False)
        if self.freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_cnn:
            self.cnn.eval()
        if self.freeze_bert:
            self.bert.eval()
        return self

    def forward(
        self,
        signal: torch.Tensor,
        signal_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.freeze_cnn:
            with torch.no_grad():
                features = self.cnn.encode(signal).transpose(1, 2)
        else:
            features = self.cnn.encode(signal).transpose(1, 2)
        encoded_lengths = (signal_lengths + self.cnn_stride - 1) // self.cnn_stride
        attention_mask = torch.zeros(
            features.shape[:2], dtype=torch.long, device=features.device
        )
        positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
        attention_mask[positions < encoded_lengths.unsqueeze(1)] = 1
        if self.freeze_bert:
            with torch.no_grad():
                hidden = self.bert.encode(features, attention_mask)
        else:
            hidden = self.bert.encode(features, attention_mask)
        hidden = self.pre_head(hidden)
        return self.basecall_head(hidden), encoded_lengths
