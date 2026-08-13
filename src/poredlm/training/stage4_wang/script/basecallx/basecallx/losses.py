# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

from basecall.ctc import ctc_label_smoothing_loss
from basecall.ctc_crf import ctc_crf_loss
from basecall.utils import BLANK_IDX


def compute_loss(
    model_cfg,
    logits_tbc: torch.Tensor,
    target_labels: torch.Tensor,
    target_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    if model_cfg.head_type == "ctc_crf":
        return ctc_crf_loss(
            logits_tbc,
            target_labels,
            input_lengths,
            target_lengths,
            blank_idx=BLANK_IDX,
        )
    loss_dict = ctc_label_smoothing_loss(
        logits_tbc,
        target_labels,
        target_lengths,
        input_lengths=input_lengths,
        blank_idx=BLANK_IDX,
    )
    return loss_dict["total_loss"]
