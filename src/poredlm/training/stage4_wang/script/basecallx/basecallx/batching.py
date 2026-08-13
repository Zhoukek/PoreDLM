# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import torch

from basecall.utils import resolve_input_lengths


@dataclass(frozen=True)
class PreparedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    input_lengths: torch.Tensor
    target_labels: torch.Tensor
    target_lengths: torch.Tensor


def prepare_batch(batch: dict, device: torch.device) -> PreparedBatch:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    input_lengths = resolve_input_lengths(
        input_ids,
        attention_mask=attention_mask,
        input_lengths=batch.get("input_lengths"),
    )
    target_labels = batch["target_labels"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    return PreparedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        target_labels=target_labels,
        target_lengths=target_lengths,
    )
