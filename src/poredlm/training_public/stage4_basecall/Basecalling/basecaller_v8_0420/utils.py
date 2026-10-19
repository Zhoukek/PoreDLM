# -*- coding: utf-8 -*-
import random
from typing import Dict, Any

import numpy as np
import torch

# 标签定义
LABELS = ["N", "A", "C", "G", "T"]  # 0,1,2,3,4
NUM_CLASSES = len(LABELS)
BLANK_IDX = 0  # CTC blank

ID2BASE = {
    0: "N",
    1: "A",
    2: "C",
    3: "G",
    4: "T",
}

BASE2ID = {b: i for i, b in ID2BASE.items()}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_lengths(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    input_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if input_lengths is not None:
        lengths = input_lengths.to(device=input_ids.device, dtype=torch.long)
    elif attention_mask is not None:
        lengths = attention_mask.sum(dim=1).to(device=input_ids.device, dtype=torch.long)
    else:
        lengths = torch.full(
            (input_ids.size(0),),
            input_ids.size(1),
            dtype=torch.long,
            device=input_ids.device,
        )

    if lengths.numel() == 0:
        return lengths

    if torch.any(lengths < 0):
        raise ValueError("input_lengths must be non-negative.")

    max_len = int(input_ids.size(1))
    if torch.any(lengths > max_len):
        lengths = torch.clamp(lengths, max=max_len)
    return lengths


def infer_head_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    default_num_classes: int | None = None,
) -> Dict[str, Any]:
    if default_num_classes is None:
        default_num_classes = len(ID2BASE)

    def _infer_head_type() -> str:
        if isinstance(state_dict.get("base_head.linear.weight"), torch.Tensor):
            return "ctc_crf"
        if isinstance(state_dict.get("base_head.proj.weight"), torch.Tensor):
            return "ctc"
        return "ctc"

    def _infer_num_classes() -> int:
        head_type = _infer_head_type()
        if head_type == "ctc":
            weight = state_dict.get("base_head.proj.weight")
            if isinstance(weight, torch.Tensor) and weight.dim() == 2:
                return int(weight.shape[0])
        weight = state_dict.get("base_head.linear.weight")
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            return int(weight.shape[0])
        weight = state_dict.get("base_head.proj.weight")
        if isinstance(weight, torch.Tensor) and weight.dim() == 2:
            return int(weight.shape[0])
        return int(default_num_classes)
    return {
        "head_type": _infer_head_type(),
        "num_classes": int(_infer_num_classes()),
    }


def infer_pre_head_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = state_dict.keys()
    if any(k.startswith("pre_head.blocks.") for k in keys):
        return "tcn"
    if any(k.startswith("pre_head.lstm.") for k in keys):
        return "bilstm"
    if any(k.startswith("pre_head.encoder.layers.") for k in keys):
        return "transformer"
    return "none"
