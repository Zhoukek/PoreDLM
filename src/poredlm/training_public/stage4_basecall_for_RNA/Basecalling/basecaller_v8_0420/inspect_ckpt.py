# -*- coding: utf-8 -*-
"""
Inspect checkpoint head config for the CTC-CRF linear head.

Example:
  python inspect_ckpt.py --ckpt ckpt_best.pt
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch

from .utils import infer_head_config_from_state_dict


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        return state
    raise ValueError(f"Unsupported checkpoint format: {path}")


def infer_pre_head_type_from_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    keys = sd.keys()
    if any(k.startswith("pre_head.blocks.") for k in keys):
        return "tcn"
    if any(k.startswith("pre_head.lstm.") for k in keys):
        return "bilstm"
    if any(k.startswith("pre_head.encoder.layers.") for k in keys):
        return "transformer"
    if any(k.startswith("pre_head.") for k in keys):
        return "custom_or_unknown"
    return "none"


def summarize_model_structure_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, object]:
    total_params = int(sum(v.numel() for v in sd.values() if torch.is_tensor(v)))
    return {
        "pre_head_type_inferred": infer_pre_head_type_from_state_dict(sd),
        "contains_backbone": any(k.startswith("backbone.") for k in sd),
        "contains_pre_head": any(k.startswith("pre_head.") for k in sd),
        "contains_base_head": any(k.startswith("base_head.") for k in sd),
        "total_parameters_in_checkpoint": total_params,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    sd = load_checkpoint_state(args.ckpt)
    head_config = infer_head_config_from_state_dict(sd)
    structure = summarize_model_structure_from_state_dict(sd)
    summary = {
        "num_classes": head_config["num_classes"],
        "head_type": head_config.get("head_type"),
        "model_structure": structure,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
