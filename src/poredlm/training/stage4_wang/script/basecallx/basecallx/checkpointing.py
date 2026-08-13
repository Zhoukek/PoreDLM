# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from basecall.utils import split_checkpoint_payload


def load_checkpoint_payload(path: str) -> tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict, model_config = split_checkpoint_payload(payload)
    full = payload if isinstance(payload, dict) else {}
    return state_dict, model_config, full


def load_model_weights(model, state_dict: Dict[str, torch.Tensor], *, strict: bool = False):
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key[len("module."):] if isinstance(key, str) and key.startswith("module.") else key
        normalized[clean_key] = value
    return model.load_state_dict(normalized, strict=strict)


def save_checkpoint(
    path: str,
    accelerator,
    model,
    *,
    optimizer=None,
    scheduler=None,
    epoch: int,
    best_acc: float,
    global_step: int,
    model_config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    raw_model = accelerator.unwrap_model(model)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_pbma": float(best_acc),
        "global_step": int(global_step),
        "model_config": dict(model_config),
        "model_state_dict": raw_model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    accelerator.save(payload, path)


def restore_training_state(
    checkpoint: Dict[str, Any],
    *,
    optimizer=None,
    scheduler=None,
    logger=None,
) -> tuple[int, float, int]:
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if logger:
            logger.info("[Resume] optimizer state loaded")
    if scheduler is not None and "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if logger:
            logger.info("[Resume] scheduler state loaded")
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_acc = float(checkpoint.get("best_pbma", -1.0))
    global_step = int(checkpoint.get("global_step", 0) or 0)
    return start_epoch, best_acc, global_step
