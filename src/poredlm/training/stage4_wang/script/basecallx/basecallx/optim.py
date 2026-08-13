# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import torch

from .config import TrainConfig


def build_optimizer(model, cfg: TrainConfig) -> torch.optim.Optimizer:
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    head_lr = float(cfg.lr)
    backbone_lr = float(cfg.backbone_lr) if cfg.backbone_lr is not None else head_lr * 0.1
    buckets = {
        ("head", True): [],
        ("head", False): [],
        ("backbone", True): [],
        ("backbone", False): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        kind = "backbone" if name.startswith("backbone.") else "head"
        use_decay = not any(key in name for key in no_decay_keywords)
        buckets[(kind, use_decay)].append(param)
    groups = []
    for kind, use_decay in (
        ("head", True),
        ("head", False),
        ("backbone", True),
        ("backbone", False),
    ):
        params = buckets[(kind, use_decay)]
        if not params:
            continue
        group_lr = backbone_lr if kind == "backbone" else head_lr
        groups.append(
            {
                "params": params,
                "weight_decay": cfg.weight_decay if use_decay else 0.0,
                "lr": group_lr,
                "initial_lr": group_lr,
                "group_name": f"{kind}_{'decay' if use_decay else 'no_decay'}",
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=head_lr,
    )


def build_scheduler(optimizer, cfg: TrainConfig, *, steps_per_epoch: int):
    total_steps = max(int(steps_per_epoch) * int(cfg.num_epochs), 1)
    warmup_steps = int(cfg.warmup_steps) if cfg.warmup_steps >= 0 else int(total_steps * cfg.warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    base_lr = float(optimizer.param_groups[0]["lr"] or 1e-8)
    min_ratio = min(max(float(cfg.min_lr), 0.0) / base_lr, 1.0)

    def lr_lambda(current_step: int) -> float:
        step = min(max(int(current_step), 0), total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(max(warmup_steps, 1))
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), total_steps, warmup_steps
