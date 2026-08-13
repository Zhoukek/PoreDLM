# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass

from basecall.model import BasecallModel
from basecall.utils import BASE2ID, NUM_CLASSES

from .config import ModelConfig


@dataclass
class ModelBuild:
    model: BasecallModel
    num_classes: int
    n_base: int
    checkpoint_model_config: dict


def build_model(cfg: ModelConfig, *, resume_after_delayed_unfreeze: bool = False) -> ModelBuild:
    if not cfg.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless restored from checkpoint model_config.")
    if cfg.feature_source != "hidden" and cfg.learnable_fuse_last_n_layers > 0:
        raise ValueError("--learnable_fuse_last_n_layers requires --feature_source hidden.")

    n_base = len(BASE2ID)
    if n_base <= 0:
        raise ValueError("Alphabet must contain at least one non-blank base.")
    os.environ["CTC_CRF_STATE_LEN"] = str(cfg.ctc_crf_state_len)
    num_classes = (n_base ** cfg.ctc_crf_state_len) * (n_base + 1) if cfg.head_type == "ctc_crf" else NUM_CLASSES

    delayed_unfreeze = cfg.unfreeze_after_epoch > 0 and not resume_after_delayed_unfreeze
    initial_freeze = bool(cfg.freeze_backbone or delayed_unfreeze)
    model = BasecallModel(
        model_path=cfg.model_name_or_path,
        num_classes=num_classes,
        hidden_layer=cfg.hidden_layer,
        learnable_fuse_last_n_layers=cfg.learnable_fuse_last_n_layers,
        feature_source=cfg.feature_source,
        vq_device=cfg.vq_device,
        vq_token_batch_size=cfg.vq_token_batch_size,
        dlm_output=cfg.dlm_output,
        dlm_ode_steps=cfg.dlm_ode_steps,
        dlm_ode_start_t=cfg.dlm_ode_start_t,
        dlm_ode_self_cond_cfg_scale=cfg.dlm_ode_self_cond_cfg_scale,
        freeze_backbone=initial_freeze,
        reset_backbone_weights=cfg.reset_backbone_weights,
        unfreeze_last_n_layers=0 if delayed_unfreeze else cfg.unfreeze_last_n_layers,
        unfreeze_layer_start=None if delayed_unfreeze else cfg.unfreeze_layer_start,
        unfreeze_layer_end=None if delayed_unfreeze else cfg.unfreeze_layer_end,
        head_output_activation=cfg.head_output_activation,
        head_output_scale=cfg.head_output_scale,
        pre_head_type=cfg.pre_head_type,
        pre_head_transformer_nhead=cfg.pre_head_transformer_nhead,
        head_type=cfg.head_type,
        head_crf_blank_score=cfg.ctc_crf_blank_score,
        head_crf_n_base=n_base,
        head_crf_state_len=cfg.ctc_crf_state_len,
        head_crf_expand_blanks=True,
    )
    return ModelBuild(
        model=model,
        num_classes=num_classes,
        n_base=n_base,
        checkpoint_model_config=cfg.checkpoint_payload(num_classes=num_classes, n_base=n_base),
    )


def count_parameters(model) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def apply_delayed_unfreeze(
    model,
    cfg: ModelConfig,
    weight_decay: float,
    optimizer,
    backbone_lr: float | None = None,
    scheduler=None,
) -> int:
    raw_model = model
    if raw_model.backbone is None:
        raise ValueError("Delayed unfreeze is only supported for feature_source hidden/embedding.")
    if raw_model.feature_source != "hidden":
        raise ValueError("Delayed layer unfreeze requires feature_source='hidden'.")
    layers = raw_model._get_transformer_layers()
    n_layers = len(layers)
    if cfg.unfreeze_layer_start is not None or cfg.unfreeze_layer_end is not None:
        start = 0 if cfg.unfreeze_layer_start is None else int(cfg.unfreeze_layer_start)
        end = n_layers if cfg.unfreeze_layer_end is None else int(cfg.unfreeze_layer_end)
        if start < 0:
            start += n_layers
        if end < 0:
            end += n_layers
        if not 0 <= start <= end <= n_layers:
            raise ValueError(f"Invalid unfreeze layer range [{start}, {end}) for {n_layers} layers.")
        target_layer_indices = list(range(start, end))
    else:
        n_unfreeze = min(int(cfg.unfreeze_last_n_layers), n_layers)
        target_layer_indices = list(range(n_layers - n_unfreeze, n_layers)) if n_unfreeze > 0 else []

    raw_model._validate_unfreeze_layers_receive_grad(
        n_layers=n_layers,
        target_layer_indices=target_layer_indices,
        hidden_layer=raw_model.hidden_layer,
        learnable_fuse_last_n_layers=raw_model.learnable_fuse_last_n_layers,
    )
    for layer_idx in target_layer_indices:
        layer = layers[layer_idx]
        for param in layer.parameters():
            param.requires_grad = True
    raw_model.freeze_backbone = False

    tracked = {id(param) for group in optimizer.param_groups for param in group["params"]}
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    buckets = {
        ("head", True): [],
        ("head", False): [],
        ("backbone", True): [],
        ("backbone", False): [],
    }
    for name, param in raw_model.named_parameters():
        if not param.requires_grad or id(param) in tracked:
            continue
        kind = "backbone" if name.startswith("backbone.") else "head"
        use_decay = not any(key in name for key in no_decay_keywords)
        buckets[(kind, use_decay)].append(param)

    head_base_lr = float(optimizer.defaults.get("lr", optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])))
    current_head_lr = float(optimizer.param_groups[0]["lr"])
    schedule_factor = current_head_lr / head_base_lr if head_base_lr > 0 else 1.0
    backbone_base_lr = float(backbone_lr) if backbone_lr is not None else head_base_lr * 0.1
    new_base_lrs = []
    added_groups = 0
    for kind, use_decay in (
        ("head", True),
        ("head", False),
        ("backbone", True),
        ("backbone", False),
    ):
        params = buckets[(kind, use_decay)]
        if not params:
            continue
        group_base_lr = backbone_base_lr if kind == "backbone" else head_base_lr
        optimizer.add_param_group(
            {
                "params": params,
                "weight_decay": weight_decay if use_decay else 0.0,
                "lr": group_base_lr * schedule_factor,
                "initial_lr": group_base_lr,
                "group_name": f"{kind}_{'decay' if use_decay else 'no_decay'}",
            }
        )
        new_base_lrs.append(group_base_lr)
        added_groups += 1
    if scheduler is not None and added_groups > 0:
        if hasattr(scheduler, "base_lrs") and scheduler.base_lrs:
            scheduler.base_lrs.extend(new_base_lrs)
        if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
            scheduler.lr_lambdas.extend([scheduler.lr_lambdas[0]] * added_groups)
    return sum(len(params) for params in buckets.values())
