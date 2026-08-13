# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    jsonl_paths: Optional[str] = None
    train_jsonl_paths: Optional[str] = None
    val_jsonl_paths: Optional[str] = None
    test_jsonl_paths: Optional[str] = None
    group_by: str = "file"
    recursive: bool = False
    token_offset: int = 0
    train_ratio: float = 0.98
    val_ratio: float = 0.01
    test_ratio: float = 0.01
    split_seed: int = 42
    streaming: bool = True
    shuffle_buffer_size: int = 0
    allow_eager_record_split: bool = False


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    hidden_layer: int = -1
    learnable_fuse_last_n_layers: int = 0
    feature_source: str = "hidden"
    vq_device: str = "cuda"
    vq_token_batch_size: int = 100
    dlm_output: str = "last"
    dlm_ode_steps: int = 2
    dlm_ode_start_t: float = 0.98
    dlm_ode_self_cond_cfg_scale: float = 0.0
    freeze_backbone: bool = True
    reset_backbone_weights: bool = False
    unfreeze_last_n_layers: int = 0
    unfreeze_after_epoch: int = 0
    unfreeze_layer_start: Optional[int] = None
    unfreeze_layer_end: Optional[int] = None
    head_output_activation: Optional[str] = "tanh"
    head_output_scale: Optional[float] = 5.0
    head_type: str = "ctc"
    pre_head_type: str = "none"
    pre_head_transformer_nhead: int = 8
    ctc_crf_state_len: int = 5
    ctc_crf_blank_score: float = 2.0

    def apply_checkpoint_config(self, saved: Dict[str, Any], *, keep_model_path: bool) -> None:
        if not saved:
            return
        for key in asdict(self).keys():
            if key not in saved:
                continue
            if key == "model_name_or_path" and keep_model_path:
                continue
            setattr(self, key, saved[key])

    def checkpoint_payload(self, *, num_classes: int, n_base: int) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "num_classes": int(num_classes),
                "head_crf_n_base": int(n_base),
                "head_crf_expand_blanks": True,
            }
        )
        return payload


@dataclass
class TrainConfig:
    output_dir: str = "outputs_basecallx"
    batch_size: int = 128
    num_epochs: int = 50
    max_steps_per_epoch: int = 0
    steps_per_epoch: int = 0
    num_workers: int = 0
    lr: float = 5e-4
    backbone_lr: Optional[float] = None
    weight_decay: float = 5e-4
    warmup_ratio: float = 0.1
    warmup_steps: int = -1
    min_lr: float = 1e-5
    clip_grad_norm: float = 2.0
    seed: int = 42
    log_interval: int = 100
    eval_interval: int = 0
    save_every: int = 1
    save_best: bool = True
    resume_ckpt: Optional[str] = None
    pretrained_ckpt: Optional[str] = None
    pretrained_strict: bool = False
    train_decoder: str = "ctc_viterbi"
    koi_blank_score: float = 2.0
    acc_balanced: bool = False
    acc_min_coverage: float = 0.0


@dataclass
class RuntimeConfig:
    ddp_backend: str = "nccl"
    ddp_backend_fallback: bool = False
    find_unused_parameters: bool = False
    ddp_broadcast_buffers: bool = False
    amp: bool = False
    use_wandb: bool = False
    wandb_project: str = "basecaller"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    runtime: RuntimeConfig


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    data = parser.add_argument_group("data")
    data.add_argument("--jsonl_paths", default=None)
    data.add_argument("--train_jsonl_paths", default=None)
    data.add_argument("--val_jsonl_paths", default=None)
    data.add_argument("--test_jsonl_paths", default=None)
    data.add_argument("--group_by", choices=["folder", "file", "record", "record_per_file"], default="file")
    data.add_argument("--recursive", action="store_true")
    data.add_argument("--token_offset", type=int, default=0)
    data.add_argument("--train_ratio", type=float, default=0.98)
    data.add_argument("--val_ratio", type=float, default=0.01)
    data.add_argument("--test_ratio", type=float, default=0.01)
    data.add_argument("--split_seed", type=int, default=42)
    data.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--shuffle_buffer_size", type=int, default=0)
    data.add_argument("--allow_eager_record_split", action="store_true")

    model = parser.add_argument_group("model")
    model.add_argument("--model_name_or_path", default=None)
    model.add_argument("--hidden_layer", "--hidden-layer", type=int, default=-1)
    model.add_argument("--learnable_fuse_last_n_layers", type=int, default=0)
    model.add_argument("--feature_source", "--feature-source", choices=["hidden", "embedding", "vq_embedding"], default="hidden")
    model.add_argument("--vq_device", default="cuda")
    model.add_argument("--vq_token_batch_size", type=int, default=100)
    model.add_argument("--dlm_output", choices=["last", "context", "ode"], default="last")
    model.add_argument("--dlm_ode_steps", type=int, default=2)
    model.add_argument("--dlm_ode_start_t", type=float, default=0.98)
    model.add_argument("--dlm_ode_self_cond_cfg_scale", type=float, default=0.0)
    model.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--reset_backbone_weights", action="store_true")
    model.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    model.add_argument("--unfreeze_after_epoch", type=int, default=0)
    model.add_argument("--unfreeze_layer_start", type=int, default=None)
    model.add_argument("--unfreeze_layer_end", type=int, default=None)
    model.add_argument("--head_output_activation", choices=["none", "tanh", "relu"], default="tanh")
    model.add_argument("--head_output_scale", type=float, default=5.0)
    model.add_argument("--head_type", choices=["ctc", "ctc_crf"], default="ctc")
    model.add_argument("--pre_head_type", choices=["none", "bilstm", "transformer", "tcn", "tiny_tcn"], default="none")
    model.add_argument("--pre_head_transformer_nhead", type=int, default=8)
    model.add_argument("--ctc_crf_state_len", type=int, default=5)
    model.add_argument("--ctc_crf_blank_score", type=float, default=2.0)

    train = parser.add_argument_group("training")
    train.add_argument("--output_dir", default="outputs_basecallx")
    train.add_argument("--batch_size", type=int, default=128)
    train.add_argument("--num_epochs", type=int, default=50)
    train.add_argument("--max_steps_per_epoch", type=int, default=0)
    train.add_argument("--steps_per_epoch", type=int, default=0)
    train.add_argument("--num_workers", type=int, default=0)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--backbone_lr", type=float, default=None)
    train.add_argument("--weight_decay", type=float, default=5e-4)
    train.add_argument("--warmup_ratio", type=float, default=0.1)
    train.add_argument("--warmup_steps", type=int, default=-1)
    train.add_argument("--min_lr", type=float, default=1e-5)
    train.add_argument("--clip_grad_norm", type=float, default=2.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log_interval", type=int, default=100)
    train.add_argument("--eval_interval", type=int, default=0)
    train.add_argument("--save_every", type=int, default=1)
    train.add_argument("--save_best", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--resume_ckpt", default=None)
    train.add_argument("--pretrained_ckpt", default=None)
    train.add_argument("--pretrained_strict", action="store_true")
    train.add_argument("--train_decoder", choices=["ctc_viterbi", "ctc_crf", "koi"], default="ctc_viterbi")
    train.add_argument("--koi_blank_score", type=float, default=2.0)
    train.add_argument("--acc_balanced", action="store_true")
    train.add_argument("--acc_min_coverage", type=float, default=0.0)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--ddp_backend", choices=["nccl", "gloo"], default="nccl")
    runtime.add_argument("--ddp_backend_fallback", action="store_true")
    runtime.add_argument("--find_unused_parameters", action="store_true")
    runtime.add_argument("--ddp_broadcast_buffers", action="store_true")
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--use_wandb", action="store_true")
    runtime.add_argument("--wandb_project", default="basecaller")
    runtime.add_argument("--wandb_run_name", default=None)
    runtime.add_argument("--wandb_entity", default=None)
    runtime.add_argument("--wandb_group", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    activation = None if args.head_output_activation == "none" else args.head_output_activation
    model_path = args.model_name_or_path
    return ExperimentConfig(
        data=DataConfig(
            jsonl_paths=args.jsonl_paths,
            train_jsonl_paths=args.train_jsonl_paths,
            val_jsonl_paths=args.val_jsonl_paths,
            test_jsonl_paths=args.test_jsonl_paths,
            group_by=args.group_by,
            recursive=bool(args.recursive),
            token_offset=int(args.token_offset),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            split_seed=int(args.split_seed),
            streaming=bool(args.streaming),
            shuffle_buffer_size=int(args.shuffle_buffer_size),
            allow_eager_record_split=bool(args.allow_eager_record_split),
        ),
        model=ModelConfig(
            model_name_or_path=model_path,
            hidden_layer=int(args.hidden_layer),
            learnable_fuse_last_n_layers=int(args.learnable_fuse_last_n_layers),
            feature_source=args.feature_source,
            vq_device=args.vq_device,
            vq_token_batch_size=int(args.vq_token_batch_size),
            dlm_output=args.dlm_output,
            dlm_ode_steps=int(args.dlm_ode_steps),
            dlm_ode_start_t=float(args.dlm_ode_start_t),
            dlm_ode_self_cond_cfg_scale=float(args.dlm_ode_self_cond_cfg_scale),
            freeze_backbone=bool(args.freeze_backbone),
            reset_backbone_weights=bool(args.reset_backbone_weights),
            unfreeze_last_n_layers=int(args.unfreeze_last_n_layers),
            unfreeze_after_epoch=int(args.unfreeze_after_epoch),
            unfreeze_layer_start=args.unfreeze_layer_start,
            unfreeze_layer_end=args.unfreeze_layer_end,
            head_output_activation=activation,
            head_output_scale=args.head_output_scale,
            head_type=args.head_type,
            pre_head_type=args.pre_head_type,
            pre_head_transformer_nhead=int(args.pre_head_transformer_nhead),
            ctc_crf_state_len=int(args.ctc_crf_state_len),
            ctc_crf_blank_score=float(args.ctc_crf_blank_score),
        ),
        train=TrainConfig(
            output_dir=args.output_dir,
            batch_size=int(args.batch_size),
            num_epochs=int(args.num_epochs),
            max_steps_per_epoch=int(args.max_steps_per_epoch),
            steps_per_epoch=int(args.steps_per_epoch),
            num_workers=int(args.num_workers),
            lr=float(args.lr),
            backbone_lr=args.backbone_lr,
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            warmup_steps=int(args.warmup_steps),
            min_lr=float(args.min_lr),
            clip_grad_norm=float(args.clip_grad_norm),
            seed=int(args.seed),
            log_interval=int(args.log_interval),
            eval_interval=int(args.eval_interval),
            save_every=int(args.save_every),
            save_best=bool(args.save_best),
            resume_ckpt=args.resume_ckpt,
            pretrained_ckpt=args.pretrained_ckpt,
            pretrained_strict=bool(args.pretrained_strict),
            train_decoder=args.train_decoder,
            koi_blank_score=float(args.koi_blank_score),
            acc_balanced=bool(args.acc_balanced),
            acc_min_coverage=float(args.acc_min_coverage),
        ),
        runtime=RuntimeConfig(
            ddp_backend=args.ddp_backend,
            ddp_backend_fallback=bool(args.ddp_backend_fallback),
            find_unused_parameters=bool(args.find_unused_parameters),
            ddp_broadcast_buffers=bool(args.ddp_broadcast_buffers),
            amp=bool(args.amp),
            use_wandb=bool(args.use_wandb),
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group,
        ),
    )
