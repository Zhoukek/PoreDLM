from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class ModelConfig:
    context_encoder_path: str = ""
    freeze_context_encoder: bool = True
    size: str = "base"
    max_length: int = 1536
    vocab_size: int = 65664
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_condition_tokens: int = 4
    num_mode_tokens: int = 4
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0


@dataclass
class FlowMatchingConfig:
    timestep_mean: float = -1.5
    timestep_std: float = 0.8
    timestep_schedule: str = "logit_normal"
    noise_scale: float = 2.0
    timestep_epsilon: float = 0.05
    self_condition_probability: float = 0.5
    guidance_scale_min: float = 0.0
    guidance_scale_max: float = 1.0
    decoder_probability: float = 0.2
    decoder_noise_scale: float = 5.0
    decoder_timestep_mean: float = 0.8
    decoder_timestep_std: float = 0.8


@dataclass
class ConditioningConfig:
    mode: str = "mixed"
    unconditional_probability: float = 0.2
    pattern: str = "mixed"
    min_span_length: int = 30
    max_span_length: int = 50
    multi_min_spans: int = 4
    multi_max_spans: int = 8
    prefix_suffix_weight: float = 0.2
    single_span_weight: float = 0.3
    multi_span_weight: float = 0.5


@dataclass
class DataConfig:
    train_paths: list[str] = field(default_factory=list)
    eval_paths: list[str] = field(default_factory=list)
    dtype: str = "uint32"
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class OptimizerConfig:
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 200
    minimum_lr_ratio: float = 0.01


@dataclass
class TrackingConfig:
    enabled: bool = False
    project: str = "porelm-stage3"
    entity: str | None = None


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/flow_matching"
    seed: int = 6198
    epochs: int = 5
    global_batch_size: int = 64
    micro_batch_size: int = 8
    precision: str = "bf16"
    log_interval: int = 10
    eval_interval: int = 1000
    eval_batches: int = 20
    save_interval: int = 1000
    keep_checkpoints: int = 5
    resume_from: str | None = None


@dataclass
class ExperimentConfig:
    run_name: str = "porelm-flow-matching"
    objective: str = "flow_matching"
    model: ModelConfig = field(default_factory=ModelConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def validate(self) -> None:
        if self.objective != "flow_matching":
            raise ValueError(f"Unknown objective {self.objective!r}; available: flow_matching")
        if not self.model.context_encoder_path:
            raise ValueError("model.context_encoder_path is required")
        if not self.data.train_paths:
            raise ValueError("data.train_paths must contain at least one path")
        if self.training.global_batch_size % self.training.micro_batch_size:
            raise ValueError("training.global_batch_size must be divisible by training.micro_batch_size")
        if self.conditioning.mode not in {"unconditional", "conditional", "mixed"}:
            raise ValueError("conditioning.mode must be unconditional, conditional, or mixed")
        if self.conditioning.pattern not in {"mixed", "prefix_suffix", "single_span", "multi_span"}:
            raise ValueError("conditioning.pattern must select a supported generation task")
        if self.model.size not in {"base", "medium", "large"}:
            raise ValueError("model.size must be base, medium, or large")
        if self.training.precision not in {"bf16", "fp32"}:
            raise ValueError("training.precision must be bf16 or fp32")
        if not 0 <= self.flow_matching.self_condition_probability <= 1:
            raise ValueError("flow_matching.self_condition_probability must be in [0, 1]")
        if not 0 <= self.flow_matching.decoder_probability <= 1:
            raise ValueError("flow_matching.decoder_probability must be in [0, 1]")
        if self.flow_matching.guidance_scale_min < 0:
            raise ValueError("flow_matching.guidance_scale_min must be non-negative")
        if self.flow_matching.guidance_scale_max < self.flow_matching.guidance_scale_min:
            raise ValueError("flow_matching.guidance_scale_max must be at least guidance_scale_min")
        if self.flow_matching.noise_scale <= 0 or self.flow_matching.timestep_epsilon <= 0:
            raise ValueError("flow-matching noise scale and timestep epsilon must be positive")
        if not 0 <= self.conditioning.unconditional_probability <= 1:
            raise ValueError("conditioning.unconditional_probability must be in [0, 1]")
        if self.conditioning.min_span_length < 1:
            raise ValueError("conditioning.min_span_length must be positive")
        if self.conditioning.max_span_length < self.conditioning.min_span_length:
            raise ValueError("conditioning.max_span_length must be at least min_span_length")
        if self.conditioning.multi_min_spans < 1:
            raise ValueError("conditioning.multi_min_spans must be positive")
        if self.conditioning.multi_max_spans < self.conditioning.multi_min_spans:
            raise ValueError("conditioning.multi_max_spans must be at least multi_min_spans")
        if min(
            self.training.log_interval,
            self.training.eval_interval,
            self.training.save_interval,
        ) < 1:
            raise ValueError("training intervals must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _from_dict(cls: type[T], values: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {', '.join(unknown)}")
    converted = {}
    for key, value in values.items():
        target = hints[key]
        if is_dataclass(target) and isinstance(value, dict):
            value = _from_dict(target, value)
        if key == "betas" and isinstance(value, list):
            value = tuple(value)
        converted[key] = value
    return cls(**converted)


def _set_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"Unknown override path: {dotted_key}")
        target = target[part]
    if parts[-1] not in target:
        raise ValueError(f"Unknown override key: {dotted_key}")
    target[parts[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("The top-level configuration must be a mapping")
    defaults = ExperimentConfig().to_dict()

    def merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    merged = merge(defaults, values)
    for expression in overrides or []:
        _set_override(merged, expression)
    config = _from_dict(ExperimentConfig, merged)
    config.validate()
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PoreLM with flow matching")
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="Configuration overrides in key=value form")
    return parser.parse_args()
