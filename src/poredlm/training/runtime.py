"""Shared training runtime bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poredlm.utils.config import load_yaml
from poredlm.utils.distributed import (
    DistributedConfig,
    DistributedContext,
    build_distributed_config,
    setup_distributed,
)
from poredlm.utils.reproducibility import (
    ReproducibilityConfig,
    build_reproducibility_config,
    configure_determinism,
)


@dataclass(frozen=True)
class TrainingRuntime:
    """Resolved config and runtime state used by all training stages."""

    config: dict[str, Any]
    reproducibility: ReproducibilityConfig
    distributed: DistributedConfig
    context: DistributedContext


def bootstrap_training(config_path: str) -> TrainingRuntime:
    """Load config, seed RNGs, and initialize distributed training."""

    config = load_yaml(config_path)
    reproducibility = build_reproducibility_config(config)
    configure_determinism(reproducibility)
    distributed = build_distributed_config(config)
    context = setup_distributed(distributed)

    return TrainingRuntime(
        config=config,
        reproducibility=reproducibility,
        distributed=distributed,
        context=context,
    )
