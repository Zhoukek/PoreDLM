"""Utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Controls global randomness and deterministic backend behavior."""

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    warn_only: bool = True


def build_reproducibility_config(config: dict[str, Any]) -> ReproducibilityConfig:
    """Build reproducibility settings from a project config dictionary."""

    settings = config.get("reproducibility", {})
    return ReproducibilityConfig(
        seed=int(settings.get("seed", config.get("seed", 42))),
        deterministic=bool(settings.get("deterministic", True)),
        benchmark=bool(settings.get("benchmark", False)),
        warn_only=bool(settings.get("warn_only", True)),
    )


def seed_everything(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def configure_determinism(config: ReproducibilityConfig) -> None:
    """Apply deterministic algorithm settings after RNG seeding."""

    seed_everything(
        seed=config.seed,
        deterministic=config.deterministic,
        benchmark=config.benchmark,
    )
    torch.use_deterministic_algorithms(config.deterministic, warn_only=config.warn_only)
