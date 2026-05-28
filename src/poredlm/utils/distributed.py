"""Distributed training helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DistributedConfig:
    """Distributed runtime settings."""

    enabled: bool = True
    backend: str = "nccl"
    find_unused_parameters: bool = False


@dataclass(frozen=True)
class DistributedContext:
    """Resolved process information for single-card or distributed runs."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def build_distributed_config(config: dict[str, Any]) -> DistributedConfig:
    """Build distributed settings from a project config dictionary."""

    settings = config.get("distributed", {})
    return DistributedConfig(
        enabled=bool(settings.get("enabled", True)),
        backend=str(settings.get("backend", "nccl")),
        find_unused_parameters=bool(settings.get("find_unused_parameters", False)),
    )


def setup_distributed(config: DistributedConfig) -> DistributedContext:
    """Initialize torch.distributed when launched with torchrun."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    enabled = config.enabled and world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if enabled and not dist.is_initialized():
        backend = config.backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    """Destroy the process group if it was initialized."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(
    model: nn.Module,
    context: DistributedContext,
    config: DistributedConfig,
) -> nn.Module:
    """Move a model to device and wrap it with DistributedDataParallel if needed."""

    model = model.to(context.device)
    if not context.enabled:
        return model

    if context.device.type == "cuda":
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=config.find_unused_parameters,
        )

    return nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=config.find_unused_parameters,
    )


def barrier() -> None:
    """Synchronize all distributed workers when available."""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
