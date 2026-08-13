# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
import socket
from datetime import timedelta
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

from .config import RuntimeConfig


def _gpu_socket_preflight(backend: str) -> Tuple[bool, Optional[str], Optional[str]]:
    socket_env_name = "NCCL_SOCKET_IFNAME"
    if os.environ.get(socket_env_name):
        iface_names = {name for _, name in socket.if_nameindex()}
        requested = [x.strip() for x in os.environ[socket_env_name].split(",") if x.strip()]
        advanced = [item for item in requested if item.startswith("=") or item.startswith("^")]
        if advanced:
            return False, f"{socket_env_name} uses advanced syntax {advanced!r}", None
        matched = [name for name in requested if name in iface_names]
        if matched:
            return True, None, ",".join(matched)
        return False, f"{socket_env_name} does not match visible interfaces {sorted(iface_names)}", ",".join(requested)

    iface_names = [name for _, name in socket.if_nameindex()]
    non_loopback = [name for name in iface_names if name != "lo" and not name.startswith("lo:")]
    if non_loopback:
        return True, None, non_loopback[0]
    return False, f"no non-loopback interface found: {iface_names}", None


def resolve_backend(runtime: RuntimeConfig) -> Tuple[str, Optional[str]]:
    ddp_env = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    backend = runtime.ddp_backend
    note: Optional[str] = None
    if ddp_env and backend == "nccl" and runtime.ddp_backend_fallback:
        socket_ok, reason, normalized = _gpu_socket_preflight(backend)
        if not socket_ok:
            backend = "gloo"
            note = f"NCCL preflight failed: {reason}. Falling back to gloo."
        elif normalized and normalized != os.environ.get("NCCL_SOCKET_IFNAME"):
            old = os.environ.get("NCCL_SOCKET_IFNAME")
            os.environ["NCCL_SOCKET_IFNAME"] = normalized
            note = f"Normalized NCCL_SOCKET_IFNAME from {old!r} to {normalized!r}."

    if backend == "nccl" and hasattr(dist, "is_nccl_available") and not dist.is_nccl_available():
        if runtime.ddp_backend_fallback:
            return "gloo", "NCCL is not available. Falling back to gloo."
        raise RuntimeError("NCCL backend requested but torch.distributed NCCL is unavailable.")
    return backend, note


def create_accelerator(runtime: RuntimeConfig, *, streaming: bool) -> Tuple[Accelerator, str, Optional[str]]:
    backend, note = resolve_backend(runtime)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=runtime.find_unused_parameters,
        broadcast_buffers=runtime.ddp_broadcast_buffers,
    )
    kwargs_handlers = [ddp_kwargs]
    if int(os.environ.get("WORLD_SIZE", "1") or 1) > 1:
        kwargs_handlers.append(
            InitProcessGroupKwargs(backend=backend, timeout=timedelta(minutes=30))
        )
    data_config = DataLoaderConfiguration(
        dispatch_batches=False if streaming else None,
        split_batches=False,
    )
    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
        dataloader_config=data_config,
        mixed_precision="fp16" if runtime.amp and torch.cuda.is_available() else "no",
        log_with="wandb" if runtime.use_wandb else None,
    )
    return accelerator, backend, note


def is_main(accelerator: Accelerator) -> bool:
    return accelerator.is_main_process


def reduce_mean(accelerator: Accelerator, value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    return float(accelerator.gather_for_metrics(tensor.unsqueeze(0)).mean().item())


def reduce_min_bool(accelerator: Accelerator, value: bool, device: torch.device) -> bool:
    tensor = torch.tensor(1 if value else 0, dtype=torch.int, device=device)
    return bool(accelerator.gather(tensor.unsqueeze(0)).min().item())


def setup_logger(output_dir: str, accelerator: Accelerator) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("basecallx")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    if not is_main(accelerator):
        logger.addHandler(logging.NullHandler())
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
