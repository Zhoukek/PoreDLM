from __future__ import annotations

import logging
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, load_config, parse_args
from .data import build_dataloader
from .model import PoreLM
from .objectives import OBJECTIVES

LOGGER = logging.getLogger("porelm_training")


def distributed_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def setup_distributed() -> tuple[int, int, int]:
    rank, local_rank, world_size = distributed_info()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if device.type == "cpu" and dtype == torch.float16:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    values = {key: value.detach().float() for key, value in metrics.items()}
    if world_size > 1:
        for value in values.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value.div_(world_size)
    return {key: value.item() for key, value in values.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    objective: Any,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_context(device, precision):
            output = objective(model, batch)
        current = {"loss": output.loss.detach(), **output.metrics}
        for key, value in current.items():
            totals[key] = totals.get(key, torch.zeros_like(value)) + value
        count += 1
        if count >= max_batches:
            break
    model.train()
    if count == 0:
        return {}
    return reduce_metrics({key: value / count for key, value in totals.items()}, world_size)


def train(config: ExperimentConfig) -> None:
    rank, local_rank, world_size = setup_distributed()
    device = get_device(local_rank)
    seed_everything(config.training.seed, rank)
    if world_size > 1 and not config.model.freeze_context_encoder:
        raise ValueError("Multi-GPU training currently requires model.freeze_context_encoder=true")
    if config.training.global_batch_size % world_size:
        raise ValueError("global_batch_size must be divisible by WORLD_SIZE")
    per_device_batch = config.training.global_batch_size // world_size
    if per_device_batch % config.training.micro_batch_size:
        raise ValueError("Per-device batch size must be divisible by micro_batch_size")
    accumulation_steps = per_device_batch // config.training.micro_batch_size

    output_dir = Path(config.training.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    if world_size > 1:
        dist.barrier()

    train_loader, train_sampler = build_dataloader(
        config.data.train_paths,
        config.data,
        config.conditioning,
        max_length=config.model.max_length,
        batch_size=config.training.micro_batch_size,
        distributed=world_size > 1,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = None
    if config.data.eval_paths:
        eval_loader, _ = build_dataloader(
            config.data.eval_paths,
            config.data,
            config.conditioning,
            max_length=config.model.max_length,
            batch_size=config.training.micro_batch_size,
            distributed=world_size > 1,
            shuffle=False,
            drop_last=False,
        )

    model: torch.nn.Module = PoreLM(config.model).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    objective = OBJECTIVES[config.objective](config.flow_matching)
    objective.setup(model.module if hasattr(model, "module") else model, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.epsilon,
        weight_decay=config.optimizer.weight_decay,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation_steps))
    total_steps = updates_per_epoch * config.training.epochs
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        total_steps=total_steps,
        minimum_ratio=config.optimizer.minimum_lr_ratio,
    )
    start_epoch, start_batch, global_step = 0, 0, 0
    if config.training.resume_from:
        start_epoch, start_batch, global_step = load_checkpoint(
            config.training.resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=objective,
            device=device,
        )

    tracker = None
    if config.tracking.enabled and rank == 0:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install the tracking extra to enable experiment tracking") from error
        tracker = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.entity,
            name=config.run_name,
            config=config.to_dict(),
            dir=output_dir,
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.training.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        final_group_size = len(train_loader) % accumulation_steps or accumulation_steps
        for micro_step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and micro_step <= start_batch:
                continue
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            in_final_group = micro_step > len(train_loader) - final_group_size
            group_size = final_group_size if in_final_group else accumulation_steps
            synchronize = micro_step % accumulation_steps == 0 or micro_step == len(train_loader)
            sync_context = nullcontext()
            if world_size > 1 and not synchronize:
                sync_context = model.no_sync()  # type: ignore[attr-defined]
            with sync_context:
                with autocast_context(device, config.training.precision):
                    output = objective(model, batch)
                    loss = output.loss / group_size
                loss.backward()
            if not synchronize:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
            optimizer.step()
            scheduler.step()
            objective.after_optimizer_step(model.module if hasattr(model, "module") else model)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config.training.log_interval == 0:
                metrics = reduce_metrics(
                    {"loss": output.loss, "grad_norm": grad_norm, **output.metrics}, world_size
                )
                metrics["learning_rate"] = scheduler.get_last_lr()[0]
                if rank == 0:
                    LOGGER.info("step=%d %s", global_step, " ".join(f"{k}={v:.5g}" for k, v in metrics.items()))
                    if tracker is not None:
                        tracker.log({f"train/{key}": value for key, value in metrics.items()}, step=global_step)

            if eval_loader is not None and global_step % config.training.eval_interval == 0:
                metrics = evaluate(
                    model,
                    objective,
                    eval_loader,
                    device,
                    config.training.precision,
                    config.training.eval_batches,
                    world_size,
                )
                if rank == 0:
                    LOGGER.info("evaluation step=%d %s", global_step, metrics)
                    if tracker is not None:
                        tracker.log({f"eval/{key}": value for key, value in metrics.items()}, step=global_step)

            if rank == 0 and global_step % config.training.save_interval == 0:
                save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    objective=objective,
                    epoch=epoch,
                    batch_in_epoch=micro_step,
                    step=global_step,
                    config=config.to_dict(),
                    keep=config.training.keep_checkpoints,
                )
        start_batch = 0

    if rank == 0:
        save_checkpoint(
            output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            objective=objective,
            epoch=config.training.epochs,
            batch_in_epoch=0,
            step=global_step,
            config=config.to_dict(),
            keep=config.training.keep_checkpoints,
        )
        if tracker is not None:
            tracker.finish()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
