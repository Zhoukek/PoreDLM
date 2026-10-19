from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    objective: Any,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    config: dict[str, Any],
    keep: int,
) -> Path:
    checkpoint_dir = output_dir / f"step-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    core_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": core_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "objective": objective.state_dict(),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "step": step,
            "config": config,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_dir / "training_state.pt",
    )
    checkpoints = sorted(output_dir.glob("step-*"))
    if keep > 0:
        for old_checkpoint in checkpoints[:-keep]:
            state_file = old_checkpoint / "training_state.pt"
            if state_file.exists():
                state_file.unlink()
            try:
                old_checkpoint.rmdir()
            except OSError:
                pass
    return checkpoint_dir


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    objective: Any,
    device: torch.device,
) -> tuple[int, int, int]:
    path = Path(path)
    state_path = path / "training_state.pt" if path.is_dir() else path
    state = torch.load(state_path, map_location=device, weights_only=False)
    core_model = model.module if hasattr(model, "module") else model
    core_model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    objective.load_state_dict(state.get("objective", {}))
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    return int(state["epoch"]), int(state.get("batch_in_epoch", 0)), int(state["step"])
