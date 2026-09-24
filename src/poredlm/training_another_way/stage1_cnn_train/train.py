"""Train the continuous 1D-CNN warm-up model."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import PoreSignalDataset
from modeling_continuous_cnn import ContinuousCNNConfig, ContinuousSignalCNN


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_loader(cfg: dict, train: bool):
    data = cfg["data"]["train" if train else "valid"]
    dataset = PoreSignalDataset(
        data["paths"], chunk_size=data.get("chunk_size", 6000),
        memmap_dtype=data.get("memmap_dtype", "float32"),
        buffer_size=data.get("buffer_size", 20000 if train else 0),
        shuffle_buffer=train, repeat=train, seed=cfg.get("seed", 42),
    )
    batch_size = cfg["training"].get("device_micro_batch_size", cfg["training"].get("device_train_microbatch_size", 8))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=data.get("num_workers", 0), pin_memory=data.get("pin_memory", True),
        persistent_workers=data.get("persistent_workers", False) and data.get("num_workers", 0) > 0,
        prefetch_factor=data.get("prefetch_factor", 2) if data.get("num_workers", 0) > 0 else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    seed_everything(int(cfg.get("seed", 42)))
    accelerator = Accelerator(mixed_precision=cfg["training"].get("mixed_precision", "no"))
    model = ContinuousSignalCNN(ContinuousCNNConfig(**cfg.get("model", {})))
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 0.01))
    train_loader = build_loader(cfg, True)
    valid_loader = build_loader(cfg, False)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    output_dir = Path(cfg["training"]["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["training"]["max_train_steps"])
    eval_every = int(cfg["training"].get("eval_every_steps", 1000))
    save_every = int(cfg["training"].get("save_every_steps", 5000))
    step = 0
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    while step < max_steps:
        for batch in train_loader:
            model.train(); optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["signal"])
            accelerator.backward(outputs["loss"])
            accelerator.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clipping", 1.0))
            optimizer.step(); step += 1; progress.update(1)
            if accelerator.is_local_main_process and step % cfg["training"].get("log_every_steps", 10) == 0:
                progress.set_postfix(loss=f"{outputs['loss'].item():.5f}")
            if step % eval_every == 0:
                model.eval(); losses = []
                with torch.no_grad():
                    for index, valid_batch in enumerate(valid_loader):
                        losses.append(model(valid_batch["signal"])["loss"].detach())
                        if index + 1 >= cfg["training"].get("max_eval_batches", 20): break
                mean_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=accelerator.device)
                mean_loss = accelerator.gather_for_metrics(mean_loss.unsqueeze(0)).mean().item()
                if accelerator.is_local_main_process: accelerator.print(f"step={step} eval_loss={mean_loss:.6f}")
            if step % save_every == 0 and accelerator.is_local_main_process:
                unwrapped = accelerator.unwrap_model(model)
                torch.save({"model": unwrapped.state_dict(), "step": step, "config": cfg}, output_dir / f"step_{step}.pt")
            if step >= max_steps: break
    if accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save({"model": unwrapped.state_dict(), "step": step, "config": cfg}, output_dir / "last.pt")
    progress.close()


if __name__ == "__main__":
    main()
