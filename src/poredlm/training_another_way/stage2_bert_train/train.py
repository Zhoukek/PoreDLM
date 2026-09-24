"""Train continuous BERT on frozen CNN feature memmaps."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

from continuous_feature_dataset import ContinuousFeatureDataset
from modeling_continuous_bert import ContinuousBert, ContinuousBertConfig


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_loader(cfg: dict, split: str, train: bool):
    data = cfg["data"][split]
    dataset = ContinuousFeatureDataset(data["directory"], feature_dim=cfg["model"]["feature_dim"],
                                       max_length=cfg["model"]["max_position_embeddings"],
                                       dtype=data.get("dtype", "float16"), shuffle_files=train,
                                       repeat=train, seed=cfg.get("seed", 42))
    workers = int(data.get("num_workers", 0))
    batch_size = cfg["training"].get("device_micro_batch_size", cfg["training"].get("device_train_microbatch_size", 8))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       num_workers=workers, pin_memory=data.get("pin_memory", True),
                                       persistent_workers=data.get("persistent_workers", False) and workers > 0,
                                       prefetch_factor=data.get("prefetch_factor", 2) if workers > 0 else None)


def make_mask(attention_mask: torch.Tensor, probability: float) -> torch.Tensor:
    mask = torch.rand(attention_mask.shape, device=attention_mask.device) < probability
    mask = mask & attention_mask.bool()
    # Keep at least one valid prediction position per non-empty sample.
    for row in range(mask.shape[0]):
        valid = torch.nonzero(attention_mask[row].bool(), as_tuple=False).flatten()
        if valid.numel() and not mask[row].any():
            mask[row, valid[torch.randint(valid.numel(), (1,), device=mask.device)]] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); seed_everything(int(cfg.get("seed", 42)))
    accelerator = Accelerator(mixed_precision=cfg["training"].get("mixed_precision", "no"))
    model = ContinuousBert(ContinuousBertConfig(**cfg["model"]))
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 0.01))
    train_loader = build_loader(cfg, "train", True); valid_loader = build_loader(cfg, "valid", False)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    output_dir = Path(cfg["training"]["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["training"]["max_train_steps"]); step = 0
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    while step < max_steps:
        for batch in train_loader:
            model.train(); optimizer.zero_grad(set_to_none=True)
            embeddings = batch["embeddings"]
            attention = batch["attention_mask"]
            positions = make_mask(attention, cfg["model"].get("mask_probability", 0.15))
            outputs = model(embeddings, attention, embeddings.detach(), positions)
            accelerator.backward(outputs.loss)
            accelerator.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clipping", 1.0))
            optimizer.step(); step += 1; progress.update(1)
            if accelerator.is_local_main_process and step % cfg["training"].get("log_every_steps", 10) == 0:
                progress.set_postfix(loss=f"{outputs.loss.item():.5f}")
            if step % cfg["training"].get("eval_every_steps", 1000) == 0:
                model.eval(); values = []
                with torch.no_grad():
                    for index, valid_batch in enumerate(valid_loader):
                        valid_mask = make_mask(valid_batch["attention_mask"], cfg["model"].get("mask_probability", 0.15))
                        result = model(valid_batch["embeddings"], valid_batch["attention_mask"], valid_batch["embeddings"], valid_mask)
                        if result.loss is not None: values.append(result.loss)
                        if index + 1 >= cfg["training"].get("max_eval_batches", 20): break
                loss = torch.stack(values).mean() if values else torch.tensor(0.0, device=accelerator.device)
                loss = accelerator.gather_for_metrics(loss.unsqueeze(0)).mean().item()
                if accelerator.is_local_main_process: accelerator.print(f"step={step} eval_loss={loss:.6f}")
            if step % cfg["training"].get("save_every_steps", 5000) == 0 and accelerator.is_local_main_process:
                torch.save({"model": accelerator.unwrap_model(model).state_dict(), "step": step, "config": cfg}, output_dir / f"step_{step}.pt")
            if step >= max_steps: break
    if accelerator.is_local_main_process:
        torch.save({"model": accelerator.unwrap_model(model).state_dict(), "step": step, "config": cfg}, output_dir / "last.pt")
    progress.close()


if __name__ == "__main__": main()
