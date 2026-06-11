#!/usr/bin/env python3
"""Train Stage2 MLM with Jiaheng's MaskedSignalLM and Accelerate.

This entrypoint keeps Jiaheng's simple TransformerEncoder MLM architecture,
while adapting data, token ids, and distributed training to the current PoreDLM
Stage2 Accelerate workflow.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import get_scheduler

from dataset import Stage2Collator, Stage2TokenJsonlDataset, Stage2TokenJsonlIterableDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage2 MaskedSignalLM with Accelerate.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(config: dict[str, Any], split: str):
    data_cfg = config["data"]
    model_cfg = config["model"]
    path_key = f"{split}_dir"
    pattern = str(data_cfg.get(f"{split}_pattern", data_cfg.get("file_pattern", "*.jsonl.gz")))
    use_streaming = bool(data_cfg.get("streaming", False))
    if use_streaming:
        return Stage2TokenJsonlIterableDataset(
            data_dir=str(data_cfg[path_key]),
            pattern=pattern,
            tokenizer_path=model_cfg.get("tokenizer_path"),
            unk_token_id=int(model_cfg.get("unk_token_id", 0)),
            vocab_size=int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
            shuffle_files=(split == "train"),
            seed=int(config.get("reproducibility", {}).get("seed", config.get("seed", 42))),
        )
    return Stage2TokenJsonlDataset(
        data_dir=str(data_cfg[path_key]),
        pattern=pattern,
        max_cache_files=int(data_cfg.get("max_cache_files", data_cfg.get("max_cache_size", 2))),
        tokenizer_path=model_cfg.get("tokenizer_path"),
        unk_token_id=int(model_cfg.get("unk_token_id", 0)),
        vocab_size=int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
    )


def build_dataloader(
    config: dict[str, Any],
    split: str,
) -> DataLoader:
    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    dataset = build_dataset(config, split)
    collator = Stage2Collator(
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
        max_length=int(model_cfg.get("max_position_embeddings", 4096)),
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(training_cfg.get("device_micro_batch_size", training_cfg.get("batch_size", 8))),
        "shuffle": (split == "train" and not isinstance(dataset, IterableDataset)),
        "num_workers": int(data_cfg.get("num_workers", 8)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": collator,
        "drop_last": (split == "train"),
    }
    if int(data_cfg.get("num_workers", 8)) > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return DataLoader(**loader_kwargs)


class MaskedSignalLM(nn.Module):
    """Jiaheng's TransformerEncoder masked-token LM, with configurable vocab size."""

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return self.lm_head(self.norm(x))


def build_masked_signal_lm(config: dict[str, Any]) -> MaskedSignalLM:
    model_cfg = config["model"]
    return MaskedSignalLM(
        max_seq_len=int(model_cfg.get("max_position_embeddings", model_cfg.get("max_seq_len", 1024))),
        d_model=int(model_cfg.get("d_model", model_cfg.get("hidden_size", 512))),
        layers=int(model_cfg.get("layers", model_cfg.get("num_hidden_layers", 8))),
        heads=int(model_cfg.get("heads", model_cfg.get("num_attention_heads", 8))),
        dropout=float(model_cfg.get("dropout", model_cfg.get("hidden_dropout_prob", 0.1))),
        vocab_size=int(model_cfg.get("vocab_size", 2056)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
    )


def mask_token_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int,
    random_token_max_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full_like(input_ids, -100)
    valid_positions = attention_mask.bool()
    maskable_positions = valid_positions & (input_ids >= random_token_min_id) & (input_ids < random_token_max_id)

    probability_matrix = torch.full(input_ids.shape, mask_probability, device=input_ids.device)
    selected = torch.bernoulli(probability_matrix).bool() & maskable_positions
    for row in range(input_ids.shape[0]):
        if not bool(selected[row].any()) and bool(maskable_positions[row].any()):
            candidates = torch.nonzero(maskable_positions[row], as_tuple=False).view(-1)
            selected[row, candidates[torch.randint(candidates.numel(), (1,), device=input_ids.device).item()]] = True

    labels[selected] = input_ids[selected]
    corrupted = input_ids.clone()

    replace_rand = torch.rand(input_ids.shape, device=input_ids.device)
    replace_with_mask = selected & (replace_rand < 0.8)
    replace_with_random = selected & (replace_rand >= 0.8) & (replace_rand < 0.9)
    corrupted[replace_with_mask] = mask_token_id

    random_token_upper_bound = min(random_token_max_id, vocab_size)
    if bool(replace_with_random.any()):
        random_tokens = torch.randint(
            random_token_min_id,
            random_token_upper_bound,
            input_ids.shape,
            dtype=torch.long,
            device=input_ids.device,
        )
        corrupted[replace_with_random] = random_tokens[replace_with_random]

    return corrupted, labels


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
    model_cfg: dict[str, Any],
    max_eval_batches: int | None,
) -> dict[str, float]:
    model.eval()
    vocab_size = int(model_cfg.get("vocab_size", 65536))
    mask_token_id = int(model_cfg.get("mask_token_id", vocab_size - 1))
    mask_probability = float(model_cfg.get("mask_probability", 0.15))
    random_token_min_id = int(model_cfg.get("random_token_min_id", 0))
    random_token_max_id = int(model_cfg.get("random_token_max_id", vocab_size))

    losses = []
    masked_counts = []
    correct_counts = []
    for batch_index, batch in enumerate(loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
        input_ids = batch.input_ids.to(accelerator.device, non_blocking=True)
        attention_mask = batch.attention_mask.to(accelerator.device, non_blocking=True)
        corrupted, labels = mask_token_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            mask_probability=mask_probability,
            random_token_min_id=random_token_min_id,
            random_token_max_id=random_token_max_id,
        )

        logits = model(corrupted, attention_mask)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100, reduction="sum")

        valid = labels != -100
        masked_count = valid.sum()
        losses.append(accelerator.gather_for_metrics(loss.detach().reshape(1)))
        masked_counts.append(accelerator.gather_for_metrics(masked_count.detach().reshape(1)))
        if bool(valid.any()):
            correct = (logits.argmax(dim=-1)[valid] == labels[valid]).sum()
        else:
            correct = torch.zeros((), dtype=torch.long, device=labels.device)
        correct_counts.append(accelerator.gather_for_metrics(correct.detach().reshape(1)))

    model.train()
    if not losses:
        return {"eval/loss": float("nan"), "eval/accuracy": float("nan"), "eval/masked_tokens": 0.0}

    total_loss = torch.cat(losses).sum().item()
    total_masked = torch.cat(masked_counts).sum().item()
    total_correct = torch.cat(correct_counts).sum().item()
    return {
        "eval/loss": total_loss / max(total_masked, 1),
        "eval/accuracy": total_correct / total_masked if total_masked else float("nan"),
        "eval/masked_tokens": total_masked,
    }


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int | str,
    epoch: int,
    val_loss: float,
    config: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    raw_model = accelerator.unwrap_model(model)
    if isinstance(step, str):
        path = output_dir / f"{step}.pt"
    else:
        path = output_dir / f"step{step}.pt"
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
            "vocab_size": int(config["model"].get("vocab_size", 2056)),
            "pad_token_id": int(config["model"].get("pad_token_id", 0)),
            "model_class": "MaskedSignalLM",
        },
        path,
    )


def infer_scheduler_steps(
    train_loader: DataLoader,
    epochs: int,
    grad_accum: int,
    training_cfg: dict[str, Any],
) -> int:
    if training_cfg.get("scheduler_num_training_steps") is not None:
        return int(training_cfg["scheduler_num_training_steps"])
    if training_cfg.get("max_steps") is not None:
        return int(training_cfg["max_steps"])
    try:
        return max(1, math.ceil(len(train_loader) / max(1, grad_accum)) * epochs)
    except TypeError:
        raise ValueError(
            "Set training.scheduler_num_training_steps when data.streaming=true, "
            "because an IterableDataset has no cheap length."
        )


def print_startup_summary(
    accelerator: Accelerator,
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    model: torch.nn.Module,
) -> None:
    if not accelerator.is_main_process:
        return
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset if val_loader is not None else None
    train_files = getattr(train_ds, "files", [])
    val_files = getattr(val_ds, "files", []) if val_ds is not None else []
    param_count = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("\n" + "=" * 80)
    print("Starting Stage2 MaskedSignalLM Training: v6_jiaheng")
    print("=" * 80)
    print(
        pformat(
            {
                "distributed": {
                    "num_processes": accelerator.num_processes,
                    "process_index": accelerator.process_index,
                    "local_process_index": accelerator.local_process_index,
                    "device": str(accelerator.device),
                    "mixed_precision": accelerator.mixed_precision,
                },
                "data": {
                    "train_dir": config["data"].get("train_dir"),
                    "valid_dir": config["data"].get("valid_dir"),
                    "streaming": config["data"].get("streaming", False),
                    "train_files": len(train_files),
                    "valid_files": len(val_files),
                },
                "model": {
                    "vocab_size": config["model"].get("vocab_size"),
                    "mask_token_id": config["model"].get("mask_token_id"),
                    "pad_token_id": config["model"].get("pad_token_id"),
                    "random_token_min_id": config["model"].get("random_token_min_id"),
                    "random_token_max_id": config["model"].get("random_token_max_id"),
                    "d_model": config["model"].get("d_model", config["model"].get("hidden_size")),
                    "layers": config["model"].get("layers", config["model"].get("num_hidden_layers")),
                    "heads": config["model"].get("heads", config["model"].get("num_attention_heads")),
                    "max_position_embeddings": config["model"].get("max_position_embeddings"),
                    "mask_probability": config["model"].get("mask_probability"),
                    "total_parameters": f"{param_count:,}",
                    "trainable_parameters": f"{trainable_count:,}",
                },
                "training": config["training"],
            },
            width=120,
            sort_dicts=False,
        )
    )
    print("=" * 80 + "\n", flush=True)


def train(config: dict[str, Any]) -> None:
    seed = int(config.get("reproducibility", {}).get("seed", config.get("seed", 42)))
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    training_cfg = config["training"]
    model_cfg = config["model"]
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(training_cfg.get("mixed_precision", "no")),
        log_with="wandb" if config.get("wandb", {}).get("use_wandb", False) else None,
        project_dir=str(training_cfg.get("log_dir", "log")),
    )
    output_dir = Path(str(training_cfg.get("output_dir", "models")))
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader = build_dataloader(config, "train")
    val_loader = None
    if config["data"].get("valid_dir"):
        val_loader = build_dataloader(config, "valid")

    model = build_masked_signal_lm(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", training_cfg.get("lr", 5e-5))),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", training_cfg.get("grad_accum", 1)))
    epochs = int(training_cfg.get("num_train_epochs", training_cfg.get("epochs", 1)))
    scheduler_steps = infer_scheduler_steps(train_loader, epochs, grad_accum, training_cfg)
    scheduler = get_scheduler(
        name=str(training_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        num_training_steps=scheduler_steps,
    )

    print_startup_summary(
        accelerator=accelerator,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
    )

    if config.get("wandb", {}).get("use_wandb", False):
        accelerator.init_trackers(
            project_name=str(config["wandb"].get("project", "stage2_BERT_Encoder_experiments")),
            config=config,
            init_kwargs={"wandb": {"name": config["wandb"].get("name")}},
        )

    if val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            scheduler,
        )

    vocab_size = int(model_cfg.get("vocab_size", 65536))
    mask_token_id = int(model_cfg.get("mask_token_id", vocab_size - 1))
    mask_probability = float(model_cfg.get("mask_probability", 0.15))
    random_token_min_id = int(model_cfg.get("random_token_min_id", 0))
    random_token_max_id = int(model_cfg.get("random_token_max_id", vocab_size))
    log_every = int(training_cfg.get("log_every_steps", 50))
    eval_every = int(training_cfg.get("eval_every_steps", 500))
    save_every = int(training_cfg.get("save_every_steps", 1000))
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    grad_clip = float(training_cfg.get("gradient_clipping", 1.0))

    global_step = 0
    best_val = math.inf
    optimizer.zero_grad(set_to_none=True)
    model.train()
    progress = tqdm(total=scheduler_steps, disable=not accelerator.is_local_main_process)
    for epoch in range(epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                input_ids = batch.input_ids.to(accelerator.device, non_blocking=True)
                attention_mask = batch.attention_mask.to(accelerator.device, non_blocking=True)
                corrupted, labels = mask_token_ids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    mask_probability=mask_probability,
                    random_token_min_id=random_token_min_id,
                    random_token_max_id=random_token_max_id,
                )
                logits = model(corrupted, attention_mask)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % log_every == 0:
                    loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
                    logs = {
                        "train/loss": loss_value,
                        "train/loss_log10": math.log10(loss_value + 1e-12),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step,
                    }
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        print(f"[TRAIN] {logs}", flush=True)

                if save_every > 0 and global_step % save_every == 0:
                    save_checkpoint(accelerator, output_dir, model, optimizer, scheduler, global_step, epoch, best_val, config)

                if val_loader is not None and eval_every > 0 and global_step % eval_every == 0:
                    eval_logs = evaluate(
                        accelerator=accelerator,
                        model=model,
                        loader=val_loader,
                        model_cfg=model_cfg,
                        max_eval_batches=max_eval_batches,
                    )
                    eval_logs["epoch"] = epoch
                    eval_logs["step"] = global_step
                    accelerator.log(eval_logs, step=global_step)
                    if accelerator.is_main_process:
                        print(f"[VAL] {eval_logs}", flush=True)

                    eval_loss = eval_logs["eval/loss"]
                    if eval_loss == eval_loss:
                        save_checkpoint(accelerator, output_dir, model, optimizer, scheduler, "last", epoch, eval_loss, config)
                        if eval_loss < best_val:
                            best_val = eval_loss
                            save_checkpoint(accelerator, output_dir, model, optimizer, scheduler, "best", epoch, eval_loss, config)

    final_loss = math.inf
    final_acc = float("nan")
    if val_loader is not None:
        final_logs = evaluate(
            accelerator=accelerator,
            model=model,
            loader=val_loader,
            model_cfg=model_cfg,
            max_eval_batches=max_eval_batches,
        )
        final_loss = final_logs["eval/loss"]
        final_acc = final_logs["eval/accuracy"]
    if accelerator.is_main_process:
        print(f"[FINAL] step={global_step} val_loss={final_loss:.6f} val_acc={final_acc:.4f}", flush=True)
    save_checkpoint(accelerator, output_dir, model, optimizer, scheduler, "last", epochs, final_loss, config)
    if final_loss < best_val:
        save_checkpoint(accelerator, output_dir, model, optimizer, scheduler, "best", epochs, final_loss, config)
    progress.close()
    accelerator.end_training()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
