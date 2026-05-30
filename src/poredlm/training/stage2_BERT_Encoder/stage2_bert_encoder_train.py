"""Train a BERT encoder for Stage 2 representation learning."""

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
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from bert_encoder_model import build_bert_mlm
from dataset import Stage2Collator, Stage2TokenJsonlDataset


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_number(value: int) -> str:
    """Format large integer counts for logs."""

    return f"{value:,}"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def print_startup_summary(
    accelerator: Accelerator,
    config: dict[str, Any],
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader | None,
    seed: int,
) -> None:
    """Print model, data, and runtime information before training starts."""

    if not accelerator.is_main_process:
        return

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset if valid_loader is not None else None
    total_params, trainable_params = count_parameters(model)

    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    device_micro_batch_size = int(training_cfg.get("device_micro_batch_size", 8))
    effective_global_batch_size = (
        device_micro_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    train_files = getattr(train_dataset, "files", [])
    valid_files = getattr(valid_dataset, "files", []) if valid_dataset is not None else []
    train_line_counts = getattr(train_dataset, "file_line_counts", [])
    valid_line_counts = getattr(valid_dataset, "file_line_counts", []) if valid_dataset is not None else []

    print("\n" + "=" * 80)
    print("Starting Stage 2 BERT Encoder Training")
    print("=" * 80)
    print(
        pformat(
            {
                "seed": seed,
                "distributed": {
                    "num_processes": accelerator.num_processes,
                    "process_index": accelerator.process_index,
                    "local_process_index": accelerator.local_process_index,
                    "device": str(accelerator.device),
                    "mixed_precision": accelerator.mixed_precision,
                },
                "data": {
                    "train_dir": data_cfg.get("train_dir"),
                    "valid_dir": data_cfg.get("valid_dir") or None,
                    "file_pattern": data_cfg.get("file_pattern", "*.jsonl.gz"),
                    "train_files": len(train_files),
                    "valid_files": len(valid_files),
                    "train_samples": len(train_dataset),
                    "valid_samples": len(valid_dataset) if valid_dataset is not None else 0,
                    "train_lines_per_file_head": train_line_counts[:5],
                    "valid_lines_per_file_head": valid_line_counts[:5],
                    "num_workers": data_cfg.get("num_workers", 8),
                    "prefetch_factor": data_cfg.get("prefetch_factor", 2),
                },
                "model": {
                    "type": model.__class__.__name__,
                    "vocab_size": model_cfg.get("vocab_size"),
                    "tokenizer_path": model_cfg.get("tokenizer_path"),
                    "mask_token_id": model_cfg.get("mask_token_id"),
                    "pad_token_id": model_cfg.get("pad_token_id"),
                    "unk_token_id": model_cfg.get("unk_token_id"),
                    "random_token_min_id": model_cfg.get("random_token_min_id"),
                    "random_token_max_id": model_cfg.get("random_token_max_id"),
                    "hidden_size": model_cfg.get("hidden_size"),
                    "num_hidden_layers": model_cfg.get("num_hidden_layers"),
                    "num_attention_heads": model_cfg.get("num_attention_heads"),
                    "intermediate_size": model_cfg.get("intermediate_size"),
                    "max_position_embeddings": model_cfg.get("max_position_embeddings"),
                    "mask_probability": model_cfg.get("mask_probability"),
                    "total_parameters": format_number(total_params),
                    "trainable_parameters": format_number(trainable_params),
                },
                "training": {
                    "max_steps": training_cfg.get("max_steps"),
                    "learning_rate": training_cfg.get("learning_rate"),
                    "weight_decay": training_cfg.get("weight_decay"),
                    "warmup_steps": training_cfg.get("warmup_steps"),
                    "lr_scheduler_type": training_cfg.get("lr_scheduler_type"),
                    "device_micro_batch_size": device_micro_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_global_batch_size": effective_global_batch_size,
                    "gradient_clipping": training_cfg.get("gradient_clipping"),
                    "output_dir": training_cfg.get("output_dir"),
                    "log_every_steps": training_cfg.get("log_every_steps"),
                    "eval_every_steps": training_cfg.get("eval_every_steps"),
                    "max_eval_batches": training_cfg.get("max_eval_batches"),
                    "save_every_steps": training_cfg.get("save_every_steps"),
                },
            },
            width=120,
            sort_dicts=False,
        )
    )
    print("-" * 80)
    print("Model architecture:")
    print(model)
    print("=" * 80 + "\n")


def mask_token_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int = 0,
    random_token_max_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT MLM masking to VQ token ids."""

    labels = input_ids.clone()
    valid_positions = attention_mask.bool()
    probability_matrix = torch.full(labels.shape, mask_probability, device=input_ids.device)
    masked_indices = torch.bernoulli(probability_matrix).bool() & valid_positions

    if not masked_indices.any():
        first_valid = valid_positions.float().argmax(dim=1)
        masked_indices[torch.arange(input_ids.shape[0], device=input_ids.device), first_valid] = True

    labels[~masked_indices] = -100

    corrupted = input_ids.clone()
    replace_with_mask = torch.bernoulli(
        torch.full(labels.shape, 0.8, device=input_ids.device)
    ).bool() & masked_indices
    corrupted[replace_with_mask] = mask_token_id

    replace_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
        & masked_indices
        & ~replace_with_mask
    )
    random_token_upper_bound = min(random_token_max_id or vocab_size, vocab_size)
    random_tokens = torch.randint(
        random_token_min_id,
        random_token_upper_bound,
        labels.shape,
        dtype=torch.long,
        device=input_ids.device,
    )
    corrupted[replace_with_random] = random_tokens[replace_with_random]

    return corrupted, labels


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    path_key = f"{split}_dir"
    dataset = Stage2TokenJsonlDataset(
        data_dir=data_cfg[path_key],
        pattern=str(data_cfg.get(f"{split}_pattern", data_cfg.get("file_pattern", "*.jsonl.gz"))),
        max_cache_files=int(data_cfg.get("max_cache_files", 2)),
        tokenizer_path=model_cfg.get("tokenizer_path"),
        unk_token_id=int(model_cfg.get("unk_token_id", 0)),
        vocab_size=int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
    )
    collator = Stage2Collator(
        pad_token_id=int(config.get("model", {}).get("pad_token_id", 0)),
        max_length=int(config.get("model", {}).get("max_position_embeddings", 4096)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("device_micro_batch_size", 8)),
        shuffle=(split == "train"),
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
        drop_last=(split == "train"),
    )

def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: str,
    step: int | str,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    save_dir = Path(output_dir) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(save_dir)
    else:
        torch.save(unwrapped.state_dict(), save_dir / "pytorch_model.bin")


def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    valid_loader: DataLoader,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int = 0,
    random_token_max_id: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """Run MLM evaluation on the validation split."""

    model.eval()
    losses = []
    top1_correct = []
    top5_correct = []
    top10_correct = []
    top50_correct = []
    masked_counts = []

    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break

            input_ids = batch.input_ids.to(accelerator.device)
            attention_mask = batch.attention_mask.to(accelerator.device)
            corrupted, labels = mask_token_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                mask_probability=mask_probability,
                random_token_min_id=random_token_min_id,
                random_token_max_id=random_token_max_id,
            )
            outputs = model(
                input_ids=corrupted,
                attention_mask=attention_mask,
                labels=labels,
            )
            losses.append(accelerator.gather_for_metrics(outputs.loss.detach()))

            masked_positions = labels != -100
            masked_count = masked_positions.sum()
            masked_counts.append(accelerator.gather_for_metrics(masked_count.detach().reshape(1)))

            if masked_count.item() > 0:
                masked_logits = outputs.logits[masked_positions]
                masked_labels = labels[masked_positions]
                topk = torch.topk(masked_logits, k=min(50, masked_logits.shape[-1]), dim=-1).indices
                top1 = (topk[:, :1] == masked_labels[:, None]).any(dim=-1).sum()
                top5 = (topk[:, : min(5, topk.shape[-1])] == masked_labels[:, None]).any(dim=-1).sum()
                top10 = (topk[:, : min(10, topk.shape[-1])] == masked_labels[:, None]).any(dim=-1).sum()
                top50 = (topk == masked_labels[:, None]).any(dim=-1).sum()
            else:
                top1 = torch.zeros((), dtype=torch.long, device=labels.device)
                top5 = torch.zeros((), dtype=torch.long, device=labels.device)
                top10 = torch.zeros((), dtype=torch.long, device=labels.device)
                top50 = torch.zeros((), dtype=torch.long, device=labels.device)

            top1_correct.append(accelerator.gather_for_metrics(top1.detach().reshape(1)))
            top5_correct.append(accelerator.gather_for_metrics(top5.detach().reshape(1)))
            top10_correct.append(accelerator.gather_for_metrics(top10.detach().reshape(1)))
            top50_correct.append(accelerator.gather_for_metrics(top50.detach().reshape(1)))

    model.train()

    if not losses:
        return {
            "eval/loss": float("nan"),
            "eval/perplexity": float("nan"),
            "eval/top1_accuracy": float("nan"),
            "eval/top5_accuracy": float("nan"),
            "eval/top10_accuracy": float("nan"),
            "eval/top50_accuracy": float("nan"),
        }

    loss = torch.cat([loss.reshape(-1) for loss in losses]).mean().item()
    perplexity = math.exp(loss) if loss < 50 else float("inf")
    total_masked = torch.cat(masked_counts).sum().item()
    top1 = torch.cat(top1_correct).sum().item()
    top5 = torch.cat(top5_correct).sum().item()
    top10 = torch.cat(top10_correct).sum().item()
    top50 = torch.cat(top50_correct).sum().item()

    return {
        "eval/loss": loss,
        "eval/perplexity": perplexity,
        "eval/top1_accuracy": top1 / total_masked if total_masked else float("nan"),
        "eval/top5_accuracy": top5 / total_masked if total_masked else float("nan"),
        "eval/top10_accuracy": top10 / total_masked if total_masked else float("nan"),
        "eval/top50_accuracy": top50 / total_masked if total_masked else float("nan"),
        "eval/masked_tokens": total_masked,
    }


def train(config: dict[str, Any]) -> None:
    seed = int(config.get("reproducibility", {}).get("seed", config.get("seed", 42)))
    seed_everything(seed)

    training_cfg = config["training"]
    model_cfg = config["model"]
    mask_probability = float(model_cfg.get("mask_probability", 0.15))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(training_cfg.get("mixed_precision", "no")),
        log_with="wandb" if config.get("wandb", {}).get("use_wandb", False) else None,
        project_dir=str(training_cfg.get("log_dir", "log")),
    )

    train_loader = build_dataloader(config, "train")
    valid_loader = build_dataloader(config, "valid") if config["data"].get("valid_dir") else None

    model = build_bert_mlm(config)
    print_startup_summary(
        accelerator=accelerator,
        config=config,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seed=seed,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    max_steps = int(training_cfg.get("max_steps", 100000))
    scheduler = get_scheduler(
        name=str(training_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        num_training_steps=max_steps,
    )

    if config.get("wandb", {}).get("use_wandb", False):
        accelerator.init_trackers(
            project_name=str(config["wandb"].get("project", "poredlm-stage2-bert")),
            config=config,
            init_kwargs={"wandb": {"name": config["wandb"].get("name")}},
        )

    if valid_loader is not None:
        model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            valid_loader,
            scheduler,
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            scheduler,
        )

    global_step = 0
    output_dir = str(training_cfg.get("output_dir", "outputs/stage2_BERT_Encoder"))
    save_every = int(training_cfg.get("save_every_steps", 1000))
    log_every = int(training_cfg.get("log_every_steps", 10))
    eval_every = int(training_cfg.get("eval_every_steps", 0))
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    vocab_size = int(model_cfg.get("vocab_size", 65537))
    mask_token_id = int(model_cfg.get("mask_token_id", vocab_size - 1))
    random_token_min_id = int(model_cfg.get("random_token_min_id", 0))
    random_token_max_id = int(model_cfg.get("random_token_max_id", vocab_size))
    best_eval_loss = float("inf")

    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    model.train()

    while global_step < max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                input_ids = batch.input_ids.to(accelerator.device)
                attention_mask = batch.attention_mask.to(accelerator.device)
                corrupted, labels = mask_token_ids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    mask_probability=mask_probability,
                    random_token_min_id=random_token_min_id,
                    random_token_max_id=random_token_max_id,
                )
                outputs = model(
                    input_ids=corrupted,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(training_cfg.get("gradient_clipping", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % log_every == 0:
                    loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
                    logs = {
                        "train/loss": loss_value,
                        "train/loss_log10": math.log10(loss_value + 1e-12),
                        "train/lr": scheduler.get_last_lr()[0],
                        "step": global_step,
                    }
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        print(logs)

                if global_step % save_every == 0:
                    save_checkpoint(accelerator, model, output_dir, global_step)

                if valid_loader is not None and eval_every > 0 and global_step % eval_every == 0:
                    eval_logs = evaluate(
                        accelerator=accelerator,
                        model=model,
                        valid_loader=valid_loader,
                        vocab_size=vocab_size,
                        mask_token_id=mask_token_id,
                        mask_probability=mask_probability,
                        random_token_min_id=random_token_min_id,
                        random_token_max_id=random_token_max_id,
                        max_eval_batches=max_eval_batches,
                    )
                    eval_logs["step"] = global_step
                    accelerator.log(eval_logs, step=global_step)
                    if accelerator.is_main_process:
                        print(eval_logs)

                    eval_loss = eval_logs["eval/loss"]
                    if eval_loss == eval_loss and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(accelerator, model, output_dir, "best")

                if global_step >= max_steps:
                    break

    save_checkpoint(accelerator, model, output_dir, global_step)
    progress.close()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 BERT Encoder")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config)


if __name__ == "__main__":
    main()
