"""Train Stage 2 BERT MLM with epoch-driven symmetric 50% masking.

V8 uses the plain BERT MLM model from ``bert_encoder_model.py`` and changes the
masking policy to:

1. sample 50% of valid token positions per sequence;
2. replace every sampled token with ``[MASK]``;
3. build a complementary masked view in the same batch, so positions not masked
   in the first view are masked in the second view.

No BERT 80/10/10 keep/random replacement is used in this version.
"""

from __future__ import annotations

import argparse
import inspect
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
from dataset import Stage2Collator, Stage2TokenJsonlDataset, Stage2TokenJsonlIterableDataset


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


def build_accelerator(config: dict[str, Any]) -> Accelerator:
    """Build Accelerator with iterable datasets read independently on each rank."""

    training_cfg = config["training"]
    accelerator_kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "mixed_precision": str(training_cfg.get("mixed_precision", "no")),
        "log_with": "wandb" if config.get("wandb", {}).get("use_wandb", False) else None,
        "project_dir": str(training_cfg.get("log_dir", "log")),
    }
    if bool(config.get("data", {}).get("streaming", False)):
        accelerator_params = inspect.signature(Accelerator).parameters
        if "dataloader_config" in accelerator_params:
            from accelerate import DataLoaderConfiguration

            accelerator_kwargs["dataloader_config"] = DataLoaderConfiguration(dispatch_batches=False)
        elif "dispatch_batches" in accelerator_params:
            accelerator_kwargs["dispatch_batches"] = False
    return Accelerator(**accelerator_kwargs)


def safe_len(value: Any) -> int | str:
    """Return len(value), or a descriptive string for iterable-only datasets."""

    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return "streaming"


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
    symmetric_effective_views = effective_global_batch_size * 2

    train_files = getattr(train_dataset, "files", [])
    valid_files = getattr(valid_dataset, "files", []) if valid_dataset is not None else []
    train_line_counts = getattr(train_dataset, "file_line_counts", [])
    valid_line_counts = getattr(valid_dataset, "file_line_counts", []) if valid_dataset is not None else []

    print("\n" + "=" * 80)
    print("Starting Stage 2 BERT Encoder V8 Training")
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
                    "streaming": data_cfg.get("streaming", False),
                    "train_files": len(train_files),
                    "valid_files": len(valid_files),
                    "train_samples": safe_len(train_dataset),
                    "valid_samples": safe_len(valid_dataset),
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
                    "hidden_size": model_cfg.get("hidden_size"),
                    "num_hidden_layers": model_cfg.get("num_hidden_layers"),
                    "num_attention_heads": model_cfg.get("num_attention_heads"),
                    "intermediate_size": model_cfg.get("intermediate_size"),
                    "max_position_embeddings": model_cfg.get("max_position_embeddings"),
                    "mask_probability": model_cfg.get("mask_probability", 0.5),
                    "mask_policy": "symmetric_complement_all_mask",
                    "total_parameters": format_number(total_params),
                    "trainable_parameters": format_number(trainable_params),
                },
                "training": {
                    "num_train_epochs": training_cfg.get(
                        "num_train_epochs",
                        training_cfg.get("epochs", 1),
                    ),
                    "steps_per_epoch": training_cfg.get("steps_per_epoch"),
                    "scheduler_num_training_steps": training_cfg.get("scheduler_num_training_steps"),
                    "learning_rate": training_cfg.get("learning_rate"),
                    "weight_decay": training_cfg.get("weight_decay"),
                    "warmup_steps": training_cfg.get("warmup_steps"),
                    "lr_scheduler_type": training_cfg.get("lr_scheduler_type"),
                    "device_micro_batch_size": device_micro_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_global_batch_size": effective_global_batch_size,
                    "symmetric_effective_views": symmetric_effective_views,
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


def build_primary_mask_indices(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_probability: float,
) -> torch.Tensor:
    """Select a fixed fraction of valid positions per sample."""

    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    valid_positions = attention_mask.bool()

    for batch_index in range(input_ids.shape[0]):
        valid_token_indices = torch.nonzero(valid_positions[batch_index], as_tuple=False).flatten()
        valid_count = int(valid_token_indices.numel())
        if valid_count == 0:
            continue

        target_mask_count = int(round(valid_count * float(mask_probability)))
        if mask_probability > 0.0:
            target_mask_count = max(1, target_mask_count)
        if valid_count > 1:
            target_mask_count = min(target_mask_count, valid_count - 1)
        else:
            target_mask_count = min(target_mask_count, valid_count)

        permutation = torch.randperm(valid_count, device=input_ids.device)
        selected = valid_token_indices[permutation[:target_mask_count]]
        masked_indices[batch_index, selected] = True

    if not masked_indices.any():
        valid_positions_flat = torch.nonzero(valid_positions, as_tuple=False)
        if int(valid_positions_flat.numel()) > 0:
            first_batch = int(valid_positions_flat[0, 0].item())
            first_position = int(valid_positions_flat[0, 1].item())
            masked_indices[first_batch, first_position] = True

    return masked_indices


def build_symmetric_masked_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    mask_probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Create original and complementary all-mask MLM views in one batch."""

    primary_mask = build_primary_mask_indices(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_probability=mask_probability,
    )
    valid_positions = attention_mask.bool()
    complement_mask = valid_positions & ~primary_mask

    primary_corrupted = input_ids.clone()
    primary_corrupted[primary_mask] = mask_token_id
    primary_labels = input_ids.clone()
    primary_labels[~primary_mask] = -100

    complement_corrupted = input_ids.clone()
    complement_corrupted[complement_mask] = mask_token_id
    complement_labels = input_ids.clone()
    complement_labels[~complement_mask] = -100

    corrupted = torch.cat([primary_corrupted, complement_corrupted], dim=0)
    labels = torch.cat([primary_labels, complement_labels], dim=0)
    doubled_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
    stats = {
        "primary_masked": primary_mask.sum().detach().reshape(1),
        "complement_masked": complement_mask.sum().detach().reshape(1),
        "valid_tokens": valid_positions.sum().detach().reshape(1),
    }
    return corrupted, doubled_attention_mask, labels, stats


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    path_key = f"{split}_dir"
    pattern = str(data_cfg.get(f"{split}_pattern", data_cfg.get("file_pattern", "*.jsonl.gz")))
    use_streaming = bool(data_cfg.get("streaming", False))
    if use_streaming:
        dataset = Stage2TokenJsonlIterableDataset(
            data_dir=data_cfg[path_key],
            pattern=pattern,
            tokenizer_path=model_cfg.get("tokenizer_path"),
            unk_token_id=int(model_cfg.get("unk_token_id", 0)),
            vocab_size=int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
            shuffle_files=(split == "train"),
            seed=int(config.get("reproducibility", {}).get("seed", config.get("seed", 42))),
        )
    else:
        dataset = Stage2TokenJsonlDataset(
            data_dir=data_cfg[path_key],
            pattern=pattern,
            max_cache_files=int(data_cfg.get("max_cache_files", data_cfg.get("max_cache_size", 2))),
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
        shuffle=(split == "train" and not use_streaming),
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
        drop_last=(split == "train"),
    )


def get_batch_tensor(batch: Any, name: str) -> torch.Tensor:
    """Read tensors from either the local Stage2Batch object or a plain mapping."""

    if isinstance(batch, dict):
        return batch[name]
    return getattr(batch, name)


def infer_update_steps_per_epoch(
    train_loader: DataLoader,
    gradient_accumulation_steps: int,
) -> int | None:
    """Infer optimizer update steps per epoch when the loader exposes length."""

    try:
        batches_per_epoch = len(train_loader)
    except TypeError:
        return None

    if batches_per_epoch <= 0:
        return None
    return math.ceil(batches_per_epoch / max(1, int(gradient_accumulation_steps)))


def resolve_epoch_training_steps(
    training_cfg: dict[str, Any],
    train_loader: DataLoader,
) -> tuple[int, int | None, int]:
    """Resolve epoch count, optional progress steps, and scheduler total steps."""

    num_train_epochs = int(training_cfg.get("num_train_epochs", training_cfg.get("epochs", 1)))
    if num_train_epochs <= 0:
        raise ValueError("training.num_train_epochs/training.epochs must be > 0.")

    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    inferred_steps_per_epoch = infer_update_steps_per_epoch(train_loader, gradient_accumulation_steps)
    configured_steps_per_epoch = training_cfg.get("steps_per_epoch")
    if configured_steps_per_epoch is not None:
        steps_per_epoch = int(configured_steps_per_epoch)
        if steps_per_epoch <= 0:
            raise ValueError("training.steps_per_epoch must be > 0 when provided.")
    else:
        steps_per_epoch = inferred_steps_per_epoch

    if steps_per_epoch is not None:
        total_training_steps = steps_per_epoch * num_train_epochs
    else:
        total_training_steps = int(
            training_cfg.get(
                "scheduler_num_training_steps",
                training_cfg.get("max_steps", 100000),
            )
        )
        if total_training_steps <= 0:
            raise ValueError(
                "Could not infer scheduler training steps for an iterable dataloader. "
                "Set training.steps_per_epoch or training.scheduler_num_training_steps."
            )

    return num_train_epochs, steps_per_epoch, total_training_steps


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
    mask_token_id: int,
    mask_probability: float,
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """Run symmetric-mask MLM evaluation on the validation split."""

    model.eval()
    losses = []
    top1_correct = []
    top5_correct = []
    top10_correct = []
    top50_correct = []
    masked_counts = []
    primary_masked_counts = []
    complement_masked_counts = []

    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break

            input_ids = get_batch_tensor(batch, "input_ids").to(accelerator.device)
            attention_mask = get_batch_tensor(batch, "attention_mask").to(accelerator.device)
            corrupted, doubled_attention_mask, labels, mask_stats = build_symmetric_masked_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_token_id=mask_token_id,
                mask_probability=mask_probability,
            )
            outputs = model(
                input_ids=corrupted,
                attention_mask=doubled_attention_mask,
                labels=labels,
            )
            losses.append(accelerator.gather_for_metrics(outputs.loss.detach()))

            masked_positions = labels != -100
            masked_count = masked_positions.sum()
            masked_counts.append(accelerator.gather_for_metrics(masked_count.detach().reshape(1)))
            primary_masked_counts.append(accelerator.gather_for_metrics(mask_stats["primary_masked"]))
            complement_masked_counts.append(accelerator.gather_for_metrics(mask_stats["complement_masked"]))

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
    primary_masked = torch.cat(primary_masked_counts).sum().item()
    complement_masked = torch.cat(complement_masked_counts).sum().item()

    return {
        "eval/loss": loss,
        "eval/perplexity": perplexity,
        "eval/top1_accuracy": top1 / total_masked if total_masked else float("nan"),
        "eval/top5_accuracy": top5 / total_masked if total_masked else float("nan"),
        "eval/top10_accuracy": top10 / total_masked if total_masked else float("nan"),
        "eval/top50_accuracy": top50 / total_masked if total_masked else float("nan"),
        "eval/masked_tokens": total_masked,
        "eval/primary_masked_tokens": primary_masked,
        "eval/complement_masked_tokens": complement_masked,
    }


def train(config: dict[str, Any]) -> None:
    seed = int(config.get("reproducibility", {}).get("seed", config.get("seed", 42)))
    seed_everything(seed)

    training_cfg = config["training"]
    model_cfg = config["model"]
    mask_probability = float(model_cfg.get("mask_probability", 0.5))

    accelerator = build_accelerator(config)

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

    num_train_epochs, steps_per_epoch, total_training_steps = resolve_epoch_training_steps(
        training_cfg,
        train_loader,
    )
    scheduler = get_scheduler(
        name=str(training_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        num_training_steps=total_training_steps,
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
    mask_token_id = int(model_cfg.get("mask_token_id", int(model_cfg.get("vocab_size", 65537)) - 1))
    best_eval_loss = float("inf")

    progress_total = total_training_steps if steps_per_epoch is not None else None
    progress = tqdm(total=progress_total, disable=not accelerator.is_local_main_process)
    model.train()

    for epoch in range(num_train_epochs):
        epoch_index = epoch + 1
        if accelerator.is_main_process:
            print(f"Starting epoch {epoch_index}/{num_train_epochs}", flush=True)

        for batch in train_loader:
            with accelerator.accumulate(model):
                input_ids = get_batch_tensor(batch, "input_ids").to(accelerator.device)
                attention_mask = get_batch_tensor(batch, "attention_mask").to(accelerator.device)
                corrupted, doubled_attention_mask, labels, mask_stats = build_symmetric_masked_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mask_token_id=mask_token_id,
                    mask_probability=mask_probability,
                )
                outputs = model(
                    input_ids=corrupted,
                    attention_mask=doubled_attention_mask,
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
                    masked_positions = labels != -100
                    masked_count = (
                        accelerator.gather_for_metrics(masked_positions.sum().detach().reshape(1))
                        .sum()
                        .item()
                    )
                    primary_masked = (
                        accelerator.gather_for_metrics(mask_stats["primary_masked"]).sum().item()
                    )
                    complement_masked = (
                        accelerator.gather_for_metrics(mask_stats["complement_masked"]).sum().item()
                    )
                    valid_count = (
                        accelerator.gather_for_metrics(mask_stats["valid_tokens"]).sum().item()
                    )
                    logs = {
                        "train/loss": loss_value,
                        "train/loss_log10": math.log10(loss_value + 1e-12),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/masked_tokens": masked_count,
                        "train/primary_masked_tokens": primary_masked,
                        "train/complement_masked_tokens": complement_masked,
                        "train/mask_ratio_per_view": primary_masked / valid_count if valid_count else float("nan"),
                        "train/symmetric_views": 2,
                        "train/epoch": epoch_index,
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
                        mask_token_id=mask_token_id,
                        mask_probability=mask_probability,
                        max_eval_batches=max_eval_batches,
                    )
                    eval_logs["eval/epoch"] = epoch_index
                    eval_logs["step"] = global_step
                    accelerator.log(eval_logs, step=global_step)
                    if accelerator.is_main_process:
                        print(eval_logs)

                    eval_loss = eval_logs["eval/loss"]
                    if eval_loss == eval_loss and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(accelerator, model, output_dir, "best")

        save_checkpoint(accelerator, model, output_dir, f"epoch_{epoch_index}")

    save_checkpoint(accelerator, model, output_dir, global_step)
    progress.close()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 BERT Encoder V8")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config)


if __name__ == "__main__":
    main()
