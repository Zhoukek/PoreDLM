"""Train CTC basecalling directly from raw signals and continuous BERT."""

from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from data import SignalReferenceDataset, collate_signal_reference
from modeling_continuous_basecall import ContinuousBasecallModel
from poredlm.training_public.stage4_basecall.Basecalling.basecaller_v8_0420.ctc_crf import ctc_crf_loss, decode as ctc_crf_decode
from poredlm.training_public.stage4_basecall.Basecalling.basecaller_v8_0420.metrics import batch_bonito_accuracy, ctc_viterbi_decode


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def ctc_loss(logits: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor, label_lengths: torch.Tensor) -> torch.Tensor:
    log_probs = logits.transpose(0, 1).float().log_softmax(dim=-1)
    return torch.nn.functional.ctc_loss(
        log_probs,
        labels,
        lengths.cpu(),
        label_lengths.cpu(),
        blank=0,
        zero_infinity=True,
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr: float):
    warmup_steps = int(total_steps * warmup_ratio)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def lr_lambda(step: int, base_lr: float):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min(float(min_lr) / max(base_lr, 1e-12), 1.0) + (1.0 - min(float(min_lr) / max(base_lr, 1e-12), 1.0)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda step, base_lr=base_lr: lr_lambda(step, base_lr) for base_lr in base_lrs],
    )


def decode_predictions(logits: torch.Tensor, lengths: torch.Tensor, head_type: str) -> list[list[int]]:
    logits_tbc = logits.detach().float().transpose(0, 1)
    if head_type == "ctc_crf":
        predictions = []
        for index, length in enumerate(lengths.cpu().tolist()):
            if int(length) <= 0:
                predictions.append([])
                continue
            predictions.append(ctc_crf_decode(logits_tbc[: int(length), index:index + 1, :])[0])
        return predictions
    return ctc_viterbi_decode(logits_tbc, input_lengths=lengths, blank_idx=0)


def prediction_stats(
    predictions: list[list[int]],
    references: list[list[int]],
    input_lengths: torch.Tensor | list[int],
) -> dict[str, float]:
    """Return the same lightweight decoding diagnostics used by the public trainer."""
    lengths = input_lengths.detach().cpu().tolist() if isinstance(input_lengths, torch.Tensor) else input_lengths
    coverages = []
    blanks = []
    nonzero_lengths = []
    for prediction, reference, input_length in zip(predictions, references, lengths):
        decoded_length = float(len(prediction))
        coverages.append(decoded_length / max(len(reference), 1))
        blanks.append(max(1.0 - decoded_length / max(int(input_length), 1), 0.0))
        nonzero_lengths.append(decoded_length)
    return {
        "coverage": float(np.mean(coverages)) if coverages else 0.0,
        "blank": float(np.mean(blanks)) if blanks else 0.0,
        "nonzero_len": float(np.mean(nonzero_lengths)) if nonzero_lengths else 0.0,
    }


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    output_dir: Path,
    epoch: int,
    global_step: int,
    name: str,
    config: dict,
    best_accuracy: float,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": accelerator.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            # Keep the old key so older tooling can still inspect the checkpoint.
            "step": global_step,
            "best_accuracy": best_accuracy,
            "config": config,
        },
        output_dir / name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    seed_everything(int(cfg.get("seed", 42)) + int(os.environ.get("RANK", "0")))
    model_cfg = cfg["model"]
    backbone_cfg = cfg["backbone"]
    head_type = model_cfg.get("head_type", "ctc")
    mixed_precision = "no" if head_type == "ctc_crf" else cfg["training"].get("mixed_precision", "no")
    bert_finetune = not bool(backbone_cfg.get("freeze_bert", True))
    ddp_kwargs = DistributedDataParallelKwargs(
        # The BERT reconstruction prediction_head is not used by basecalling.
        find_unused_parameters=bert_finetune
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    if head_type == "ctc_crf":
        os.environ["CTC_CRF_STATE_LEN"] = str(model_cfg.get("ctc_crf_state_len", 5))
    model = ContinuousBasecallModel(
        cnn_checkpoint=backbone_cfg["cnn_checkpoint"],
        bert_checkpoint=backbone_cfg["bert_checkpoint"],
        num_classes=int(model_cfg.get("num_classes", 5)),
        freeze_cnn=bool(backbone_cfg.get("freeze_cnn", True)),
        freeze_bert=bool(backbone_cfg.get("freeze_bert", True)),
        bert_trainable_last_n_layers=backbone_cfg.get("bert_trainable_last_n_layers"),
        head_type=head_type,
        ctc_crf_state_len=int(model_cfg.get("ctc_crf_state_len", 5)),
        ctc_crf_blank_score=float(model_cfg.get("ctc_crf_blank_score", 2.0)),
        pre_head_type=model_cfg.get("pre_head_type", "none"),
        pre_head_transformer_nhead=int(model_cfg.get("pre_head_transformer_nhead", 8)),
        head_output_activation=model_cfg.get("head_output_activation"),
        head_output_scale=model_cfg.get("head_output_scale"),
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-3)),
    )
    data_cfg = cfg["data"]
    split_mode = data_cfg.get("split_mode", "explicit")
    if split_mode == "record":
        record_cfg = data_cfg.get("record", {})
        record_paths = record_cfg.get("paths", data_cfg.get("all", {}).get("paths"))
        if not record_paths:
            raise ValueError(
                "data.split_mode=record requires data.record.paths or data.all.paths."
            )
        full_dataset = SignalReferenceDataset(record_paths)
        train_ratio = float(record_cfg.get("train_ratio", 0.9))
        valid_ratio = float(record_cfg.get("valid_ratio", 0.1))
        if train_ratio <= 0 or valid_ratio <= 0 or abs(train_ratio + valid_ratio - 1.0) > 1e-6:
            raise ValueError("Record split requires positive train_ratio and valid_ratio summing to 1.")
        split_rng = np.random.default_rng(int(record_cfg.get("seed", cfg.get("seed", 42))))
        indices = np.arange(len(full_dataset))
        split_rng.shuffle(indices)
        train_size = int(round(len(indices) * train_ratio))
        train_size = min(max(train_size, 1), len(indices) - 1)
        train_dataset = Subset(full_dataset, indices[:train_size].tolist())
        valid_dataset = Subset(full_dataset, indices[train_size:].tolist())
        print(
            f"[Dataset] split_mode=record train={len(train_dataset)} "
            f"valid={len(valid_dataset)} seed={record_cfg.get('seed', cfg.get('seed', 42))}",
            flush=True,
        )
    elif split_mode == "explicit":
        train_dataset = SignalReferenceDataset(data_cfg["train"]["paths"])
        valid_dataset = SignalReferenceDataset(data_cfg["valid"]["paths"])
    else:
        raise ValueError(f"Unsupported data.split_mode: {split_mode}")
    batch_size = int(cfg["training"].get("device_micro_batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=int(cfg["data"]["train"].get("num_workers", 4)),
                              pin_memory=True, collate_fn=collate_signal_reference)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=int(cfg["data"]["valid"].get("num_workers", 2)),
                              pin_memory=True, collate_fn=collate_signal_reference)
    scheduler = build_scheduler(
        optimizer,
        int(cfg["training"]["num_epochs"]) * max(len(train_loader), 1),
        float(cfg["training"].get("warmup_ratio", 0.02)),
        float(cfg["training"].get("min_lr", 1e-6)),
    )
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    if accelerator.is_main_process:
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        accelerator.print(
            f"[Trainable] parameters={len(trainable)} "
            f"total={sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
        )
        if bert_finetune:
            accelerator.print(
                "[DDP] find_unused_parameters=True for partial BERT fine-tuning"
            )

    start_epoch = 1
    global_step = 0
    best_accuracy = -1.0
    resume_checkpoint = cfg["training"].get("resume_checkpoint")
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            # Compatibility with checkpoints produced by the former step loop.
            old_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
            start_epoch = old_step // max(len(train_loader), 1) + 1
        global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))

    wandb_run = None
    wandb_cfg = cfg.get("wandb", {})
    if accelerator.is_main_process and wandb_cfg.get("enabled", False):
        import wandb
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "continuous_basecall"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            config=cfg,
            mode=os.environ.get("WANDB_MODE", "online"),
        )

    output_dir = Path(cfg["training"]["output_dir"])
    num_epochs = int(cfg["training"]["num_epochs"])
    save_every_epochs = int(cfg["training"].get("save_every_epochs", 1))
    log_every_steps = int(cfg["training"].get("log_every_steps", 10))
    progress = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        progress.reset(total=len(train_loader))
        progress.set_description(f"[train epoch {epoch}/{num_epochs}]")
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            signal = batch["signal"].to(accelerator.device, non_blocking=True)
            lengths = batch["signal_lengths"].to(accelerator.device)
            labels = batch["target_labels"].to(accelerator.device)
            label_lengths = batch["target_lengths"].to(accelerator.device)
            with accelerator.autocast() if mixed_precision != "no" else nullcontext():
                logits, input_lengths = model(signal, lengths)
                if torch.any(label_lengths > input_lengths):
                    raise ValueError("A reference sequence is longer than the encoded signal sequence.")
                if head_type == "ctc_crf":
                    loss = ctc_crf_loss(logits.transpose(0, 1), labels, input_lengths, label_lengths, blank_idx=0)
                else:
                    loss = ctc_loss(logits, input_lengths, labels, label_lengths)
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), float(cfg["training"].get("gradient_clipping", 2.0))
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss_sum += float(loss.detach().item())
            epoch_batches += 1
            progress.update(1)
            if accelerator.is_main_process and global_step % log_every_steps == 0:
                progress.set_postfix(loss=f"{loss.item():.5f}")
                if wandb_run is not None:
                    train_predictions = decode_predictions(logits, input_lengths, head_type)
                    train_references = batch["target_seqs"]
                    train_stats = prediction_stats(train_predictions, train_references, input_lengths)
                    train_accuracy = batch_bonito_accuracy(train_predictions, train_references)
                    wandb_run.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/acc": float(train_accuracy),
                            "train/coverage": train_stats["coverage"],
                            "train/blank": train_stats["blank"],
                            "train/nonzero_len": train_stats["nonzero_len"],
                            "train/grad_norm": float(grad_norm),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "epoch": epoch,
                            "step": global_step,
                        },
                        step=global_step,
                    )

        model.eval()
        values = []
        predictions = []
        references = []
        validation_stats = []
        with torch.no_grad():
            for index, valid_batch in enumerate(valid_loader):
                valid_signal = valid_batch["signal"].to(accelerator.device, non_blocking=True)
                valid_lengths = valid_batch["signal_lengths"].to(accelerator.device)
                valid_logits, valid_input_lengths = model(valid_signal, valid_lengths)
                valid_labels = valid_batch["target_labels"].to(accelerator.device)
                valid_label_lengths = valid_batch["target_lengths"].to(accelerator.device)
                if torch.any(valid_label_lengths > valid_input_lengths):
                    raise ValueError("A validation reference is longer than its encoded signal sequence.")
                if head_type == "ctc_crf":
                    valid_loss = ctc_crf_loss(valid_logits.transpose(0, 1), valid_labels, valid_input_lengths, valid_label_lengths, blank_idx=0)
                else:
                    valid_loss = ctc_loss(valid_logits, valid_input_lengths, valid_labels, valid_label_lengths)
                values.append(valid_loss)
                batch_predictions = decode_predictions(valid_logits, valid_input_lengths, head_type)
                batch_references = valid_batch["target_seqs"]
                predictions.extend(batch_predictions)
                references.extend(batch_references)
                validation_stats.append(
                    prediction_stats(batch_predictions, batch_references, valid_input_lengths)
                )
                if index + 1 >= int(cfg["training"].get("max_eval_batches", 20)):
                    break
        eval_loss = torch.stack(values).mean() if values else torch.zeros((), device=accelerator.device)
        eval_loss = accelerator.gather_for_metrics(eval_loss.reshape(1)).mean().item()
        local_accuracy = batch_bonito_accuracy(predictions, references) if predictions else 0.0
        eval_accuracy = accelerator.gather_for_metrics(
            torch.tensor([local_accuracy], device=accelerator.device)
        ).mean().item()
        eval_coverage = float(np.mean([item["coverage"] for item in validation_stats])) if validation_stats else 0.0
        eval_blank = float(np.mean([item["blank"] for item in validation_stats])) if validation_stats else 0.0
        eval_nonzero_len = float(np.mean([item["nonzero_len"] for item in validation_stats])) if validation_stats else 0.0
        eval_diagnostics = accelerator.gather_for_metrics(
            torch.tensor([[eval_coverage, eval_blank, eval_nonzero_len]], device=accelerator.device)
        ).mean(dim=0).tolist()
        eval_coverage, eval_blank, eval_nonzero_len = eval_diagnostics
        is_new_best = eval_accuracy > best_accuracy
        if is_new_best:
            best_accuracy = eval_accuracy
        epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
        if accelerator.is_main_process:
            accelerator.print(
                f"epoch={epoch} train_loss={epoch_loss:.6f} "
                f"eval_loss={eval_loss:.6f} eval_accuracy={eval_accuracy:.4f} "
                f"coverage={eval_coverage:.4f} blank={eval_blank:.4f} "
                f"nonzero_len={eval_nonzero_len:.2f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": epoch_loss,
                        "val/loss": eval_loss,
                        "val/acc": eval_accuracy,
                        "val/coverage": eval_coverage,
                        "val/blank": eval_blank,
                        "val/nonzero_len": eval_nonzero_len,
                        "best/acc": best_accuracy,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )

        if is_new_best:
            save_checkpoint(
                accelerator, model, optimizer, scheduler, output_dir,
                epoch, global_step, "ckpt_best.pt", cfg, best_accuracy,
            )
        if epoch % save_every_epochs == 0:
            save_checkpoint(
                accelerator, model, optimizer, scheduler, output_dir,
                epoch, global_step, f"epoch_{epoch}.pt", cfg, best_accuracy,
            )
            save_checkpoint(
                accelerator, model, optimizer, scheduler, output_dir,
                epoch, global_step, "ckpt_last.pt", cfg, best_accuracy,
            )

    if num_epochs < start_epoch:
        accelerator.print(f"No training performed: checkpoint already reached epoch {start_epoch - 1}.")
    if wandb_run is not None: wandb_run.finish()
    progress.close()


if __name__ == "__main__":
    main()
