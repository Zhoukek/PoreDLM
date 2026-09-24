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
from torch.optim import AdamW
from torch.utils.data import DataLoader
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


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    output_dir: Path,
    step: int,
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
            "step": step,
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
    head_type = model_cfg.get("head_type", "ctc")
    mixed_precision = "no" if head_type == "ctc_crf" else cfg["training"].get("mixed_precision", "no")
    accelerator = Accelerator(mixed_precision=mixed_precision)

    if head_type == "ctc_crf":
        os.environ["CTC_CRF_STATE_LEN"] = str(model_cfg.get("ctc_crf_state_len", 5))
    model = ContinuousBasecallModel(
        cnn_checkpoint=cfg["backbone"]["cnn_checkpoint"],
        bert_checkpoint=cfg["backbone"]["bert_checkpoint"],
        num_classes=int(model_cfg.get("num_classes", 5)),
        freeze_cnn=bool(cfg["backbone"].get("freeze_cnn", True)),
        freeze_bert=bool(cfg["backbone"].get("freeze_bert", True)),
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
    train_dataset = SignalReferenceDataset(cfg["data"]["train"]["paths"])
    valid_dataset = SignalReferenceDataset(cfg["data"]["valid"]["paths"])
    batch_size = int(cfg["training"].get("device_micro_batch_size", 8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=int(cfg["data"]["train"].get("num_workers", 4)),
                              pin_memory=True, collate_fn=collate_signal_reference)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=int(cfg["data"]["valid"].get("num_workers", 2)),
                              pin_memory=True, collate_fn=collate_signal_reference)
    scheduler = build_scheduler(
        optimizer,
        int(cfg["training"]["max_train_steps"]),
        float(cfg["training"].get("warmup_ratio", 0.02)),
        float(cfg["training"].get("min_lr", 1e-6)),
    )
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    start_step = 0
    best_accuracy = -1.0
    resume_checkpoint = cfg["training"].get("resume_checkpoint")
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("step", 0))
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
    max_steps = int(cfg["training"]["max_train_steps"])
    eval_every = int(cfg["training"].get("eval_every_steps", 1000))
    save_every = int(cfg["training"].get("save_every_steps", 5000))
    step = start_step
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    while step < max_steps:
        for batch in train_loader:
            model.train(); optimizer.zero_grad(set_to_none=True)
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
            accelerator.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clipping", 2.0)))
            optimizer.step(); scheduler.step(); step += 1; progress.update(1)
            if accelerator.is_main_process and step % int(cfg["training"].get("log_every_steps", 10)) == 0:
                progress.set_postfix(loss=f"{loss.item():.5f}")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=step)
            if step % eval_every == 0:
                model.eval(); values = []; predictions = []; references = []
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
                        predictions.extend(decode_predictions(valid_logits, valid_input_lengths, head_type))
                        references.extend(valid_batch["target_seqs"])
                        if index + 1 >= int(cfg["training"].get("max_eval_batches", 20)): break
                eval_loss = torch.stack(values).mean() if values else torch.zeros((), device=accelerator.device)
                eval_loss = accelerator.gather_for_metrics(eval_loss.reshape(1)).mean().item()
                local_accuracy = batch_bonito_accuracy(predictions, references) if predictions else 0.0
                eval_accuracy = accelerator.gather_for_metrics(
                    torch.tensor([local_accuracy], device=accelerator.device)
                ).mean().item()
                is_new_best = eval_accuracy > best_accuracy
                if is_new_best:
                    best_accuracy = eval_accuracy
                if accelerator.is_main_process:
                    accelerator.print(f"step={step} eval_loss={eval_loss:.6f} eval_accuracy={eval_accuracy:.4f}")
                    if wandb_run is not None:
                        wandb_run.log({"valid/loss": eval_loss, "valid/accuracy": eval_accuracy}, step=step)
                if is_new_best:
                    save_checkpoint(accelerator, model, optimizer, scheduler, output_dir, step, "best.pt", cfg, best_accuracy)
            if step % save_every == 0:
                save_checkpoint(accelerator, model, optimizer, scheduler, output_dir, step, f"step_{step}.pt", cfg, best_accuracy)
            if step >= max_steps: break
    save_checkpoint(accelerator, model, optimizer, scheduler, output_dir, step, "last.pt", cfg, best_accuracy)
    if wandb_run is not None: wandb_run.finish()
    progress.close()


if __name__ == "__main__":
    main()
