# -*- coding: utf-8 -*-
from __future__ import annotations

from contextlib import nullcontext
from typing import Tuple

import torch
from tqdm.auto import tqdm

from basecall.metrics import batch_bonito_accuracy

from .batching import prepare_batch
from .decoding import decode_batch, rebuild_target_seqs
from .distributed import is_main, reduce_mean, reduce_min_bool
from .losses import compute_loss

try:
    import wandb
except Exception:
    wandb = None


def train_one_epoch(
    *,
    accelerator,
    model,
    loader,
    optimizer,
    scheduler,
    cfg,
    device,
    decoder_mode: str,
    use_amp: bool,
    global_step: int,
    logger,
    wandb_enabled: bool,
) -> Tuple[float, int]:
    model.train()
    total_loss, n_batches = 0.0, 0
    total_hint = cfg.train.max_steps_per_epoch or None
    iterator = tqdm(loader, total=total_hint, disable=not is_main(accelerator), desc="[train]")

    for step, batch in enumerate(iterator, start=1):
        if cfg.train.max_steps_per_epoch > 0 and step > cfg.train.max_steps_per_epoch:
            break

        prepared = prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast() if use_amp else nullcontext():
            logits_btc = model(prepared.input_ids, attention_mask=prepared.attention_mask)
            logits_tbc = logits_btc.transpose(0, 1)
            loss = compute_loss(
                cfg.model,
                logits_tbc,
                prepared.target_labels,
                prepared.target_lengths,
                prepared.input_lengths,
            )

        finite = reduce_min_bool(accelerator, bool(torch.isfinite(loss).item()), device)
        if finite:
            accelerator.backward(loss)
            if cfg.train.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
        elif is_main(accelerator):
            logger.warning("[Train] non-finite loss detected across ranks; skipped optimizer step.")

        loss_value = float(loss.detach().item())
        if finite:
            total_loss += loss_value
            n_batches += 1

        if is_main(accelerator) and cfg.train.log_interval > 0 and step % cfg.train.log_interval == 0:
            lr = float(optimizer.param_groups[0]["lr"])
            logger.info("[Train] step=%s loss=%.4f lr=%.6g", step, loss_value, lr)
            if wandb_enabled and wandb is not None:
                wandb.log({"train/loss": loss_value, "lr": lr, "trainer/global_step": global_step}, step=global_step)

        if (
            cfg.train.eval_interval > 0
            and step % cfg.train.eval_interval == 0
            and cfg.loaders.val_loader is not None
        ):
            val_loss, val_acc = evaluate(
                accelerator=accelerator,
                model=model,
                loader=cfg.loaders.val_loader,
                cfg=cfg,
                device=device,
                decoder_mode=decoder_mode,
                use_amp=use_amp,
                split_name="val@step",
            )
            if is_main(accelerator):
                logger.info("[Val@Step] step=%s loss=%.4f acc=%.4f", step, val_loss, val_acc)
                if wandb_enabled and wandb is not None:
                    wandb.log({"val/step_loss": val_loss, "val/step_acc": val_acc}, step=global_step)

    return reduce_mean(accelerator, total_loss / max(n_batches, 1), device), global_step


@torch.no_grad()
def evaluate(
    *,
    accelerator,
    model,
    loader,
    cfg,
    device,
    decoder_mode: str,
    use_amp: bool,
    split_name: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    iterator = tqdm(loader, disable=not is_main(accelerator), desc=f"[{split_name}]")

    for batch in iterator:
        prepared = prepare_batch(batch, device)
        target_seqs = rebuild_target_seqs(prepared.target_labels, prepared.target_lengths)

        with accelerator.autocast() if use_amp else nullcontext():
            logits_btc = model(prepared.input_ids, attention_mask=prepared.attention_mask)
            logits_tbc = logits_btc.transpose(0, 1)
            loss = compute_loss(
                cfg.model,
                logits_tbc,
                prepared.target_labels,
                prepared.target_lengths,
                prepared.input_lengths,
            )

        pred_seqs = decode_batch(
            logits_tbc,
            prepared.input_lengths,
            decoder=decoder_mode,
            koi_blank_score=cfg.train.koi_blank_score,
        )
        acc = batch_bonito_accuracy(
            pred_seqs,
            target_seqs,
            balanced=cfg.train.acc_balanced,
            min_coverage=cfg.train.acc_min_coverage,
        )
        total_loss += float(loss.detach().item())
        total_acc += float(acc)
        n_batches += 1

    avg_loss = reduce_mean(accelerator, total_loss / max(n_batches, 1), device)
    avg_acc = reduce_mean(accelerator, total_acc / max(n_batches, 1), device)
    model.train()
    return avg_loss, avg_acc
