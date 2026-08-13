# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

from .checkpointing import (
    load_checkpoint_payload,
    load_model_weights,
    restore_training_state,
    save_checkpoint,
)
from .config import add_train_arguments, config_from_args
from .data import build_datasets, build_loaders
from .distributed import create_accelerator, is_main, setup_logger
from .loops import evaluate, train_one_epoch
from .modeling import apply_delayed_unfreeze, build_model, count_parameters
from .optim import build_optimizer, build_scheduler

try:
    import wandb
except Exception:
    wandb = None


class RunState:
    def __init__(self, cfg, loaders):
        self.data = cfg.data
        self.model = cfg.model
        self.train = cfg.train
        self.runtime = cfg.runtime
        self.loaders = loaders


def _has_unfreeze_targets(model_cfg) -> bool:
    return (
        model_cfg.unfreeze_last_n_layers > 0
        or model_cfg.unfreeze_layer_start is not None
        or model_cfg.unfreeze_layer_end is not None
    )


def _resolve_decoder_and_amp(cfg, device, logger, accelerator) -> tuple[str, bool]:
    decoder_mode = cfg.train.train_decoder
    if cfg.model.head_type == "ctc" and decoder_mode != "ctc_viterbi":
        if is_main(accelerator):
            logger.warning("[Decoder] CTC head only supports ctc_viterbi during training metrics; overriding.")
        decoder_mode = "ctc_viterbi"
    if decoder_mode == "ctc_crf" and cfg.model.head_type != "ctc_crf":
        raise ValueError("--train_decoder ctc_crf requires --head_type ctc_crf.")
    use_amp = bool(cfg.runtime.amp and device.type == "cuda" and decoder_mode != "ctc_crf")
    return decoder_mode, use_amp


def run_training(argv: Optional[List[str]] = None) -> None:
    parser = add_train_arguments(argparse.ArgumentParser(description="Train basecaller with a cleaner pipeline."))
    args = parser.parse_args(argv)
    explicit_model_path = "--model_name_or_path" in (argv if argv is not None else sys.argv[1:])
    cfg = config_from_args(args)

    resume_state_dict = None
    resume_checkpoint: Dict = {}
    if cfg.train.resume_ckpt:
        resume_state_dict, saved_model_config, resume_checkpoint = load_checkpoint_payload(cfg.train.resume_ckpt)
        cfg.model.apply_checkpoint_config(saved_model_config, keep_model_path=explicit_model_path)
    if not cfg.model.model_name_or_path:
        raise ValueError("--model_name_or_path is required unless --resume_ckpt contains model_config.model_name_or_path.")

    resume_epoch = int(resume_checkpoint.get("epoch", 0) or 0)
    has_unfreeze_targets = _has_unfreeze_targets(cfg.model)
    resume_after_delayed_unfreeze = bool(
        cfg.train.resume_ckpt
        and cfg.model.unfreeze_after_epoch > 0
        and has_unfreeze_targets
        and resume_epoch > cfg.model.unfreeze_after_epoch
    )

    accelerator, backend, backend_note = create_accelerator(cfg.runtime, streaming=cfg.data.streaming)
    device = accelerator.device
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)

    from basecall.utils import seed_everything

    seed_everything(cfg.train.seed + accelerator.process_index)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    logger = setup_logger(cfg.train.output_dir, accelerator)
    if is_main(accelerator):
        logger.info("[Accelerate] world_size=%s rank=%s device=%s backend=%s", accelerator.num_processes, accelerator.process_index, device, backend)
        if backend_note:
            logger.warning(backend_note)
        logger.info("[Config]\n%s", json.dumps(asdict(cfg), indent=2, ensure_ascii=False, default=str))

    model_build = build_model(cfg.model, resume_after_delayed_unfreeze=resume_after_delayed_unfreeze)
    model = model_build.model
    if resume_state_dict is not None:
        missing, unexpected = load_model_weights(model, resume_state_dict, strict=False)
        if is_main(accelerator):
            logger.info("[Resume] model weights loaded missing=%s unexpected=%s", len(missing), len(unexpected))
    elif cfg.train.pretrained_ckpt:
        pretrained_state, _pretrained_model_config, _full = load_checkpoint_payload(cfg.train.pretrained_ckpt)
        missing, unexpected = load_model_weights(model, pretrained_state, strict=cfg.train.pretrained_strict)
        if is_main(accelerator):
            logger.info("[Pretrained] loaded missing=%s unexpected=%s", len(missing), len(unexpected))

    if is_main(accelerator):
        total, trainable = count_parameters(model)
        logger.info("[Model] total_params=%s trainable_params=%s", f"{total:,}", f"{trainable:,}")
        logger.info("[Model] pre_head=%s head=%s", model.pre_head.__class__.__name__, model.base_head.__class__.__name__)

    datasets = build_datasets(cfg.data)
    if is_main(accelerator):
        logger.info("[Data] %s", datasets.summary)
    loaders = build_loaders(datasets, cfg.model, cfg.train, model.tokenizer, pin_memory=(device.type == "cuda"))
    state = RunState(cfg, loaders)

    optimizer = build_optimizer(model, cfg.train)
    scheduler, total_steps, warmup_steps = build_scheduler(optimizer, cfg.train, steps_per_epoch=loaders.steps_per_epoch)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, loaders.train_loader, scheduler)
    loaders.train_loader = train_loader
    if loaders.val_loader is not None:
        loaders.val_loader = accelerator.prepare(loaders.val_loader)
    if loaders.test_loader is not None:
        loaders.test_loader = accelerator.prepare(loaders.test_loader)

    start_epoch, best_acc, global_step = 1, -1.0, 0
    if cfg.train.resume_ckpt:
        start_epoch, best_acc, global_step = restore_training_state(
            resume_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger if is_main(accelerator) else None,
        )

    decoder_mode, use_amp = _resolve_decoder_and_amp(cfg, device, logger, accelerator)

    wandb_enabled = bool(cfg.runtime.use_wandb and wandb is not None and is_main(accelerator))
    if cfg.runtime.use_wandb and wandb is None and is_main(accelerator):
        logger.warning("[wandb] requested but wandb is not installed.")
    if wandb_enabled:
        wandb.init(
            project=cfg.runtime.wandb_project,
            entity=cfg.runtime.wandb_entity,
            name=cfg.runtime.wandb_run_name,
            group=cfg.runtime.wandb_group,
            job_type="train",
            config=asdict(cfg),
        )

    if is_main(accelerator):
        logger.info("[Scheduler] steps_per_epoch=%s total_steps=%s warmup_steps=%s", loaders.steps_per_epoch, total_steps, warmup_steps)
        if cfg.train.max_steps_per_epoch > 0:
            logger.info("[Train] max_steps_per_epoch=%s is active.", cfg.train.max_steps_per_epoch)

    delayed_unfreeze_done = resume_after_delayed_unfreeze
    for epoch in range(start_epoch, cfg.train.num_epochs + 1):
        if (
            not delayed_unfreeze_done
            and cfg.model.unfreeze_after_epoch > 0
            and epoch > cfg.model.unfreeze_after_epoch
            and has_unfreeze_targets
        ):
            raw_model = accelerator.unwrap_model(model)
            added = apply_delayed_unfreeze(
                raw_model,
                cfg.model,
                cfg.train.weight_decay,
                optimizer,
                backbone_lr=cfg.train.backbone_lr,
                scheduler=scheduler,
            )
            delayed_unfreeze_done = True
            if is_main(accelerator):
                logger.info("[Unfreeze] epoch=%s added_params=%s", epoch, added)

        train_loss, global_step = train_one_epoch(
            accelerator=accelerator,
            model=model,
            loader=loaders.train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=state,
            device=device,
            decoder_mode=decoder_mode,
            use_amp=use_amp,
            global_step=global_step,
            logger=logger,
            wandb_enabled=wandb_enabled,
        )
        val_loss, val_acc = None, None
        if loaders.val_loader is not None:
            val_loss, val_acc = evaluate(
                accelerator=accelerator,
                model=model,
                loader=loaders.val_loader,
                cfg=state,
                device=device,
                decoder_mode=decoder_mode,
                use_amp=use_amp,
                split_name="val",
            )

        if is_main(accelerator):
            logger.info("[Epoch] epoch=%s train_loss=%.4f val_loss=%s val_acc=%s", epoch, train_loss, val_loss, val_acc)
            if wandb_enabled and wandb is not None:
                payload = {"epoch": epoch, "train/epoch_loss": train_loss}
                if val_loss is not None:
                    payload.update({"val/loss": float(val_loss), "val/acc": float(val_acc)})
                wandb.log(payload, step=global_step)

        accelerator.wait_for_everyone()
        if is_main(accelerator) and epoch % max(cfg.train.save_every, 1) == 0:
            save_checkpoint(
                os.path.join(cfg.train.output_dir, "ckpt_last.pt"),
                accelerator,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                global_step=global_step,
                model_config=model_build.checkpoint_model_config,
                extra={"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            )
            save_checkpoint(
                os.path.join(cfg.train.output_dir, f"ckpt_epoch_{epoch}.pt"),
                accelerator,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                global_step=global_step,
                model_config=model_build.checkpoint_model_config,
                extra={"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            )
            logger.info("[CKPT] saved epoch=%s", epoch)

        if is_main(accelerator) and cfg.train.save_best and val_acc is not None and float(val_acc) > best_acc:
            best_acc = float(val_acc)
            save_checkpoint(
                os.path.join(cfg.train.output_dir, "ckpt_best.pt"),
                accelerator,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                global_step=global_step,
                model_config=model_build.checkpoint_model_config,
                extra={"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc},
            )
            logger.info("[CKPT] new best val_acc=%.4f", best_acc)

    if loaders.test_loader is not None:
        test_loss, test_acc = evaluate(
            accelerator=accelerator,
            model=model,
            loader=loaders.test_loader,
            cfg=state,
            device=device,
            decoder_mode=decoder_mode,
            use_amp=use_amp,
            split_name="test",
        )
        if is_main(accelerator):
            logger.info("[Test] loss=%.4f acc=%.4f", test_loss, test_acc)

    if wandb_enabled and wandb is not None:
        wandb.finish()
