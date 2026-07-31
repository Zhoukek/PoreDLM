"""Train a waveform decoder while keeping the DLM and Stage 1 tokenizer frozen."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel

from modeling_waveform_decoder import WaveformDecoder
from token_dataset import TokenSequenceDataset, WaveformTokenCollator


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def validate_pretrained_dir(path_value: str, name: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    config_path = path / "config.json"
    weight_names = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not config_path.is_file() or not any((path / item).is_file() for item in weight_names):
        raise FileNotFoundError(
            f"{name} path is not a complete save_pretrained directory: {path}. "
            "Expected config.json and model.safetensors/pytorch_model.bin (or a shard index)."
        )
    return path


def build_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    training_cfg = config["training"]
    paths = data_cfg.get(f"{split}_paths")
    if not paths:
        raise ValueError(f"Missing data.{split}_paths.")
    dataset = TokenSequenceDataset(
        paths,
        pattern=str(data_cfg.get("file_pattern", "*.npy")),
        dtype=str(data_cfg.get("token_dtype", "uint32")),
        shuffle_files=bool(data_cfg.get(f"{split}_shuffle_files", split == "train")),
        repeat=split == "train",
        seed=int(config.get("seed", 42)),
    )
    collator = WaveformTokenCollator(
        pad_token_id=int(data_cfg.get("pad_token_id", 1)),
        bos_token_id=int(data_cfg.get("bos_token_id", 2)),
        eos_token_id=int(data_cfg.get("eos_token_id", 3)),
        token_offset=int(data_cfg.get("token_offset", 128)),
        codebook_size=int(data_cfg.get("codebook_size", 65536)),
        max_length=int(data_cfg.get("max_length", 1536)),
        strict_boundaries=bool(data_cfg.get("strict_boundaries", True)),
    )
    workers = int(data_cfg.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "batch_size": int(training_cfg.get(f"{split}_batch_size", training_cfg["micro_batch_size"])),
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "drop_last": split == "train",
        "collate_fn": collator,
    }
    if workers:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
    return DataLoader(dataset, **kwargs)


def dlm_forward_kwargs(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    hidden_key = str(config["model"].get("hidden_state_key", "ode_hidden_state"))
    sampling = config.get("sampling", {})
    kwargs: dict[str, Any] = {}
    if hidden_key == "context_hidden_state":
        kwargs["return_context"] = True
    elif hidden_key == "ode_hidden_state":
        kwargs.update(
            return_ode_hidden=True,
            ode_steps=int(sampling.get("ode_steps", 4)),
            ode_start_t=float(sampling.get("ode_start_t", 0.85)),
            ode_self_cond_cfg_scale=float(sampling.get("ode_self_cond_cfg_scale", 1.0)),
        )
    elif hidden_key == "sde_hidden_state":
        kwargs.update(
            return_sde_hidden=True,
            sde_steps=int(sampling.get("sde_steps", 4)),
            sde_start_t=float(sampling.get("sde_start_t", 0.85)),
            sde_gamma=float(sampling.get("sde_gamma", 0.1)),
            sde_self_cond_cfg_scale=float(sampling.get("sde_self_cond_cfg_scale", 1.0)),
            sde_seed=sampling.get("sde_seed"),
        )
    elif hidden_key != "last_hidden_state":
        raise ValueError(f"Unsupported model.hidden_state_key={hidden_key!r}.")
    return hidden_key, kwargs


def make_targets(
    dlm: torch.nn.Module,
    tokenizer: torch.nn.Module,
    batch: dict[str, Any],
    hidden_key: str,
    forward_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    with torch.no_grad():
        outputs = dlm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **forward_kwargs,
        )
        hidden = outputs[hidden_key].float()
        # DLM positions are [BOS, content..., EOS]; the codec target contains content only.
        lengths = [int(length) for length in batch["content_mask"].sum(dim=1).tolist()]
        targets = [
            tokenizer.decode_token(batch["codec_token_ids"][index : index + 1, :length]).float()
            for index, length in enumerate(lengths)
        ]
    return hidden, targets, lengths


def reconstruction_loss(
    decoder: torch.nn.Module,
    hidden: torch.Tensor,
    targets: list[torch.Tensor],
    lengths: list[int],
    loss_type: str,
) -> torch.Tensor:
    losses = []
    for index, (target, length) in enumerate(zip(targets, lengths)):
        if length <= 0:
            continue
        # Trim before convolution so EOS/padding hidden states cannot affect the boundary.
        prediction = decoder(hidden[index : index + 1, 1 : 1 + length])
        common_length = min(prediction.shape[-1], target.shape[-1])
        prediction = prediction[..., :common_length]
        target = target[..., :common_length]
        if loss_type == "l1":
            losses.append(F.l1_loss(prediction, target))
        elif loss_type == "mse":
            losses.append(F.mse_loss(prediction, target))
        elif loss_type == "smooth_l1":
            losses.append(F.smooth_l1_loss(prediction, target))
        else:
            raise ValueError(f"Unsupported training.loss_type={loss_type!r}.")
    if not losses:
        raise ValueError("Batch contains no content tokens.")
    return torch.stack(losses).mean()


def scheduler_for(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> LambdaLR:
    training = config["training"]
    warmup = int(training.get("warmup_steps", 0))
    total = int(training["max_steps"])

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def save_checkpoint(
    accelerator: Accelerator,
    decoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    output_dir: Path,
    step: int,
    config: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint_dir = output_dir / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": accelerator.get_state_dict(decoder),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }
    torch.save(state, checkpoint_dir / "checkpoint.pt")
    unwrapped = accelerator.unwrap_model(decoder)
    with open(checkpoint_dir / "decoder_config.json", "w", encoding="utf-8") as handle:
        json.dump({"hidden_size": unwrapped.hidden_size}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    training = config["training"]
    wandb_enabled = bool(config.get("wandb", {}).get("enabled", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(training.get("mixed_precision", "no")),
        log_with="wandb" if wandb_enabled else None,
    )
    model_cfg = config["model"]
    dlm_path = validate_pretrained_dir(str(model_cfg["dlm_path"]), "DLM")
    tokenizer_path = validate_pretrained_dir(str(model_cfg["tokenizer_path"]), "Tokenizer")
    elf_src_path = model_cfg.get("elf_src_path")
    if elf_src_path is None:
        candidate = dlm_path.parents[2] / "ELF-pytorch-port" / "src"
        elf_src_path = str(candidate) if candidate.is_dir() else None
    if elf_src_path and str(elf_src_path) not in sys.path:
        sys.path.insert(0, str(elf_src_path))
    dlm = AutoModel.from_pretrained(
        str(dlm_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoModel.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    freeze(dlm)
    freeze(tokenizer)

    decoder = WaveformDecoder(hidden_size=int(model_cfg.get("hidden_size", 768)))
    if bool(model_cfg.get("initialize_from_tokenizer_decoder", True)):
        decoder.initialize_from_stage1(tokenizer)

    train_loader = build_loader(config, "train")
    valid_loader = build_loader(config, "valid")
    optimizer = AdamW(
        decoder.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    scheduler = scheduler_for(optimizer, config)
    decoder, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        decoder, optimizer, train_loader, valid_loader, scheduler
    )
    dlm.to(accelerator.device)
    tokenizer.to(accelerator.device)
    hidden_key, forward_kwargs = dlm_forward_kwargs(config)
    output_dir = Path(training["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    accelerator.wait_for_everyone()
    if wandb_enabled:
        accelerator.init_trackers(
            project_name=str(config["wandb"].get("project", "waveform_decoder")),
            config=config,
            init_kwargs={"wandb": {"name": config["wandb"].get("name")}},
        )

    max_steps = int(training["max_steps"])
    loss_type = str(training.get("loss_type", "smooth_l1"))
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    step = 0
    decoder.train()
    for batch in train_loader:
        with accelerator.accumulate(decoder):
            hidden, targets, lengths = make_targets(
                dlm, tokenizer, batch, hidden_key, forward_kwargs
            )
            loss = reconstruction_loss(decoder, hidden, targets, lengths, loss_type)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(decoder.parameters(), float(training.get("max_grad_norm", 1.0)))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        step += 1
        progress.update(1)
        if step % int(training.get("log_every_steps", 10)) == 0:
            gathered_loss = accelerator.gather(loss.detach()).mean().item()
            progress.set_postfix(loss=f"{gathered_loss:.5f}")
            accelerator.log({"train/loss": gathered_loss, "train/lr": scheduler.get_last_lr()[0]}, step=step)

        if step % int(training.get("eval_every_steps", 1000)) == 0:
            decoder.eval()
            eval_losses = []
            for batch_index, valid_batch in enumerate(valid_loader):
                with torch.no_grad():
                    hidden, targets, lengths = make_targets(
                        dlm, tokenizer, valid_batch, hidden_key, forward_kwargs
                    )
                    eval_losses.append(
                        reconstruction_loss(decoder, hidden, targets, lengths, loss_type)
                    )
                if batch_index + 1 >= int(training.get("max_eval_batches", 100)):
                    break
            if eval_losses:
                eval_loss = accelerator.gather(torch.stack(eval_losses).mean()).mean().item()
                accelerator.log({"valid/loss": eval_loss}, step=step)
            decoder.train()

        if step % int(training.get("save_every_steps", 5000)) == 0:
            save_checkpoint(accelerator, decoder, optimizer, scheduler, output_dir, step, config)
        if step >= max_steps:
            break

    save_checkpoint(accelerator, decoder, optimizer, scheduler, output_dir, step, config)
    accelerator.end_training()


if __name__ == "__main__":
    main()
