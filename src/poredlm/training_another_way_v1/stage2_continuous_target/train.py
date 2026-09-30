"""Train Stage 2 with continuous CNN inputs and frozen Stage 1 VQ targets."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

from poredlm.training_another_way_v1.data import PoreSignalDataset
from poredlm.training_another_way_v1.stage1_vq_train.modeling import PoreVQCodec
from poredlm.training_another_way_v1.stage2_continuous_target.modeling import (
    ContinuousTargetBert,
    ContinuousTargetConfig,
)


def _build_loader(cfg: dict, split: str, train: bool):
    data_cfg = cfg["data"][split]
    dataset = PoreSignalDataset(
        data_cfg["paths"],
        chunk_size=data_cfg.get("chunk_size", 6000),
        memmap_dtype=data_cfg.get("memmap_dtype", "float32"),
        buffer_size=data_cfg.get("buffer_size", 20000 if train else 0),
        shuffle_buffer=train,
        repeat=train,
        seed=cfg.get("seed", 42),
    )
    workers = int(data_cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(cfg["training"].get("device_micro_batch_size", 8)),
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", False)) and workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def _make_span_mask(
    attention_mask: torch.Tensor,
    start_probability: float,
    span_length: int,
) -> torch.Tensor:
    """Sample overlapping masked spans without touching padded positions."""
    if start_probability <= 0:
        raise ValueError("mask_probability must be positive.")
    starts = (torch.rand(attention_mask.shape, device=attention_mask.device) < start_probability)
    starts &= attention_mask.bool()
    mask = torch.zeros_like(starts)
    for offset in range(max(1, int(span_length))):
        shifted = torch.zeros_like(starts)
        if offset == 0:
            shifted = starts
        elif offset < starts.shape[1]:
            shifted[:, offset:] = starts[:, :-offset]
        mask |= shifted
    mask &= attention_mask.bool()
    for row in range(mask.shape[0]):
        valid = torch.nonzero(attention_mask[row].bool(), as_tuple=False).flatten()
        if valid.numel() and not mask[row].any():
            mask[row, valid[torch.randint(valid.numel(), (1,), device=mask.device)]] = True
    return mask


def _encoded_attention(lengths: torch.Tensor, sequence_length: int, stride: int) -> torch.Tensor:
    encoded_lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
    positions = torch.arange(sequence_length, device=lengths.device).unsqueeze(0)
    return (positions < encoded_lengths.unsqueeze(1)).long()


@torch.no_grad()
def _stage1_features_and_targets(stage1, signal: torch.Tensor):
    z = stage1.cnn_model.encode(signal.unsqueeze(1)).transpose(1, 2)
    q, indices, _, _ = stage1.vq(z, return_loss_breakdown=True)
    return z, q, indices


def _save_checkpoint(accelerator, model, output_dir: Path, step: int, cfg: dict) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    save_dir = output_dir / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    torch.save(
        {
            "model_state_dict": unwrapped.state_dict(),
            "config": vars(unwrapped.config),
            "global_step": step,
        },
        save_dir / "pytorch_model.pt",
    )
    (save_dir / "training_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    latest = output_dir / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(save_dir, latest)


def _make_model(cfg: dict) -> ContinuousTargetBert:
    return ContinuousTargetBert(ContinuousTargetConfig(**cfg["model"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    accelerator = Accelerator(mixed_precision=cfg["training"].get("mixed_precision", "no"))
    stage1_checkpoint = cfg["stage1"]["checkpoint"]
    stage1 = PoreVQCodec.from_pretrained(stage1_checkpoint).to(accelerator.device)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)

    wandb_run = None
    wandb_cfg = cfg.get("wandb", {})
    if accelerator.is_main_process and wandb_cfg.get("enabled", False):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled, but the `wandb` package is not installed."
            ) from exc
        init_kwargs = {
            "project": wandb_cfg.get("project", "training_another_way_v1_stage2_vq_target"),
            "config": cfg,
            "mode": os.environ.get("WANDB_MODE", "online"),
        }
        if wandb_cfg.get("entity"):
            init_kwargs["entity"] = wandb_cfg["entity"]
        if wandb_cfg.get("name"):
            init_kwargs["name"] = wandb_cfg["name"]
        if wandb_cfg.get("group"):
            init_kwargs["group"] = wandb_cfg["group"]
        wandb_run = wandb.init(**init_kwargs)

    model = _make_model(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 5e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    train_loader = _build_loader(cfg, "train", train=True)
    valid_loader = _build_loader(cfg, "valid", train=False)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    model.to(accelerator.device)

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["training"].get("max_train_steps", 100000))
    eval_every = int(cfg["training"].get("eval_every_steps", 1000))
    save_every = int(cfg["training"].get("save_every_steps", 5000))
    max_eval_batches = int(cfg["training"].get("max_eval_batches", 20))
    stride = int(cfg["stage1"].get("cnn_stride", 5))
    mask_probability = float(cfg["masking"].get("start_probability", 0.065))
    span_length = int(cfg["masking"].get("span_length", 10))
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)

    step = 0
    while step < max_steps:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            signal = batch["signal"].to(accelerator.device, non_blocking=True)
            lengths = batch["length"].to(accelerator.device, non_blocking=True)
            with torch.no_grad():
                embeddings, target_embeddings, indices = _stage1_features_and_targets(stage1, signal)
            encoded_attention = _encoded_attention(
                lengths, embeddings.shape[1], stride
            )
            mask_positions = _make_span_mask(
                encoded_attention, mask_probability, span_length
            )
            outputs = model(
                embeddings,
                encoded_attention,
                target_embeddings.detach(),
                mask_positions,
            )
            if outputs.loss is None:
                raise RuntimeError("Stage 2 produced no masked target positions.")
            accelerator.backward(outputs.loss)
            accelerator.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clipping", 1.0)))
            optimizer.step()
            step += 1
            progress.update(1)
            if accelerator.is_local_main_process:
                progress.set_postfix(
                    loss=f"{outputs.loss.item():.4f}",
                    smooth=f"{outputs.smooth_l1_loss.item():.4f}",
                    cosine=f"{outputs.cosine_loss.item():.4f}",
                )
                if wandb_run is not None and step % int(cfg["training"].get("log_every_steps", 10)) == 0:
                    selected = mask_positions
                    prediction_norm = outputs.predictions[selected].norm(dim=-1).mean()
                    target_norm = target_embeddings[selected].norm(dim=-1).mean()
                    mask_ratio = selected.sum().float() / encoded_attention.sum().clamp_min(1).float()
                    wandb_run.log(
                        {
                            "train/loss": float(outputs.loss.item()),
                            "train/smooth_l1": float(outputs.smooth_l1_loss.item()),
                            "train/cosine_loss": float(outputs.cosine_loss.item()),
                            "train/mask_ratio": float(mask_ratio.item()),
                            "train/prediction_norm": float(prediction_norm.item()),
                            "train/target_norm": float(target_norm.item()),
                            "train/codebook_usage_ratio": float(
                                torch.unique(indices.detach()).numel() / max(1, stage1.codebook_size)
                            ),
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "step": step,
                        },
                        step=step,
                    )

            if step % eval_every == 0:
                model.eval()
                values = []
                smooth_values = []
                cosine_values = []
                mask_ratios = []
                codebook_usages = []
                with torch.no_grad():
                    for index, valid_batch in enumerate(valid_loader):
                        valid_signal = valid_batch["signal"].to(accelerator.device, non_blocking=True)
                        valid_lengths = valid_batch["length"].to(accelerator.device, non_blocking=True)
                        valid_embeddings, valid_targets, valid_indices = _stage1_features_and_targets(
                            stage1, valid_signal
                        )
                        valid_attention = _encoded_attention(
                            valid_lengths, valid_embeddings.shape[1], stride
                        )
                        valid_mask = _make_span_mask(
                            valid_attention, mask_probability, span_length
                        )
                        result = model(
                            valid_embeddings,
                            valid_attention,
                            valid_targets,
                            valid_mask,
                        )
                        if result.loss is not None:
                            values.append(result.loss.detach())
                            smooth_values.append(result.smooth_l1_loss.detach())
                            cosine_values.append(result.cosine_loss.detach())
                            mask_ratios.append(
                                valid_mask.sum().float()
                                / valid_attention.sum().clamp_min(1).float()
                            )
                            codebook_usages.append(
                                torch.unique(valid_indices.detach()).numel()
                                / max(1, stage1.codebook_size)
                            )
                        if index + 1 >= max_eval_batches:
                            break
                if values:
                    eval_loss = accelerator.gather_for_metrics(torch.stack(values)).mean().item()
                    eval_smooth = accelerator.gather_for_metrics(
                        torch.stack(smooth_values)
                    ).mean().item()
                    eval_cosine = accelerator.gather_for_metrics(
                        torch.stack(cosine_values)
                    ).mean().item()
                    eval_mask_ratio = accelerator.gather_for_metrics(
                        torch.stack(mask_ratios)
                    ).mean().item()
                    eval_codebook_usage = accelerator.gather_for_metrics(
                        torch.tensor(codebook_usages, device=accelerator.device)
                    ).mean().item()
                    if accelerator.is_local_main_process:
                        accelerator.print(f"step={step} valid_loss={eval_loss:.6f}")
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "valid/loss": float(eval_loss),
                                    "valid/smooth_l1": float(eval_smooth),
                                    "valid/cosine_loss": float(eval_cosine),
                                    "valid/mask_ratio": float(eval_mask_ratio),
                                    "valid/codebook_usage_ratio": float(eval_codebook_usage),
                                    "step": step,
                                },
                                step=step,
                            )

            if step % save_every == 0:
                _save_checkpoint(accelerator, model, output_dir, step, cfg)
            if step >= max_steps:
                break
    _save_checkpoint(accelerator, model, output_dir, step, cfg)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
