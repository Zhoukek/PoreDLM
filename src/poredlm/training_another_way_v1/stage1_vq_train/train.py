"""Train Stage 1: raw signal -> CNN -> VQ -> signal reconstruction.

Run from the repository root with:

    python -m poredlm.training_another_way_v1.stage1_vq_train.train \
        --config src/poredlm/training_another_way_v1/stage1_vq_train/config.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

from poredlm.training_another_way_v1.data import PoreSignalDataset
from poredlm.training_another_way_v1.stage1_vq_train.modeling import (
    PoreVQCodec,
    PoreVQCodecConfig,
)


def _breakdown_value(breakdown, name: str, device: torch.device) -> torch.Tensor:
    value = getattr(breakdown, name, None)
    if value is None:
        return torch.zeros((), device=device)
    if not isinstance(value, torch.Tensor):
        return torch.tensor(float(value), device=device)
    return value


def _masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    signal: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    prediction = reconstruction.squeeze(1)
    valid = attention_mask.bool()
    if not valid.any():
        return F.smooth_l1_loss(prediction, signal)
    return F.smooth_l1_loss(prediction[valid], signal[valid])


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
    loader_kwargs = {
        "batch_size": int(cfg["training"].get("device_micro_batch_size", 8)),
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", False)) and workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


def _make_model(cfg: dict) -> PoreVQCodec:
    model_cfg = cfg["model"]
    vq_cfg = model_cfg.get("vq", {})
    config = PoreVQCodecConfig(
        codebook_size=int(model_cfg.get("codebook_size", 1024)),
        codebook_decay=float(vq_cfg.get("codebook_decay", 0.99)),
        codebook_emadc=int(vq_cfg.get("codebook_emadc", 2)),
        commitment_weight=float(cfg["loss_weights"].get("commitment_weight", 1.0)),
        orthogonal_reg_weight=float(cfg["loss_weights"].get("orthogonal_reg_weight", 0.0)),
        codebook_diversity_loss_weight=float(
            cfg["loss_weights"].get("codebook_diversity_loss_weight", 0.0)
        ),
        cnn_type=int(vq_cfg.get("cnn_type", 0)),
        learnable_codebook=bool(vq_cfg.get("learnable_codebook", False)),
        init_codebook_path=vq_cfg.get("init_codebook_path"),
        freeze_cnn=bool(vq_cfg.get("freeze_cnn", False)),
        cnn_checkpoint_path=vq_cfg.get("cnn_checkpoint_path"),
    )
    return PoreVQCodec(config)


def _compute_loss(model, batch: dict[str, torch.Tensor], cfg: dict):
    signal = batch["signal"]
    attention_mask = batch["attention_mask"]
    reconstruction, indices, vq_loss, breakdown, _ = model(signal)
    recon_loss = _masked_reconstruction_loss(reconstruction, signal, attention_mask)
    commitment = _breakdown_value(breakdown, "commitment", signal.device)
    diversity = _breakdown_value(breakdown, "codebook_diversity", signal.device)
    orthogonal = _breakdown_value(breakdown, "orthogonal_reg", signal.device)

    weights = cfg["loss_weights"]
    loss = (
        recon_loss
        + float(weights.get("commitment_weight", 1.0)) * commitment
        + float(weights.get("codebook_diversity_loss_weight", 0.0)) * diversity
        + float(weights.get("orthogonal_reg_weight", 0.0)) * orthogonal
    )
    base_model = model.module if hasattr(model, "module") else model
    metrics = {
        "loss": loss.detach(),
        "recon_loss": recon_loss.detach(),
        "commitment_loss": commitment.detach(),
        "diversity_loss": diversity.detach(),
        "orthogonal_loss": orthogonal.detach(),
        "used_code_ratio": torch.unique(indices.detach()).numel()
        / max(1, int(base_model.codebook_size)),
    }
    return loss, metrics


def _save_checkpoint(accelerator, model, output_dir: Path, step: int, cfg: dict) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    save_dir = output_dir / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_dir, safe_serialization=True)
    (save_dir / "training_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    torch.save({"global_step": step}, save_dir / "trainer_state.pt")
    latest = output_dir / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(save_dir, latest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    accelerator = Accelerator(mixed_precision=cfg["training"].get("mixed_precision", "no"))
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
            "project": wandb_cfg.get("project", "training_another_way_v1_stage1_vq"),
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

    model = _make_model(cfg).to(accelerator.device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    train_loader = _build_loader(cfg, "train", train=True)
    valid_loader = _build_loader(cfg, "valid", train=False)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["training"].get("max_train_steps", 100000))
    eval_every = int(cfg["training"].get("eval_every_steps", 1000))
    save_every = int(cfg["training"].get("save_every_steps", 5000))
    max_eval_batches = int(cfg["training"].get("max_eval_batches", 20))
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)

    step = 0
    while step < max_steps:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(accelerator.device, non_blocking=True) for key, value in batch.items()}
            loss, metrics = _compute_loss(model, batch, cfg)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clipping", 1.0)))
            optimizer.step()
            step += 1
            progress.update(1)
            if accelerator.is_local_main_process:
                progress.set_postfix(
                    loss=f"{metrics['loss'].item():.4f}",
                    recon=f"{metrics['recon_loss'].item():.4f}",
                    used=f"{metrics['used_code_ratio']:.3f}",
                )
                if wandb_run is not None and step % int(cfg["training"].get("log_every_steps", 10)) == 0:
                    wandb_run.log(
                        {
                            "train/loss": float(metrics["loss"].item()),
                            "train/reconstruction_loss": float(metrics["recon_loss"].item()),
                            "train/commitment_loss": float(metrics["commitment_loss"].item()),
                            "train/diversity_loss": float(metrics["diversity_loss"].item()),
                            "train/orthogonal_loss": float(metrics["orthogonal_loss"].item()),
                            "train/codebook_usage_ratio": float(metrics["used_code_ratio"]),
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "step": step,
                        },
                        step=step,
                    )

            if step % eval_every == 0:
                model.eval()
                values = []
                with torch.no_grad():
                    for index, valid_batch in enumerate(valid_loader):
                        valid_batch = {
                            key: value.to(accelerator.device, non_blocking=True)
                            for key, value in valid_batch.items()
                        }
                        valid_loss, _ = _compute_loss(model, valid_batch, cfg)
                        values.append(valid_loss.detach())
                        if index + 1 >= max_eval_batches:
                            break
                if values:
                    eval_loss = accelerator.gather_for_metrics(torch.stack(values)).mean().item()
                    if accelerator.is_local_main_process:
                        accelerator.print(f"step={step} valid_loss={eval_loss:.6f}")
                        if wandb_run is not None:
                            wandb_run.log({"valid/loss": float(eval_loss), "step": step}, step=step)

            if step % save_every == 0:
                _save_checkpoint(accelerator, model, output_dir, step, cfg)
            if step >= max_steps:
                break
    _save_checkpoint(accelerator, model, output_dir, step, cfg)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
