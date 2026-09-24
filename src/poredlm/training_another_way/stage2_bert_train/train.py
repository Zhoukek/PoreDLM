"""Train continuous BERT with online features from a frozen CNN."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm

from modeling_continuous_bert import ContinuousBert, ContinuousBertConfig
from poredlm.training_another_way.stage1_cnn_train.dataset import PoreSignalDataset
from poredlm.training_another_way.stage1_cnn_train.modeling_continuous_cnn import ContinuousSignalCNN


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_loader(cfg: dict, split: str, train: bool):
    data = cfg["data"][split]
    paths = data.get("paths", data.get("directory"))
    if not paths:
        raise ValueError(f"data.{split}.paths must point to the raw Stage 1 signal files.")
    dataset = PoreSignalDataset(
        paths,
        chunk_size=data.get("chunk_size", 6000),
        memmap_dtype=data.get("memmap_dtype", "float32"),
        buffer_size=data.get("buffer_size", 20000 if train else 0),
        shuffle_buffer=train,
        repeat=train,
        seed=cfg.get("seed", 42),
    )
    workers = int(data.get("num_workers", 0))
    batch_size = cfg["training"].get("device_micro_batch_size", cfg["training"].get("device_train_microbatch_size", 8))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       num_workers=workers, pin_memory=data.get("pin_memory", True),
                                       persistent_workers=data.get("persistent_workers", False) and workers > 0,
                                       prefetch_factor=data.get("prefetch_factor", 2) if workers > 0 else None)


def make_mask(attention_mask: torch.Tensor, probability: float) -> torch.Tensor:
    mask = torch.rand(attention_mask.shape, device=attention_mask.device) < probability
    mask = mask & attention_mask.bool()
    # Keep at least one valid prediction position per non-empty sample.
    for row in range(mask.shape[0]):
        valid = torch.nonzero(attention_mask[row].bool(), as_tuple=False).flatten()
        if valid.numel() and not mask[row].any():
            mask[row, valid[torch.randint(valid.numel(), (1,), device=mask.device)]] = True
    return mask


def save_hf_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: str | os.PathLike[str],
    step_name: str,
    global_step: int,
    config: dict,
) -> None:
    """Save a Stage 2 checkpoint in the same HF directory style as Stage 1."""
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    save_dir = Path(output_dir) / str(step_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.config.architectures = ["ContinuousBert"]
    unwrapped.config.auto_map = {
        "AutoConfig": "modeling_continuous_bert.ContinuousBertConfig",
        "AutoModel": "modeling_continuous_bert.ContinuousBert",
    }
    unwrapped.save_pretrained(save_dir, safe_serialization=True)
    shutil.copy2(Path(__file__).with_name("modeling_continuous_bert.py"), save_dir / "modeling_continuous_bert.py")
    with (save_dir / "training_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    torch.save({"global_step": int(global_step), "model_format": "hf_pretrained"}, save_dir / "trainer_state.pt")

    latest_dir = Path(output_dir) / "latest"
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_symlink() or latest_dir.is_file():
            latest_dir.unlink()
        else:
            shutil.rmtree(latest_dir)
    shutil.copytree(save_dir, latest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    print(
        f"rank={os.environ.get('RANK', '0')} "
        f"local_rank={local_rank} "
        f"world_size={os.environ.get('WORLD_SIZE', '1')} "
        f"device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    seed_everything(int(cfg.get("seed", 42)))
    accelerator = Accelerator(mixed_precision=cfg["training"].get("mixed_precision", "no"))
    cnn_checkpoint = cfg.get("cnn", {}).get("checkpoint")
    if not cnn_checkpoint:
        raise ValueError("cnn.checkpoint must point to a Stage 1 HF checkpoint directory.")
    cnn = ContinuousSignalCNN.from_pretrained(cnn_checkpoint).to(accelerator.device)
    cnn.eval()
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    if cnn.hidden_size != int(cfg["model"]["feature_dim"]):
        raise ValueError(
            f"Stage 1 CNN output dim ({cnn.hidden_size}) does not match "
            f"BERT feature_dim ({cfg['model']['feature_dim']})."
        )
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
            "project": wandb_cfg.get("project", "continuous_bert"),
            "config": cfg,
            "mode": os.environ.get("WANDB_MODE", "online"),
        }
        if wandb_cfg.get("entity"):
            init_kwargs["entity"] = wandb_cfg["entity"]
        if wandb_cfg.get("name"):
            init_kwargs["name"] = wandb_cfg["name"]
        wandb_run = wandb.init(**init_kwargs)
    model = ContinuousBert(ContinuousBertConfig(**cfg["model"]))
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"].get("weight_decay", 0.01))
    train_loader = build_loader(cfg, "train", True); valid_loader = build_loader(cfg, "valid", False)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    output_dir = Path(cfg["training"]["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["training"]["max_train_steps"]); step = 0
    progress = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    while step < max_steps:
        for batch in train_loader:
            model.train(); optimizer.zero_grad(set_to_none=True)
            signals = batch["signal"].to(accelerator.device, non_blocking=True)
            with torch.no_grad():
                embeddings = cnn.encode(signals).transpose(1, 2)
            attention = torch.ones(
                embeddings.shape[:2], dtype=torch.long, device=embeddings.device
            )
            positions = make_mask(attention, cfg["model"].get("mask_probability", 0.15))
            outputs = model(embeddings, attention, embeddings.detach(), positions)
            accelerator.backward(outputs.loss)
            accelerator.clip_grad_norm_(model.parameters(), cfg["training"].get("gradient_clipping", 1.0))
            optimizer.step(); step += 1; progress.update(1)
            if accelerator.is_local_main_process and step % cfg["training"].get("log_every_steps", 10) == 0:
                progress.set_postfix(loss=f"{outputs.loss.item():.5f}")
                if wandb_run is not None:
                    wandb_run.log({"train/loss": outputs.loss.item()}, step=step)
            if step % cfg["training"].get("eval_every_steps", 1000) == 0:
                model.eval(); values = []
                with torch.no_grad():
                    for index, valid_batch in enumerate(valid_loader):
                        valid_signals = valid_batch["signal"].to(accelerator.device, non_blocking=True)
                        with torch.no_grad():
                            valid_embeddings = cnn.encode(valid_signals).transpose(1, 2)
                        valid_attention = torch.ones(
                            valid_embeddings.shape[:2], dtype=torch.long, device=valid_embeddings.device
                        )
                        valid_mask = make_mask(valid_attention, cfg["model"].get("mask_probability", 0.15))
                        result = model(valid_embeddings, valid_attention, valid_embeddings, valid_mask)
                        if result.loss is not None: values.append(result.loss)
                        if index + 1 >= cfg["training"].get("max_eval_batches", 20): break
                loss = torch.stack(values).mean() if values else torch.tensor(0.0, device=accelerator.device)
                loss = accelerator.gather_for_metrics(loss.unsqueeze(0)).mean().item()
                if accelerator.is_local_main_process: accelerator.print(f"step={step} eval_loss={loss:.6f}")
                if wandb_run is not None:
                    wandb_run.log({"valid/loss": loss}, step=step)
            if step % cfg["training"].get("save_every_steps", 5000) == 0:
                save_hf_checkpoint(accelerator, model, output_dir, f"step_{step}", step, cfg)
            if step >= max_steps: break
    save_hf_checkpoint(accelerator, model, output_dir, "final", step, cfg)
    if wandb_run is not None:
        wandb_run.finish()
    progress.close()


if __name__ == "__main__": main()
