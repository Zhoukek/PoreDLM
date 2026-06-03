# -*- coding: utf-8 -*-
"""
train_ddp_multifolder.py

在你原 train_ddp.py 基础上，最小改动实现 jsonl.gz 输入 + 自动 split + wandb。

关键：保持原数据流不变
- Dataset 仍然返回 signal_str（字符串），由 BasecallModel.tokenizer 编码成 input_ids
- reference 仍按 ref_row[ref_row>0] 取 labels

新增：
- --jsonl_paths 多文件夹/多文件输入
- 自动 split（--train_ratio/--val_ratio/--test_ratio，按 folder 或 file group）
- wandb（仅 rank0）
- CTC 使用 batch["input_lengths"]（来自 tokenizer attention_mask）
- ✅ 新增 checkpoint 保存/恢复：
  - 每 epoch 保存 ckpt_last.pt
  - val_acc 最优保存 ckpt_best.pt
  - 支持 --resume_ckpt 恢复 model/optimizer/scheduler/epoch/best 指标
"""

import os
import socket
import math
import argparse
import matplotlib.pyplot as plt
import logging
from contextlib import nullcontext
from datetime import timedelta
from typing import Tuple, Optional, Any, Dict, List
import numpy as np

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from Basecalling.basecaller_v8_0420.utils import seed_everything, BLANK_IDX, ID2BASE, resolve_input_lengths
from Basecalling.basecaller_v8_0420.model import BasecallModel
from Basecalling.basecaller_v8_0420.metrics import (
    ctc_viterbi_decode,
    koi_beam_search_decode,
    batch_bonito_accuracy,
    plot_curves,
    save_metrics_csv,
)
from Basecalling.basecaller_v8_0420.ctc_crf import decode as ctc_crf_decode
from Basecalling.basecaller_v8_0420.ctc_crf import ctc_crf_loss
from Basecalling.basecaller_v8_0420.ctc import ctc_label_smoothing_loss
from Basecalling.basecaller_v8_0420.data_multifolder import (
    scan_jsonl_files,
    split_jsonl_files_by_group,
    MultiJsonlSignalRefDataset,
    scan_npy_pairs,
    split_npy_pairs_by_group,
    split_indices,
    MultiNpySignalRefDataset,
    create_collate_fn,
    create_vq_collate_fn,
)
from Basecalling.basecaller_v8_0420.callback import plot_alignment_heatmap

try:
    import wandb
except Exception:
    wandb = None


# -------------------- distributed helpers --------------------

def gpu_socket_preflight(backend: str) -> Tuple[bool, Optional[str], Optional[str]]:
    backend_name = str(backend).lower()
    socket_env_name = "NCCL_SOCKET_IFNAME"
    if os.environ.get(socket_env_name):
        iface_names = {name for _, name in socket.if_nameindex()}
        raw_expr = os.environ[socket_env_name]
        requested = [x.strip() for x in raw_expr.split(",") if x.strip()]
        normalized_iface_expr = ",".join(requested) if requested else None

        advanced_rules = [rule for rule in requested if rule.startswith("=") or rule.startswith("^")]
        if advanced_rules:
            return False, (
                f"{socket_env_name}={raw_expr!r} uses advanced {backend_name.upper()} filter syntax {advanced_rules!r}; "
                "falling back to gloo because this environment cannot validate or normalize it safely. "
                f"Use a plain visible interface name such as 'eth0' instead (visible={sorted(iface_names)})."
            ), None

        matched = [name for name in requested if name in iface_names]
        if matched:
            return True, None, ",".join(matched)
        return False, (
            f"{socket_env_name}={raw_expr!r} does not match any visible network interface "
            f"(visible={sorted(iface_names)})"
        ), normalized_iface_expr

    iface_names = [name for _, name in socket.if_nameindex()]
    non_loopback = [name for name in iface_names if name != "lo" and not name.startswith("lo:")]
    if non_loopback:
        return True, None, non_loopback[0]
    return False, f"no non-loopback network interface is visible (found: {iface_names or ['<none>']})", None


def is_backend_available(backend: str) -> bool:
    backend_name = str(backend).lower()
    if backend_name == "nccl":
        return bool(hasattr(dist, "is_nccl_available") and dist.is_nccl_available())
    if backend_name == "gloo":
        return bool(hasattr(dist, "is_gloo_available") and dist.is_gloo_available())
    return False


def resolve_distributed_backend(args) -> Tuple[str, Optional[str]]:
    ddp_env = ("RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1)
    backend = args.ddp_backend
    fallback_allowed = bool(args.ddp_backend_fallback)
    info_message: Optional[str] = None
    if ddp_env and backend == "nccl" and fallback_allowed:
        socket_ok, socket_reason, normalized_iface_expr = gpu_socket_preflight(backend)
        socket_env_name = "NCCL_SOCKET_IFNAME"
        if not socket_ok:
            info_message = f"[Accelerate] {backend.upper()} preflight failed: {socket_reason}. Falling back to gloo before Accelerator initialization."
            backend = "gloo"
        elif normalized_iface_expr and normalized_iface_expr != os.environ.get(socket_env_name):
            original_iface_expr = os.environ.get(socket_env_name)
            os.environ[socket_env_name] = normalized_iface_expr
            info_message = (
                f"[Accelerate] Normalizing {socket_env_name} from "
                f"{original_iface_expr!r} to {normalized_iface_expr!r} for {backend.upper()} compatibility."
            )
    return backend, info_message

def is_main_process(accelerator: Accelerator) -> bool:
    return accelerator.is_main_process


def reduce_mean(accelerator: Accelerator, value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    return float(accelerator.gather_for_metrics(tensor.unsqueeze(0)).mean().item())


def reduce_min(accelerator: Accelerator, value: int, device: torch.device) -> bool:
    tensor = torch.tensor(int(value), device=device, dtype=torch.int)
    return bool(accelerator.gather(tensor.unsqueeze(0)).min().item())


def setup_logger(log_file: str, accelerator: Accelerator) -> logging.Logger:
    logger = logging.getLogger("basecaller_ddp_multifolder")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    if not is_main_process(accelerator):
        logger.addHandler(logging.NullHandler())
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# -------------------- optimizer/scheduler --------------------

def build_adamw_with_no_decay(named_params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    decay_params, no_decay_params = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr: float,
                    logger: Optional[logging.Logger], accelerator: Accelerator):
    """
    Linear warmup + cosine decay with a non-zero floor (min_lr).
    This keeps behavior stable regardless of transformers version and honors --min_lr.
    """
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    base_lr = float(optimizer.param_groups[0]["lr"])
    if base_lr <= 0:
        base_lr = 1e-8
    min_lr = max(0.0, float(min_lr))
    min_ratio = min(min_lr / base_lr, 1.0)

    def lr_lambda(current_step: int) -> float:
        step = min(max(int(current_step), 0), total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if logger is not None and is_main_process(accelerator):
        logger.info(f"[Scheduler] Using LambdaLR linear_warmup+cosine with min_lr floor (base_lr={base_lr:.6g}, min_lr={min_lr:.6g})")
    return sched, "lambda_warmup_cosine_minlr"


# -------------------- checkpoint helpers --------------------

def get_raw_model(model):
    if isinstance(model, Accelerator):
        raise TypeError("Pass a model instance, not an Accelerator, to get_raw_model().")
    return model.module if hasattr(model, "module") else model


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(path: str,
                    accelerator: Accelerator,
                    model,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    epoch: Optional[int] = None,
                    best_pbma: Optional[float] = None,
                    extra: Optional[Dict[str, Any]] = None):
    raw = accelerator.unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "best_pbma": best_pbma,
        "model_state_dict": raw.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)
    accelerator.save(ckpt, path)


def load_checkpoint(path: str,
                    accelerator: Accelerator,
                    model,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    map_location: str = "cpu",
                    logger: Optional[logging.Logger] = None):
    if logger is not None:
        logger.info(f"[Resume] loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=map_location)

    # state_dict (strip possible module.)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    raw = accelerator.unwrap_model(model)
    missing, unexpected = raw.load_state_dict(new_state, strict=False)

    if logger is not None:
        logger.info(f"[Resume] load model: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            logger.info(f"[Resume] missing keys (first 20): {missing[:20]}")
        if unexpected:
            logger.info(f"[Resume] unexpected keys (first 20): {unexpected[:20]}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if logger is not None:
            logger.info("[Resume] optimizer state loaded")

    if scheduler is not None and "scheduler_state_dict" in ckpt and hasattr(scheduler, "load_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if logger is not None:
                logger.info("[Resume] scheduler state loaded")
        except Exception as e:
            if logger is not None:
                logger.warning(f"[Resume] scheduler load failed: {e}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_pbma = ckpt.get("best_pbma", None)
    return start_epoch, best_pbma


# -------------------- train/eval --------------------

def train_one_epoch(
    accelerator: Accelerator,
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
    log_interval: int,
    use_wandb: bool,
    ctc_crf_blank_score: float,
    use_amp: bool,
    clip_grad_norm: float,
    head_type: str,
):
    model.train()
    total_loss, n_batches = 0.0, 0

    it = tqdm(enumerate(data_loader, start=1), total=len(data_loader),
              disable=not is_main_process(accelerator), desc="[train]")
    for step, batch in it:
        input_ids = batch["input_ids"].to(device)
        input_lengths = resolve_input_lengths(
            input_ids,
            attention_mask=batch.get("attention_mask"),
            input_lengths=batch.get("input_lengths"),
        )

        target_labels = batch["target_labels"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with accelerator.autocast() if use_amp else nullcontext():
            logits_btc = model(input_ids, attention_mask=attention_mask)
            logits_tbc = logits_btc.transpose(0, 1)    # [T,B,C]
            if head_type == "ctc_crf":
                loss = ctc_crf_loss(
                    logits_tbc,
                    target_labels,
                    input_lengths,
                    target_lengths,
                    blank_idx=BLANK_IDX,
                )
            else:
                ctc_loss_dict = ctc_label_smoothing_loss(
                    logits_tbc,
                    target_labels,
                    target_lengths,
                    blank_idx=BLANK_IDX,
                )
                loss = ctc_loss_dict["total_loss"]
        local_finite = torch.tensor(
            1 if torch.isfinite(loss).item() else 0,
            device=device,
            dtype=torch.int,
        )
        global_finite = reduce_min(accelerator, int(local_finite.item()), device)

        if global_finite:
            accelerator.backward(loss)
            if clip_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        elif is_main_process(accelerator):
            print(f"[Train] step={step}/{len(data_loader)} non-finite loss detected on >=1 rank; skipping optimizer step on all ranks.")

        total_loss += float(loss.item())
        n_batches += 1

        if is_main_process(accelerator) and (step % log_interval == 0):
            lr = optimizer.param_groups[0]["lr"]
            msg = f"[Train] step={step}/{len(data_loader)} loss={loss.item():.4f} lr={lr:.6g}"
            print(msg)
            if use_wandb and wandb is not None:
                wandb.log({"train/loss": float(loss.item()), "lr": float(lr), "step": step})

    avg = total_loss / max(n_batches, 1)
    return reduce_mean(accelerator, avg, device)


@torch.no_grad()
def eval_one_epoch(
    accelerator: Accelerator,
    model,
    data_loader,
    device,
    split_name: str,
    ctc_crf_blank_score: float,
    koi_blank_score: float,
    acc_balanced: bool,
    acc_min_coverage: float,
    use_amp: bool,
    decoder_mode: str,
    head_type: str,
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss, n_batches = 0.0, 0
    total_acc, n_acc = 0.0, 0
    total_crf_acc, n_crf_acc = 0.0, 0
    total_cov, n_cov = 0.0, 0
    blank_ratios: List[float] = []
    nonzero_lengths: List[float] = []

    it = tqdm(data_loader, total=len(data_loader),
              disable=not is_main_process(accelerator), desc=f"[{split_name}]")
    for batch in it:
        input_ids = batch["input_ids"].to(device)
        input_lengths = resolve_input_lengths(
            input_ids,
            attention_mask=batch.get("attention_mask"),
            input_lengths=batch.get("input_lengths"),
        )

        target_labels = batch["target_labels"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with accelerator.autocast() if use_amp else nullcontext():
            logits_btc = model(input_ids, attention_mask=attention_mask)

        # logits_btc = model(input_ids)              # [B,T,C]
        logits_tbc = logits_btc.transpose(0, 1)    # [T,B,C]

        if head_type == "ctc_crf":
            loss = ctc_crf_loss(
                logits_tbc,
                target_labels,
                input_lengths,
                target_lengths,
                blank_idx=BLANK_IDX,
            )
        else:
            ctc_loss_dict = ctc_label_smoothing_loss(
                logits_tbc,
                target_labels,
                target_lengths,
                blank_idx=BLANK_IDX,
            )
            loss = ctc_loss_dict["total_loss"]
        
        total_loss += float(loss.item())
        n_batches += 1
        input_len_list = input_lengths.detach().cpu().tolist()
        if decoder_mode == "koi":
            pred_seqs = koi_beam_search_decode(
                logits_tbc,
                blank_score=float(koi_blank_score),
                input_lengths=input_lengths,
            )
        elif decoder_mode == "ctc_viterbi":
            pred_seqs = ctc_viterbi_decode(
                logits_tbc,
                input_lengths=input_lengths,
                blank_idx=BLANK_IDX,
            )
        else:
            pred_seqs = []
            for idx, input_len in enumerate(input_len_list):
                step_len = int(input_len)
                if step_len <= 0:
                    pred_seqs.append([])
                    continue
                decoded_ids = ctc_crf_decode(
                    logits_tbc[:step_len, idx : idx + 1, :].float(),
                    blank_idx=BLANK_IDX,
                )[0]
                pred_seqs.append(decoded_ids[:step_len])

        acc = batch_bonito_accuracy(
            pred_seqs,
            batch["target_seqs"],
            balanced=acc_balanced,
            min_coverage=acc_min_coverage,
        )
        total_acc += float(acc)
        n_acc += 1

        for r_ids, pred_ids, input_len in zip(batch["target_seqs"], pred_seqs, input_len_list):
            step_len = int(input_len)
            if step_len <= 0:
                blank_ratios.append(1.0)
                nonzero_lengths.append(0.0)
                total_cov += 0.0
                n_cov += 1
                continue
            decoded_len = min(len(pred_ids), step_len)
            blank_ratio = max(1.0 - (decoded_len / max(step_len, 1)), 0.0)
            blank_ratios.append(blank_ratio)
            nonzero_len = float(decoded_len)
            nonzero_lengths.append(nonzero_len)
            ref_len = max(len(r_ids), 1)
            total_cov += nonzero_len / ref_len
            n_cov += 1

        total_crf_acc += float(acc) if decoder_mode == "ctc_crf" else 0.0
        n_crf_acc += 1 if decoder_mode == "ctc_crf" else 0

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = total_acc / max(n_acc, 1)
    avg_crf_acc = total_crf_acc / max(n_crf_acc, 1)
    avg_cov = total_cov / max(n_cov, 1)
    avg_blank = float(np.mean(blank_ratios)) if blank_ratios else 0.0
    avg_nonzero_len = float(np.mean(nonzero_lengths)) if nonzero_lengths else 0.0

    avg_loss = reduce_mean(accelerator, avg_loss, device)
    avg_acc = reduce_mean(accelerator, avg_acc, device)
    avg_crf_acc = reduce_mean(accelerator, avg_crf_acc, device)
    avg_cov = reduce_mean(accelerator, avg_cov, device)
    avg_blank = reduce_mean(accelerator, avg_blank, device)
    avg_nonzero_len = reduce_mean(accelerator, avg_nonzero_len, device)
    return avg_loss, avg_acc, avg_crf_acc, avg_cov, avg_blank, avg_nonzero_len


# -------------------- pretrained loader (keep) --------------------

def load_pretrained_weights(accelerator: Accelerator, model, ckpt_path: str, strict: bool = False,
                            key: str | None = None, logger: Optional[logging.Logger] = None):
    if ckpt_path is None:
        return

    if logger is not None:
        logger.info(f"[Pretrained] loading from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(ckpt, dict):
        if key is not None and key in ckpt:
            state = ckpt[key]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            if all(isinstance(k, str) for k in ckpt.keys()):
                state = ckpt
    else:
        state = ckpt

    if state is None:
        raise ValueError(f"Cannot find a state_dict in checkpoint: {ckpt_path}. Try --pretrained_key.")

    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    target = accelerator.unwrap_model(model)
    missing, unexpected = target.load_state_dict(new_state, strict=strict)

    if logger is not None:
        logger.info(f"[Pretrained] strict={strict} missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            logger.info(f"[Pretrained] missing keys (first 20): {missing[:20]}")
        if unexpected:
            logger.info(f"[Pretrained] unexpected keys (first 20): {unexpected[:20]}")


# -------------------- args --------------------

def _parse_folder_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders (uses text/bases fields).")
    p.add_argument("--train_jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders for training set (skip auto split).")
    p.add_argument("--val_jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders for validation set (skip auto split).")
    p.add_argument("--test_jsonl_paths", type=str, default=None,
                   help="Comma-separated .jsonl.gz files or folders for test set (skip auto split).")
    p.add_argument("--npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files (uses token/reference pairs).")
    p.add_argument("--train_npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files for training set.")
    p.add_argument("--val_npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files for validation set.")
    p.add_argument("--test_npy_paths", type=str, default=None,
                   help="Comma-separated folders or tokens_*.npy/reference_*.npy files for test set.")
    p.add_argument("--group_by", type=str, default="folder", choices=["folder", "file", "record"],
                   help="Auto split granularity: folder/file keeps groups together; record shuffles all reads across files before split.")
    p.add_argument("--recursive", action="store_true",
                   help="Scan subfolders for .jsonl.gz or tokens/reference .npy inputs.")
    p.add_argument("--token_offset", type=int, default=0,
                   help="Add this offset to each <|bwav:ID|> token in input signal_str (e.g. 0->128).")

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Backbone model path for hidden/embedding; VQ tokenizer checkpoint path for vq_embedding.")
    p.add_argument("--hidden-layer",type=int,default=-1, help="Which backbone hidden layer to use when --feature_source hidden (-1=last, -2=second last, etc.)")
    p.add_argument("--learnable_fuse_last_n_layers", type=int, default=0,
                   help="If >0, learn a softmax-weighted fusion over the last N hidden layers (overrides --hidden-layer).")
    p.add_argument("--feature_source", "--feature-source", choices=["hidden", "embedding", "vq_embedding"], default="hidden",
                   help="Use backbone hidden states / backbone input embedding / tokenize-model VQ codebook embedding.")
    p.add_argument("--vq_device", type=str, default="cuda",
                   help="Device used when loading VQETokenizer for --feature_source vq_embedding.")
    p.add_argument("--vq_token_batch_size", type=int, default=100,
                   help="token_batch_size used when loading VQETokenizer for --feature_source vq_embedding.")


    p.add_argument("--pretrained_ckpt", type=str, default=None,
                   help="Optional path to a .pt/.pth checkpoint to load into the model (state_dict or dict containing one).")
    p.add_argument("--pretrained_strict", action="store_true",
                   help="Use strict=True when loading pretrained weights (default strict=False).")
    p.add_argument("--pretrained_key", type=str, default=None,
                   help="If checkpoint is a dict, optionally choose the key that contains the model state_dict (e.g. model/state_dict).")

    # ✅ resume from a full checkpoint (model+optim+sched+epoch)
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to ckpt_last.pt to resume training (restores model/optimizer/scheduler/epoch/best_acc).")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode alias: freeze backbone + ctc_crf_state_len=5 + ctc_crf_blank_score=0 + head_output_scale=5 + head_output_activation=tanh + head_type=ctc_crf + pre_ctc_module=none.")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--min_lr", type=float, default=1e-5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="./outputs_ddp")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="basecaller")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None,
                   help="Optional W&B group name (useful for grouping condition sweeps).")
    p.add_argument("--wandb_job_type", type=str, default="train",
                   help="W&B job_type for this run.")

    p.add_argument("--find_unused_parameters", action="store_true",
                   help="Enable DDP unused parameter detection (fix reduction error).")
    p.add_argument("--ddp_backend", type=str, default="gloo", choices=["nccl", "gloo"],
                   help="Distributed backend selection. Explicitly choose NCCL or GLOO for the current runtime.")
    p.add_argument("--ddp_backend_fallback", action="store_true",
                   help="Allow fallback from NCCL to GLOO if the selected GPU backend init fails.")
    p.add_argument("--ddp_broadcast_buffers", action="store_true",
                   help="Enable DDP per-forward buffer broadcast. Keep off by default to reduce desync-related NCCL broadcast stalls.")
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed precision (AMP) training on CUDA.")

    # ✅ save frequency controls (minimal)
    p.add_argument("--save_every", type=int, default=1,
                   help="Save ckpt_last.pt every N epochs (default 1).")
    p.add_argument("--save_best", action="store_true",
                   help="Save ckpt_best.pt based on best val_acc (default off unless provided).")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze backbone parameters; train only base_head.")
    p.add_argument("--reset_backbone_weights", action="store_true",
                   help="Reinitialize backbone weights for ablation (ignores pretrained backbone init).")
    p.add_argument("--unfreeze_last_n_layers", type=int, default=0,
                   help="Unfreeze only the last N backbone layers (default: 0).")
    p.add_argument("--unfreeze_layer_start", type=int, default=None,
                   help="Unfreeze backbone layers in [start, end). Optional finer control.")
    p.add_argument("--unfreeze_layer_end", type=int, default=None,
                   help="Unfreeze backbone layers in [start, end). Optional finer control.")

    p.add_argument("--head_output_activation", choices=["tanh", "relu"], default=None,
                   help="Optional activation applied to head output logits.")
    p.add_argument("--head_output_scale", type=float, default=None,
                   help="Optional scalar applied to head output logits (after activation).")
    p.add_argument("--head_type", choices=["ctc", "ctc_crf"], default="ctc_crf",
                   help="Head type: plain CTC linear head or CTC-CRF head.")
    p.add_argument("--pre_head_type", choices=["none", "bilstm", "transformer", "tcn"], default="none",
                   help="Optional module before CTC-CRF head.")
    p.add_argument("--pre_ctc_module", dest="pre_head_type", choices=["none", "bilstm", "transformer", "tcn"],
                   help="Alias of --pre_head_type for selecting module before CTC/CTC-CRF head.")
    p.add_argument("--pre_head_transformer_nhead", type=int, default=8,
                   help="Attention heads for --pre_head_type transformer.")

    p.add_argument("--acc_balanced", action="store_true",
                   help="Use Bonito balanced accuracy: (match - ins) / (match + mismatch + del).")
    p.add_argument("--acc_min_coverage", type=float, default=0.0,
                   help="Minimum reference coverage required to count a read for accuracy.")
    p.add_argument("--train_decoder", choices=["ctc_viterbi", "ctc_crf", "koi"], default="ctc_crf",
                   help="Decoder used for accuracy/blank metrics.")
    p.add_argument("--ctc_crf_state_len", type=int, default=5,
                   help="State length for Bonito CTC-CRF (used to set output classes).")
    p.add_argument("--ctc_crf_blank_score", type=float, default=2.0,
                   help="Fixed blank score for CTC-CRF (blank is not trained).")
    p.add_argument("--koi_blank_score", type=float, default=2.0,
                   help="Blank score used by koi beam search decoder.")
    p.add_argument("--clip_grad_norm", type=float, default=2.0,
                   help="Clip gradient norm before optimizer step (0 to disable).")


    return p.parse_args()


# -------------------- main --------------------

def apply_quick_overrides(args) -> None:
    if not args.quick:
        return
    args.freeze_backbone = True
    args.ctc_crf_state_len = 5
    args.ctc_crf_blank_score = 0.0
    args.koi_blank_score = 2.0
    args.head_output_scale = 5.0
    args.head_output_activation = "tanh"
    args.head_type = "ctc_crf"
    args.pre_head_type = "none"


def main():
    args = parse_args()
    if args.token_offset < 0:
        raise ValueError("--token_offset must be >= 0")
    apply_quick_overrides(args)
    backend, backend_note = resolve_distributed_backend(args)
    ddp_kwargs = DistributedDataParallelKwargs(
        # find_unused_parameters=bool(args.find_unused_parameters),
        find_unused_parameters=True,
        broadcast_buffers=bool(args.ddp_broadcast_buffers),
    )
    init_pg_kwargs = InitProcessGroupKwargs(
        backend=backend,
        timeout=timedelta(minutes=30),
    )
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
        mixed_precision="fp16" if args.amp and torch.cuda.is_available() else "no",
        log_with="wandb" if args.use_wandb and wandb is not None else None,
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    local_rank = accelerator.local_process_index
    device = accelerator.device

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    seed_everything(args.seed + rank)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, "train.log"), accelerator)
    distributed_type = str(accelerator.distributed_type).split(".")[-1].lower()
    if is_main_process(accelerator):
        logger.info(
            f"[Accelerate] world_size={world_size}, rank={rank}, local_rank={local_rank}, "
            f"device={device}, distributed_type={distributed_type}, backend={backend}"
        )
        logger.info(f"[Args] {vars(args)}")
        if backend_note:
            logger.warning(backend_note)
        logger.info(f"[PreHead] type={args.pre_head_type} transformer_nhead={args.pre_head_transformer_nhead}")
        logger.info(
            f"[FeatureSource] source={args.feature_source} hidden_layer={args.hidden_layer} "
            f"learnable_fuse_last_n_layers={args.learnable_fuse_last_n_layers}"
        )
        if args.quick:
            logger.info("[Quick] enabled: freeze_backbone=True, ctc_crf_state_len=5, ctc_crf_blank_score=0, head_output_scale=5, head_output_activation=tanh, head_type=ctc_crf, pre_ctc_module=none")

    # ---- model (先建模型拿 tokenizer，保持原数据逻辑) ----
    import os as _os
    _os.environ["CTC_CRF_STATE_LEN"] = str(args.ctc_crf_state_len)
    n_base = len(ID2BASE) - 1
    if n_base <= 0:
        raise ValueError("CTC-CRF alphabet must include at least one non-blank base.")
    # CTC-CRF head emits full (blank+base) scores; blank score is overwritten during forward.
    if args.head_type == "ctc_crf":
        num_classes = (n_base ** args.ctc_crf_state_len) * (n_base + 1)
    else:
        num_classes = len(ID2BASE)

    base_model = BasecallModel(
        model_path=args.model_name_or_path,
        num_classes=num_classes if num_classes is not None else None,
        hidden_layer=args.hidden_layer,
        learnable_fuse_last_n_layers=args.learnable_fuse_last_n_layers,
        feature_source=args.feature_source,
        vq_device=args.vq_device,
        vq_token_batch_size=args.vq_token_batch_size,
        freeze_backbone=bool(args.freeze_backbone),
        reset_backbone_weights=bool(args.reset_backbone_weights),
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        unfreeze_layer_start=args.unfreeze_layer_start,
        unfreeze_layer_end=args.unfreeze_layer_end,
        head_output_activation=args.head_output_activation,
        head_output_scale=args.head_output_scale,
        pre_head_type=args.pre_head_type,
        pre_head_transformer_nhead=args.pre_head_transformer_nhead,
        head_type=args.head_type,
        head_crf_blank_score=float(args.ctc_crf_blank_score),
        head_crf_n_base=n_base,
        head_crf_state_len=int(args.ctc_crf_state_len),
    )

    model = base_model

    if is_main_process(accelerator):
        raw_model = accelerator.unwrap_model(model)
        total_params, trainable_params = count_parameters(raw_model)
        logger.info(f"[Model] total_params={total_params:,} trainable_params={trainable_params:,}")
        logger.info(f"[Model] architecture:\n{raw_model}")

    tokenizer = model.tokenizer

    # ---- scan + split ----
    train_jsonl_paths = _parse_folder_list(args.train_jsonl_paths)
    val_jsonl_paths = _parse_folder_list(args.val_jsonl_paths)
    test_jsonl_paths = _parse_folder_list(args.test_jsonl_paths)
    train_npy_paths = _parse_folder_list(args.train_npy_paths)
    val_npy_paths = _parse_folder_list(args.val_npy_paths)
    test_npy_paths = _parse_folder_list(args.test_npy_paths)

    using_jsonl = bool(args.jsonl_paths or train_jsonl_paths or val_jsonl_paths or test_jsonl_paths)
    using_npy = bool(args.npy_paths or train_npy_paths or val_npy_paths or test_npy_paths)
    if using_jsonl and using_npy:
        raise ValueError("Do not mix jsonl inputs with tokens/reference npy inputs in the same run.")

    if using_npy:
        if train_npy_paths or val_npy_paths or test_npy_paths:
            train_pairs = scan_npy_pairs(train_npy_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive)
            val_pairs = scan_npy_pairs(val_npy_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive) if val_npy_paths else []
            test_pairs = scan_npy_pairs(test_npy_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive) if test_npy_paths else []
            train_dataset = MultiNpySignalRefDataset(train_pairs, token_offset=args.token_offset)
            val_dataset = MultiNpySignalRefDataset(val_pairs, token_offset=args.token_offset) if len(val_pairs) else None
            test_dataset = MultiNpySignalRefDataset(test_pairs, token_offset=args.token_offset) if len(test_pairs) else None
        else:
            if not args.npy_paths:
                raise ValueError("Provide --npy_paths or explicit --train_npy_paths/--val_npy_paths/--test_npy_paths.")
            npy_paths = [x.strip() for x in args.npy_paths.split(",") if x.strip()]
            npy_pairs = scan_npy_pairs(npy_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive)
            if args.group_by == "record":
                all_dataset = MultiNpySignalRefDataset(npy_pairs, token_offset=args.token_offset)
                train_idx, val_idx, test_idx = split_indices(
                    len(all_dataset),
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
                train_dataset = Subset(all_dataset, train_idx)
                val_dataset = Subset(all_dataset, val_idx) if len(val_idx) else None
                test_dataset = Subset(all_dataset, test_idx) if len(test_idx) else None
            else:
                train_pairs, val_pairs, test_pairs = split_npy_pairs_by_group(
                    npy_pairs,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
                train_dataset = MultiNpySignalRefDataset(train_pairs, token_offset=args.token_offset)
                val_dataset = MultiNpySignalRefDataset(val_pairs, token_offset=args.token_offset) if len(val_pairs) else None
                test_dataset = MultiNpySignalRefDataset(test_pairs, token_offset=args.token_offset) if len(test_pairs) else None

        if is_main_process(accelerator):
            if args.group_by == "record":
                logger.info(f"[Data] split by record: train={len(train_dataset)} val={len(val_dataset) if val_dataset is not None else 0} test={len(test_dataset) if test_dataset is not None else 0}")
            else:
                logger.info(f"[Data] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} test_pairs={len(test_pairs)}")
    else:
        if train_jsonl_paths or val_jsonl_paths or test_jsonl_paths:
            train_files = scan_jsonl_files(train_jsonl_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive)
            val_files = scan_jsonl_files(val_jsonl_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive) if val_jsonl_paths else []
            test_files = scan_jsonl_files(test_jsonl_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive) if test_jsonl_paths else []
            train_dataset = MultiJsonlSignalRefDataset(train_files, token_offset=args.token_offset)
            val_dataset = MultiJsonlSignalRefDataset(val_files, token_offset=args.token_offset) if len(val_files) else None
            test_dataset = MultiJsonlSignalRefDataset(test_files, token_offset=args.token_offset) if len(test_files) else None
        else:
            if not args.jsonl_paths:
                raise ValueError("Provide --jsonl_paths or explicit --train_jsonl_paths/--val_jsonl_paths/--test_jsonl_paths.")
            jsonl_paths = [x.strip() for x in args.jsonl_paths.split(",") if x.strip()]
            jsonl_files = scan_jsonl_files(jsonl_paths, group_by=args.group_by if args.group_by != "record" else "file", recursive=args.recursive)
            if args.group_by == "record":
                all_dataset = MultiJsonlSignalRefDataset(jsonl_files, token_offset=args.token_offset)
                train_idx, val_idx, test_idx = split_indices(
                    len(all_dataset),
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
                train_dataset = Subset(all_dataset, train_idx)
                val_dataset = Subset(all_dataset, val_idx) if len(val_idx) else None
                test_dataset = Subset(all_dataset, test_idx) if len(test_idx) else None
            else:
                train_files, val_files, test_files = split_jsonl_files_by_group(
                    jsonl_files,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.split_seed,
                )
                train_dataset = MultiJsonlSignalRefDataset(train_files, token_offset=args.token_offset)
                val_dataset = MultiJsonlSignalRefDataset(val_files, token_offset=args.token_offset) if len(val_files) else None
                test_dataset = MultiJsonlSignalRefDataset(test_files, token_offset=args.token_offset) if len(test_files) else None

        if is_main_process(accelerator):
            if args.group_by == "record":
                logger.info(f"[Data] split by record: train={len(train_dataset)} val={len(val_dataset) if val_dataset is not None else 0} test={len(test_dataset) if test_dataset is not None else 0}")
            else:
                logger.info(f"[Data] train_files={len(train_files)} val_files={len(val_files)} test_files={len(test_files)}")

    if args.feature_source == "vq_embedding":
        collate_fn = create_vq_collate_fn()
    else:
        collate_fn = create_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_fn,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_fn,
        )

    # ---- optimizer/scheduler/loss ----
    optimizer = build_adamw_with_no_decay(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler, sched_name = build_scheduler(optimizer, total_steps, warmup_steps, args.min_lr, logger, accelerator)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if test_loader is not None:
        test_loader = accelerator.prepare(test_loader)

    if is_main_process(accelerator):
        logger.info(f"[Scheduler] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
                    f"warmup_steps={warmup_steps} scheduler={sched_name}")

    # ---- optional pretrained weights (only if NOT resuming) ----
    if args.pretrained_ckpt and not args.resume_ckpt:
        load_pretrained_weights(
            accelerator,
            model,
            args.pretrained_ckpt,
            strict=bool(args.pretrained_strict),
            key=args.pretrained_key,
            logger=logger,
        )

    # ---- wandb ----
    use_wandb = bool(args.use_wandb and wandb is not None and is_main_process(accelerator))
    if args.use_wandb and wandb is None and is_main_process(accelerator):
        logger.warning("[wandb] wandb not installed; pip install wandb")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type=args.wandb_job_type,
            config=vars(args),
        )

    # ---- resume (after model+optim+sched created) ----
    start_epoch = 1
    best_pbma = -1.0
    if args.resume_ckpt:
        se, bp = load_checkpoint(
            args.resume_ckpt,
            accelerator,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
            logger=logger if is_main_process(accelerator) else None,
        )
        start_epoch = se
        if bp is not None:
            try:
                best_pbma = float(bp)
            except Exception:
                pass
        if is_main_process(accelerator):
            logger.info(f"[Resume] start_epoch={start_epoch}, best_acc={best_pbma}")

    # ---- loop ----
    train_losses, val_losses, val_accs = [], [], []

    # if resuming mid-run, you may want to pad arrays; we keep it simple:
    # curves will reflect only epochs run in this session.
    decoder_mode = args.train_decoder
    if args.head_type == "ctc":
        if args.train_decoder != "ctc_viterbi" and is_main_process(accelerator):
            logger.warning("[Decoder] CTC head supports only ctc_viterbi, overriding train_decoder.")
        decoder_mode = "ctc_viterbi"

    if decoder_mode == "ctc_crf":
        if args.head_type != "ctc_crf":
            raise ValueError("--train_decoder ctc_crf requires --head_type ctc_crf.")
        use_amp = False
        if args.amp and is_main_process(accelerator):
            logger.info("[AMP] Disabled for ctc_crf decoder (requires fp32).")
    else:
        use_amp = device.type == "cuda"
        if device.type != "cuda" and is_main_process(accelerator):
            logger.warning("[AMP] non-ctc_crf decoder selected but CUDA not available; running in fp32.")
    if is_main_process(accelerator):
        logger.info(f"[Decoder] mode={decoder_mode} use_amp={use_amp}")

    for epoch in range(start_epoch, args.num_epochs + 1):
        tr_loss = train_one_epoch(
            accelerator,
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            args.log_interval,
            use_wandb,
            args.ctc_crf_blank_score,
            use_amp,
            args.clip_grad_norm,
            args.head_type,
        )
        train_losses.append(tr_loss)

        if is_main_process(accelerator):
            logger.info(f"[Train] epoch={epoch} avg_loss={tr_loss:.4f}")

        val_loss, val_acc = None, None
        if val_loader is not None:
            val_loss, val_acc, val_crf_acc, val_cov, val_blank, val_nonzero_len = eval_one_epoch(
                accelerator,
                model,
                val_loader,
                device,
                "val",
                args.ctc_crf_blank_score,
                args.koi_blank_score,
                args.acc_balanced,
                args.acc_min_coverage,
                use_amp,
                decoder_mode,
                args.head_type,
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if is_main_process(accelerator):
                if decoder_mode == "ctc_crf":
                    logger.info(
                        f"[Val] epoch={epoch} loss={val_loss:.4f} acc={val_acc:.4f} "
                        f"coverage={val_cov:.4f} blank={val_blank:.4f} nonzero_len={val_nonzero_len:.2f}"
                    )
                else:
                    logger.info(
                        f"[Val] epoch={epoch} loss={val_loss:.4f} acc={val_acc:.4f} "
                        f"coverage={val_cov:.4f} blank={val_blank:.4f} nonzero_len={val_nonzero_len:.2f}"
                    )
                if use_wandb and wandb is not None:
                    payload = {
                        "val/loss": float(val_loss),
                        "val/acc": float(val_acc),
                        "val/coverage": float(val_cov),
                        "val/blank": float(val_blank),
                        "val/nonzero_len": float(val_nonzero_len),
                        "epoch": epoch,
                    }
                    if decoder_mode == "ctc_crf":
                        payload["val/crf_acc"] = float(val_crf_acc)
                    wandb.log(payload)

        # ---- checkpoint save ----
        accelerator.wait_for_everyone()
        if is_main_process(accelerator) and (epoch % max(args.save_every, 1) == 0):
            last_path = os.path.join(args.output_dir, "ckpt_last.pt")
            save_checkpoint(
                last_path,
                accelerator,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_pbma=best_pbma,
                extra={"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc},
            )
            logger.info(f"[CKPT] saved {last_path}")

        if is_main_process(accelerator) and args.save_best and (val_acc is not None):
            if float(val_acc) > float(best_pbma):
                best_pbma = float(val_acc)
                best_path = os.path.join(args.output_dir, "ckpt_best.pt")
                save_checkpoint(
                    best_path,
                    accelerator,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_pbma=best_pbma,
                    extra={"train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc},
                )
                logger.info(f"[CKPT] new best acc={best_pbma:.4f} @ epoch={epoch}, saved {best_path}")

        if use_wandb and wandb is not None and is_main_process(accelerator):
            wandb.log({"epoch": epoch, "train/epoch_loss": float(tr_loss)})

    # ---- test ----
    if test_loader is not None:
        test_loss, test_acc, test_crf_acc, test_cov, test_blank, test_nonzero_len = eval_one_epoch(
            accelerator,
            model,
            test_loader,
            device,
            "test",
            args.ctc_crf_blank_score,
            args.koi_blank_score,
            args.acc_balanced,
            args.acc_min_coverage,
            use_amp,
            decoder_mode,
            args.head_type,
        )
        if is_main_process(accelerator):
            if decoder_mode == "ctc_crf":
                logger.info(
                    f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} "
                    f"coverage={test_cov:.4f} blank={test_blank:.4f} nonzero_len={test_nonzero_len:.2f}"
                )
            else:
                logger.info(
                    f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} "
                    f"coverage={test_cov:.4f} blank={test_blank:.4f} nonzero_len={test_nonzero_len:.2f}"
                )
            if use_wandb and wandb is not None:
                payload = {
                    "test/loss": float(test_loss),
                    "test/acc": float(test_acc),
                    "test/coverage": float(test_cov),
                    "test/blank": float(test_blank),
                    "test/nonzero_len": float(test_nonzero_len),
                }
                if decoder_mode == "ctc_crf":
                    payload["test/crf_acc"] = float(test_crf_acc)
                wandb.log(payload)

    # ---- final save curves/csv ----
    if is_main_process(accelerator):
        try:
            if val_losses and val_accs:
                plot_curves(train_losses, val_losses, val_accs, save_path=os.path.join(args.output_dir, "curves.png"))
                save_metrics_csv(train_losses, val_losses, val_accs, os.path.join(args.output_dir, "metrics.csv"))
        except Exception as e:
            logger.warning(f"[Final Plot/CSV] failed: {e}")

    # ---- log a final alignment heatmap to wandb ----
    if use_wandb and wandb is not None and is_main_process(accelerator):
        loader = test_loader if test_loader is not None else val_loader
        if loader is not None:
            batch = next(iter(loader))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            with torch.no_grad():
                with accelerator.autocast() if use_amp else nullcontext():
                    logits_btc = model(input_ids, attention_mask=attention_mask)
                logits_tbc = logits_btc.transpose(0, 1)
            input_lengths = resolve_input_lengths(
                input_ids,
                attention_mask=attention_mask,
                input_lengths=batch.get("input_lengths"),
            )
            if decoder_mode == "koi":
                pred_seqs = koi_beam_search_decode(
                    logits_tbc,
                    blank_score=float(args.koi_blank_score),
                    input_lengths=input_lengths,
                )
            elif decoder_mode == "ctc_viterbi":
                pred_seqs = ctc_viterbi_decode(
                    logits_tbc,
                    input_lengths=input_lengths,
                    blank_idx=BLANK_IDX,
                )
            else:
                pred_seqs = []
                input_len_list = input_lengths.detach().cpu().tolist()
                for idx, input_len in enumerate(input_len_list):
                    step_len = int(input_len)
                    if step_len <= 0:
                        pred_seqs.append([])
                        continue
                    decoded_ids = ctc_crf_decode(
                        logits_tbc[:step_len, idx : idx + 1, :].float(),
                        blank_idx=BLANK_IDX,
                    )[0]
                    pred_seqs.append(decoded_ids[:step_len])
            ref_seqs = batch["target_seqs"]
            fig = plot_alignment_heatmap(pred_seqs, ref_seqs, max_reads=32, max_len=80)
            wandb.log({"final/base_alignment": wandb.Image(fig)})
            plt.close(fig)

    if use_wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
