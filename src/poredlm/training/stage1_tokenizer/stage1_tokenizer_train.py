"""Stage 1 training loop for tokenizer and encoder pretraining."""

# vqe_train_accelerate.py
# Nanopore Signal Tokenizer Training Script with VQ-VAE
# Industrial-grade training pipeline for nanopore raw signal tokenization using Vector Quantization.
# Supports distributed training (DDP), dynamic logging, checkpointing, and independent evaluation dataset.
# This version uses Hugging Face Accelerate for simplified multi-GPU/mixed precision training.
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import time
import json
from pprint import pformat
from scipy.stats import entropy
import argparse
from typing import Dict, List, Optional
import yaml
# Import Accelerate components
from accelerate import Accelerator, DistributedType
# Relative imports from the same package
# Note: The relative imports might need adjustment depending on your package structure.
# If this script is run directly, you might need to add the parent directory to sys.path.
from dataset import NanoporeSignalDataset
from tokenizer_model_v0 import Nanopore_Tokenizer_Model_V0
from tokenizer_model_v1 import Nanopore_Tokenizer_Model_V1
from tokenizer_model_v2 import Nanopore_Tokenizer_Model_V2

from accelerate import InitProcessGroupKwargs
from datetime import timedelta

def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def write_runtime_json(runtime_dict, filename="runtime.json"):
    """
    将字典写入指定的 json 文件。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=4 让生成的 JSON 文件易于阅读
            # ensure_ascii=False 确保中文字符能正常显示
            json.dump(runtime_dict, f, indent=4, ensure_ascii=False)
        print(f"成功更新 {filename}")
    except Exception as e:
        print(f"写入文件时出错: {e}")

def read_runtime_json(filename="runtime.json"):
    """
    从 json 文件读取内容并返回字典。
    如果文件不存在或损坏，返回一个空字典。
    """
    if not os.path.exists(filename):
        print(f"提示: {filename} 不存在，返回空配置。")
        return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"错误: {filename} 格式损坏。")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# =============================================================================
# Utility Functions
# =============================================================================
def log_and_save(
    epoch: int,
    global_step: int,
    total_epochs: int,
    total_global_steps: int,
    train_start_time: float,
    epoch_total_steps: int,
    avg_recon_loss: float,
    avg_total_loss: float,
    avg_comit_loss: float,
    avg_diver_loss: float,
    avg_ortho_loss: float,
    avg_rperc_ratio: float,
    codebook_usage: float,
    loss_csv_path: str,
    lr: float,
    accelerator: Accelerator
):
    """
    Log training metrics to console and append to CSV for offline analysis.
    Time estimation is based on current epoch progress.
    Only the main process writes to console and CSV.
    """
    import time

    current_time = time.time()
    elapsed_seconds = current_time - train_start_time
    steps_done = global_step % total_global_steps or 1
    avg_time_per_step = elapsed_seconds / steps_done
    remaining_seconds = avg_time_per_step * max(0, total_global_steps - steps_done)

    def format_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    elapsed_str = format_hms(elapsed_seconds)
    remaining_str = format_hms(remaining_seconds)

    epoch_width = len(str(total_epochs))
    step_width = len(str(total_global_steps))

    # Only main process prints and writes to CSV
    if accelerator.is_main_process:
        print(
            f"[Epoch {epoch+1:>{epoch_width}}/{total_epochs} | "
            f"Step {global_step:>{step_width}}/{total_global_steps} | "
            f"{elapsed_str}<{remaining_str}] "
            f"Total: {avg_total_loss:>8.6f} | "
            f"Recon: {avg_recon_loss:>8.6f} | "
            f"Comit: {avg_comit_loss:>8.6f} | "
            f"Rperc: {avg_rperc_ratio:>8.6f} | "
            #f"Diver: {avg_diver_loss:>3.2f} | "
            f"Usage: {codebook_usage*100:>3.1f}% | "
            f"LR: {lr:>7.2e} |"
        )


def print_training_args(**kwargs):
    """
    Pretty-print all training hyperparameters at startup for reproducibility and debugging.
    """
    print("\n" + "="*60)
    print(" 🚀 Starting VQE Training with Accelerate. Configuration:")
    print("="*60)
    print(pformat(kwargs, width=100, sort_dicts=False))
    print("="*60 + "\n")


def save_full_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    spoch: int,
    global_step: int,
    cnn_type: int,
    model_type: int,
    dynamic_commitment_weight: float,
    accelerator: Accelerator
):
    """
    Save a full training checkpoint (model, optimizer, RNG states) for resuming.
    Uses Accelerate's save_state method which handles DDP/DDP sharding automatically.
    Only the main process saves the checkpoint.
    """
    # Accelerate's save_state handles distributed saving internally
    accelerator.save_state(path)
    if accelerator.is_main_process:
        print(f"✅ Full checkpoint saved to {path}")
        # Optionally save additional metadata separately
        metadata = {
            'epoch': epoch,
            'spoch': spoch,
            'global_step': global_step,
            'cnn_type': cnn_type,
            "model_type": model_type,
            'dynamic_commitment_weight': dynamic_commitment_weight,
        }
        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

def load_checkpoint_metadata(path: str):
    """
    Load metadata associated with a checkpoint.
    """
    # 使用 os.path.join 来安全地拼接路径
    meta_path = os.path.join(path, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return None


# -*- coding: utf-8 -*-
"""
评估模块: 计算向量量化(VQ)模型的码本使用率、Top-K集中度、熵等指标。

该模块提供了一个核心函数，用于在验证集上评估不同类型的VQ模型（VQ-VAE, EMA-VQ, Residual VQ）。
它通过分布式训练环境（如Accelerate）运行，并计算一系列关键性能指标。
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --- 1. 通用的单层码本指标计算函数 ---
def calculate_single_codebook_metrics(
    token_counts_np: np.ndarray,
    total_tokens_in_layer: int,
    codebook_size: int,
    last_token_counts_np_global: np.ndarray # 传入对应层的 last_token_counts_np
):
    """
    计算单个码本层（VQ1/2/3的整个码本，或RVQ的一个quantizer层）的指标。

    此函数是评估流程的核心，封装了所有针对单个离散token分布的统计计算逻辑，
    如使用率、集中度、熵和KL散度，实现了代码复用。

    Args:
        token_counts_np (np.ndarray): 该层码本的使用计数数组，形状为 (codebook_size,)。
                                      每个索引i的值代表编码器输出的token i出现的次数。
        total_tokens_in_layer (int): 该层产生的总token数量。
                                     用于将计数转换为概率，计算归一化指标。
        codebook_size (int): 码本大小，即可能的token总数。
                             用于计算使用率和基础概率。
        last_token_counts_np_global (np.ndarray): 该层上次的计数，用于计算KL散度。
                                                  形状与token_counts_np相同。
                                                  如果为None，则跳过KL散度计算。

    Returns:
        tuple: 包含以下元素的元组:
               - kl_div (float): 当前分布相对于上次分布的KL散度。
               - used_code_n (int): 当前被使用的码本条目数量（计数大于0的条目数）。
               - usage_ratio (float): 码本使用率 (used_code_n / codebook_size)。
               - top1_ratio ... top10_ratio (float): Top-1 到 Top-10 集中度比率。
                                                     表示前N个最频繁token的出现频率相对于均匀分布的倍数。
               - entropy_val (float): 当前token分布的香农熵（以2为底，单位bit）。
               - last_token_counts_np_updated (np.ndarray): 本次的计数，用于更新该层的全局历史记录。
    """
    # 1. 计算基本指标
    # 计算有多少个码本条目被实际使用过
    used_code_n = np.count_nonzero(token_counts_np)
    # 计算被使用码本的比例
    usage_ratio = used_code_n / codebook_size

    # 初始化返回指标
    entropy_val = 0.0
    # 初始化Top-N集中度指标
    top1_ratio = top3_ratio = top5_ratio = top7_ratio = top9_ratio = top10_ratio = 0.0

    # 如果该层没有任何token，则所有基于频率的指标都为0
    if total_tokens_in_layer > 0:
        # 对计数进行降序排序，以便快速获取最常用的token
        sorted_counts = np.sort(token_counts_np)[::-1]
        # 计算均匀分布下的基础概率
        base_ratio = 1.0 / codebook_size

        # 内部辅助函数，用于计算Top-N集中度
        # 它衡量第N个最常见token的频率是均匀分布期望频率的多少倍
        def get_ratio(rank):
            return (sorted_counts[rank-1] / total_tokens_in_layer) / base_ratio if len(sorted_counts) >= rank else 0.0

        top1_ratio = get_ratio(1)
        top3_ratio = get_ratio(3)
        top5_ratio = get_ratio(5)
        top7_ratio = get_ratio(7)
        top9_ratio = get_ratio(9)
        # Top-10比率略有不同，计算的是前10个token的总频率占比
        top10_ratio = (np.sum(sorted_counts[:10]) / total_tokens_in_layer) / base_ratio if len(sorted_counts) >= 10 else 0.0

        # 计算香农熵 H(p) = -sum(p_i * log2(p_i))
        prob = token_counts_np / total_tokens_in_layer
        # 只对非零概率项求和，避免log(0)的问题
        nz_prob = prob[prob > 0]
        entropy_val = -np.sum(nz_prob * np.log2(nz_prob))

    # 2. 计算 Kullback-Leibler (KL) 散度
    # KL散度 D_KL(P_current || P_last) 衡量当前token分布与上次分布的差异
    kl_div = 0.0
    # 添加小的epsilon值以防止除以零和log(0)的情况
    eps = 1e-10
    current_prob = (token_counts_np + eps) / (total_tokens_in_layer + eps * codebook_size)

    if last_token_counts_np_global is not None:
        last_total = np.sum(last_token_counts_np_global)
        last_prob = (last_token_counts_np_global + eps) / (last_total + eps * codebook_size)
        # 计算 KL散度 D(P_current || P_Last)
        kl_div = np.sum(current_prob * np.log(current_prob / last_prob))
    # 如果 last_token_counts_np_global 为 None，则 kl_div 保持初始值 0.0

    # 3. 返回更新后的 last_token_counts_np，供下次调用
    # 返回当前层的计数，作为新的历史记录
    updated_last_counts = token_counts_np.copy()

    return (
        kl_div, used_code_n, usage_ratio,
        top1_ratio, top3_ratio, top5_ratio, top7_ratio, top9_ratio, top10_ratio,
        entropy_val, updated_last_counts
    )

# =============================================================================
# Main Training Function
# =============================================================================

def vqe_train(
    train_npy_dir: str,
    model_type: int = 1,
    codebook_type: str = "VQ",
    evaluation_npy_dir: Optional[str] = None,
    output_model_path: str = "nanopore_vq_tokenizer.pth",
    lr: float = 1e-4,
    num_epochs: int = 10,
    codebook_size: int = 8192,
    codebook_decay: float = 0.99,
    codebook_emadc: int = 2,
    chunk_size: int = 12000,
    num_workers: int = 8,
    update_loss_weight_every: int = 10,
    prefetch_factor: int = 128,
    val_ratio: float = 0.001,
    do_evaluate: bool = True,
    use_dynamic_commitment_weight: bool = True,
    commitment_weight: float = 1.0,
    commitment_weight_lr: float =0.01,
    commitment_weight_freeze_steps: int = 20000,
    commitment_weight_rpc: float = 1.0,
    codebook_diversity_loss_weight: float = 1.0,
    orthogonal_reg_weight: float = 1.0,
    loss_csv_path: str = "train_loss.csv",
    use_wandb: bool = True,
    wandb_project: str = "nanopore_vq",
    wandb_name: str = "default_run",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 500,
    warmup_start_factor: float = 1e-6,
    warmup_end_factor: float = 1.0,
    main_scheduler_end_factor: float = 1e-6,
    save_checkpoint_every_spoch: int = 500,
    evaluate_every_spoch: int = 10,
    checkpoint_path: Optional[str] = None,
    cnn_type: int = 0,
    init_codebook_path: Optional[str] = None,
    cnn_checkpoint_path: Optional[str] = None,
    freeze_cnn: int = 0,
    learnable_codebook: bool = False,
    global_batch_size: int = 256,
    device_micro_batch_size: int = 16, # Renamed for clarity, was 'batch_size' before
    mixed_precision: str = "bf16", # Options: "no", "fp16", "bf16", "fp8"
    gradient_clipping: float = 1.0, # Set to None to disable
    cpu: bool = False, # Set to True to force CPU training
    dataset_logic_chunk_size: int = 6000,
    teacher_model_path: str = None,
    seed: int = 42,
):
    """
    Training of Nanopore VQ tokenizer using Hugging Face Accelerate.

    Key features:
      - Distributed training handled by Accelerate (multi-GPU, multi-node).
      - Mixed precision training support (FP16, BF16).
      - Automatic gradient scaling for FP16.
      - Independent evaluation dataset via `evaluation_npy_dir`.
      - Checkpoint resume support (using Accelerate's state management).
      - Pre-trained CNN weight loading & freezing.
      - Initial codebook initialization (e.g., from Faiss).
      - WandB & CSV logging (main process only).
      - Learning rate scheduling with warmup.
      - Gradient accumulation integrated with Accelerate.

    ⚠️ NOTE ON DWA (Dynamic Weight Averager):
        The DWA module is used SOLELY for monitoring and logging purposes.
        It does NOT influence the actual loss computation or gradient updates.
        The training loss remains:
            total_loss = recon_loss + comit_loss * commitment_weight
        DWA weights are only recorded in logs/CSV/W&B for analysis.
    """
    if commitment_weight_rpc < 0.0001:
        commitment_weight_rpc = 0.0001

    print_training_args(
        train_npy_dir=train_npy_dir,
        model_type=model_type,
        evaluation_npy_dir=evaluation_npy_dir,
        output_model_path=output_model_path,
        lr=lr,
        num_epochs=num_epochs,
        codebook_size=codebook_size,
        codebook_decay=codebook_decay,
        codebook_emadc=codebook_emadc,
        chunk_size=chunk_size,
        num_workers=num_workers,
        update_loss_weight_every=update_loss_weight_every,
        prefetch_factor=prefetch_factor,
        val_ratio=val_ratio,
        do_evaluate=do_evaluate,
        commitment_weight=commitment_weight,
        commitment_weight_lr=commitment_weight_lr,
        codebook_diversity_loss_weight=codebook_diversity_loss_weight,
        orthogonal_reg_weight=orthogonal_reg_weight,
        loss_csv_path=loss_csv_path,
        use_wandb=use_wandb,
        use_dynamic_commitment_weight=use_dynamic_commitment_weight,
        commitment_weight_freeze_steps=commitment_weight_freeze_steps,
        commitment_weight_rpc=commitment_weight_rpc,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
        warmup_end_factor=warmup_end_factor,
        main_scheduler_end_factor=main_scheduler_end_factor,
        save_checkpoint_every_spoch=save_checkpoint_every_spoch,
        evaluate_every_spoch=evaluate_every_spoch,
        checkpoint_path=checkpoint_path,
        init_codebook_path=init_codebook_path,
        cnn_type=cnn_type,
        freeze_cnn= freeze_cnn,
        learnable_codebook = learnable_codebook,
        global_batch_size=global_batch_size,
        device_micro_batch_size=device_micro_batch_size,
        mixed_precision=mixed_precision,
        gradient_clipping=gradient_clipping,
        cpu=cpu,
        dataset_logic_chunk_size=dataset_logic_chunk_size,
        codebook_type=codebook_type,
        teacher_model_path=teacher_model_path,
        seed=seed
    )

    seed_all(seed)


    # Calculate accumulation steps based on global and micro batch sizes
    effective_micro_batch = device_micro_batch_size * (1 if cpu else torch.cuda.device_count() if torch.cuda.is_available() else 1) # Estimate num_processes before accelerator init
    accumulation_steps = global_batch_size // effective_micro_batch

    if accumulation_steps == 0:
        raise ValueError(
            f"global_batch_size ({global_batch_size}) is too small for the current "
            f"device_micro_batch_size ({device_micro_batch_size}). "
            f"Minimum global_batch_size required is {effective_micro_batch}."
        )
    # 设置一个足够长的超时时间，比如 2 小时 (7200秒)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    # Initialize Accelerator
    # This handles device placement, distributed setup, mixed precision, and gradient accumulation automatically.
    accelerator = Accelerator(
        mixed_precision=None,
        cpu=cpu,
        gradient_accumulation_steps=accumulation_steps, # Pass the calculated steps to Acceleratoir
        log_with="wandb", # <-- Enable wandb integration
        project_dir="log", # <--- 添加这一行，提供一个目录给 accelerator 管理日志
    )
    # Initialise your wandb run, passing wandb parameters and any config information
    accelerator.init_trackers(
        project_name=wandb_project, 
        config={
                "device_micro_batch_size": device_micro_batch_size, # Changed from 'batch_size'
                "lr": lr,
                "cnn_type":cnn_type,
                "model_type":model_type,
                "codebook_type": codebook_type,
                "teacher_model_path": teacher_model_path,
                "num_epochs": num_epochs,
                "codebook_size": codebook_size,
                "codebook_decay": codebook_decay,
                "codebook_emadc": codebook_emadc,
                "chunk_size": chunk_size,
                "dataset_logic_chunk_size":dataset_logic_chunk_size,
                "device_micro_batch_size":device_micro_batch_size,
                "global_batch_size":global_batch_size,
                "update_loss_weight_every": update_loss_weight_every,
                "commitment_weight": commitment_weight,
                "commitment_weight_lr": commitment_weight_lr,
                "commitment_weight_freeze_steps":commitment_weight_freeze_steps,
                "commitment_weight_rpc":commitment_weight_rpc,
                "codebook_diversity_loss_weight": codebook_diversity_loss_weight,
                "orthogonal_reg_weight": orthogonal_reg_weight,
                "world_size": accelerator.num_processes, # Changed from 'world_size'
                "mixed_precision": mixed_precision,
                "global_batch_size": global_batch_size,
                "seed": seed
        },
        # init_kwargs={"wandb": {"entity": "jiaoshuaihit-hit","name":wandb_name}}
        init_kwargs={"wandb": {"entity": "zhoukek-zhejiang-university","name":wandb_name}}

    )

    # Log accelerator info
    if accelerator.is_main_process:
        print(f"🚀 Accelerator initialized. Device: {accelerator.device}, Type: {accelerator.distributed_type}")
        print(f"   Number of processes: {accelerator.num_processes}")
        print(f"   Mixed Precision: {accelerator.mixed_precision}")
        print(f"   Global Batch Size: {global_batch_size}, Device Micro-Batch Size: {device_micro_batch_size}")
        print(f"   Gradient Accumulation Steps: {accumulation_steps}")


    # ========================
    # Data Loading
    # ========================
    # DataLoader's batch_size is now the micro-batch size per device/process.
    train_dataset = NanoporeSignalDataset(shards_dir=train_npy_dir,logic_chunk_size=dataset_logic_chunk_size,logic_chunk_overlap_size=100)
    # Accelerate provides a convenient way to create distributed samplers
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=device_micro_batch_size, # Micro-batch size per process
        shuffle=True, # Shuffling is handled by the sampler
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True
    )
    # Prepare dataloader with Accelerate (handles DistributedSampler internally)
    train_dataloader = accelerator.prepare(train_dataloader)

    # ========================
    # Evaluation Setup
    # ========================
    val_loader = None
    # 建议在类初始化或全局位置定义这个变量，用于持久化存储
    # 记录上一次的码表推理命中频次分布

    # --- 2. 主评估函数 ---
    # 全局变量，用于存储上一次评估时各层的token计数，以便计算KL散度
    last_token_counts_np_0 = None
    last_token_counts_np_1 = None
    last_token_counts_np_2 = None # <-- 新增
    last_token_counts_np_3 = None # <-- 新增

    def evaluate_codebook_metrics_v3():
        """
        在验证集上评估码本使用率、Top-K集中度和熵。

        该函数负责模型推理、分布式张量聚合和最终指标计算的全过程。
        它区分了单层模型（VQ1/2/3）和多层模型（RVQ），并统一调用上述通用函数。
        现在返回每层的独立指标。
        """
        # 声明使用外部的非局部变量
        nonlocal last_token_counts_np_0, last_token_counts_np_1,last_token_counts_np_2,last_token_counts_np_3

        # 如果没有加载验证集，则返回默认值
        if val_loader is None:
            # 返回值的顺序: kl_div_0, used_codes_0, usage_ratio_0, total_tokens_0,
            #              top1_ratio_0, top3_ratio_0, top5_ratio_0, top7_ratio_0, top9_ratio_0, top10_ratio_0,
            #              entropy_val_0, 
            #               kl_div_1, used_codes_1, usage_ratio_1, total_tokens_1,
            #              top1_ratio_1, top3_ratio_1, top5_ratio_1, top7_ratio_1, top9_ratio_1, top10_ratio_1,
            #              entropy_val_1, 
            #              max_entropy, recon_loss, comit_loss
            # 对于VQ1/2/3, 第二套指标 (_1) 将全部为0或默认值
            default_metrics = [0.0] * 47
            default_metrics[44] = np.log2(codebook_size) # max_entropy
            default_metrics[45] = 0.0 # recon_loss placeholder
            default_metrics[46] = 0.0 # comit_loss placeholder
            return tuple(default_metrics)

        # --- 初始化累积变量 ---
        # 用于累加验证批次上的损失
        local_recon_loss = 0.0
        local_comit_loss = 0.0
        num_batches = 0

        # --- 根据模型类型初始化分布式计数器 ---
        # 将token计数张量分配到加速器设备（GPU）上，以便在推理时直接操作
        if codebook_type == "VQ":
            token_counts_gpu_0 = torch.zeros(codebook_size, device=accelerator.device)
            token_counts_gpu_1 = None # 该模型类型无第二层
            token_counts_gpu_2 = None # <-- 新增
            token_counts_gpu_3 = None # <-- 新增
        elif codebook_type == "RVQ":
            #assert n_q == 2, f"For model_type 4, n_q must be 2, got {n_q}" # 确保模型配置正确
            token_counts_gpu_0 = torch.zeros(codebook_size, device=accelerator.device)
            token_counts_gpu_1 = torch.zeros(codebook_size, device=accelerator.device)
            token_counts_gpu_2 = None # <-- 新增
            token_counts_gpu_3 = None # <-- 新增
        else:
            raise ValueError(f"Unknown codebook_type: {codebook_type}")

        # --- 分布式调试信息 ---
        # 收集所有进程处理的batch数量，用于调试分布式训练是否均衡
        local_len = torch.tensor([len(val_loader)], device=accelerator.device)
        all_lens = accelerator.gather(local_len)
        if accelerator.is_main_process:
            print(f"\n[Eval Step {global_step}] Batch distribution across processes: {all_lens.tolist()}")

        # --- 模型推理循环 ---
        model.eval() # 设置模型为评估模式
        with torch.no_grad(): # 禁用梯度计算以节省内存和提高速度
            # 创建一个进度条，仅在主进程显示
            pbar = tqdm(val_loader, desc=f"Eval Step {global_step}",disable=not accelerator.is_main_process, leave=False)
            for i, batch in enumerate(pbar):
                x = batch # 获取输入数据
                # --- 模型前向传播与Token计数统计 ---
                if codebook_type == "VQ" and model_type == 0: # 处理 VQ-VAE, EMA-VQ 模型
                    recon, indices, _, loss_breakdown = model(x)
                    recon_loss = F.mse_loss(recon, x) # 计算重建损失
                    local_recon_loss += recon_loss.item()
                    local_comit_loss += loss_breakdown.commitment.item() # 累加承诺损失

                    # 将多维indices展平为一维，统计所有token
                    flat_indices = indices.flatten()
                    # 使用scatter_add_原地更新计数张量，高效且内存友好
                    token_counts_gpu_0.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
                elif codebook_type == "VQ_distill" and model_type == 1:
                    recon, indices, _, loss_breakdown, distill_loss = model(x)
                    recon_loss = F.mse_loss(recon, x) # 计算重建损失
                    local_recon_loss += recon_loss.item()
                    local_comit_loss += loss_breakdown.commitment.item() # 累加承诺损失

                    # 将多维indices展平为一维，统计所有token
                    flat_indices = indices.flatten()
                    # 使用scatter_add_原地更新计数张量，高效且内存友好
                    token_counts_gpu_0.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
                elif codebook_type == "VQ_distill_BERT" and model_type == 2:
                    pass
                elif codebook_type == "RVQ": # 处理 Residual Vector Quantization (RVQ) 模型
                    pass

                num_batches += 1

        # --- 分布式聚合 (All-Reduce) ---
        # 将所有GPU上计算的计数结果汇总到一起
        if codebook_type == "VQ" and model_type == 0:
            global_counts_tensor_0 = accelerator.reduce(token_counts_gpu_0, reduction="sum")
            global_counts_0 = global_counts_tensor_0.cpu().numpy() # 转换回CPU numpy数组以便计算
            global_counts_1 = None
            global_counts_2 = None
            global_counts_3 = None
        elif codebook_type == "RVQ":
            global_counts_tensor_0 = accelerator.reduce(token_counts_gpu_0, reduction="sum")
            global_counts_tensor_1 = accelerator.reduce(token_counts_gpu_1, reduction="sum")
            global_counts_0 = global_counts_tensor_0.cpu().numpy()
            global_counts_1 = global_counts_tensor_1.cpu().numpy()
            global_counts_2 = None
            global_counts_3 = None
        else:
            # Should not reach here due to earlier check
            global_counts_0 = global_counts_1 = global_counts_2 = global_counts_3 = None
        # 聚合验证损失
        metrics = torch.tensor([local_recon_loss, local_comit_loss, float(num_batches)], device=accelerator.device)
        gathered_metrics = accelerator.gather(metrics).view(-1, 3)

        # --- 后期处理（仅在主进程执行） ---
        if accelerator.is_main_process:
            # 计算全局平均 Loss
            total_batches_all = gathered_metrics[:, 2].sum()
            global_recon_loss = (gathered_metrics[:, 0].sum() / total_batches_all).item()
            global_comit_loss = (gathered_metrics[:, 1].sum() / total_batches_all).item()

            # --- 计算每一层的指标 ---
            # Layer 0 (对于 VQ1/2/3，这是唯一的层；对于 RVQ，这是第一层)
            total_tokens_0 = int(np.sum(global_counts_0))
            
            # 计算Layer 0指标，并用其更新 last_token_counts_np_0
            layer_0_metrics = calculate_single_codebook_metrics(
                token_counts_np=global_counts_0,
                total_tokens_in_layer=total_tokens_0,
                codebook_size=codebook_size,
                last_token_counts_np_global=last_token_counts_np_0 # 传入该层的全局变量
            )
            # 解包Layer 0的指标
            kl_div_0, used_code_n_0, usage_ratio_0, top1_ratio_0, top3_ratio_0, top5_ratio_0, \
                top7_ratio_0, top9_ratio_0, top10_ratio_0, entropy_val_0, last_token_counts_np_0 = layer_0_metrics # 更新该层的全局变量

            # Layer 1 (仅 RVQ)
            # 初始化Layer 1的指标，如果模型不是RVQ，这些将保持默认值
            kl_div_1, used_code_n_1, usage_ratio_1, total_tokens_1, top1_ratio_1, top3_ratio_1, \
            top5_ratio_1, top7_ratio_1, top9_ratio_1, top10_ratio_1, entropy_val_1 = 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if global_counts_1 is not None: # 如果存在Layer 1的数据 (model_type == 4)
                total_tokens_1 = int(np.sum(global_counts_1))
                # 计算Layer 1指标，并用其更新 last_token_counts_np_1
                layer_1_metrics = calculate_single_codebook_metrics(
                    token_counts_np=global_counts_1,
                    total_tokens_in_layer=total_tokens_1,
                    codebook_size=codebook_size,
                    last_token_counts_np_global=last_token_counts_np_1 # 传入该层的全局变量
                )
                # 解包Layer 1的指标
                kl_div_1, used_code_n_1, usage_ratio_1, top1_ratio_1, top3_ratio_1, top5_ratio_1, \
                    top7_ratio_1, top9_ratio_1, top10_ratio_1, entropy_val_1, last_token_counts_np_1 = layer_1_metrics # 更新该层的全局变量
            
            # 层数可以扩展

            # 计算最大可能的熵（当token分布完全均匀时）
            max_entropy = np.log2(codebook_size)

            model.train() # 评估结束后，将模型切回训练模式
            
            # 返回每层的独立指标，总共 (11 * 4 layers) + max_entropy + 2 losses = 47 个返回值
            # [Layer 0 指标 (0-10), Layer 1 指标 (11-21), Layer 2 指标 (22-32), Layer 3 指标 (33-43), max_entropy (44), loss (45, 46)]
            return (
                kl_div_0, used_code_n_0, usage_ratio_0, total_tokens_0, top1_ratio_0, top3_ratio_0, top5_ratio_0, top7_ratio_0, top9_ratio_0, top10_ratio_0, entropy_val_0,
                kl_div_1, used_code_n_1, usage_ratio_1, total_tokens_1, top1_ratio_1, top3_ratio_1, top5_ratio_1, top7_ratio_1, top9_ratio_1, top10_ratio_1, entropy_val_1,
                max_entropy, global_recon_loss, global_comit_loss
            )
        else: # 非主进程
            model.train()
            # 非主进程不进行计算，返回None，由Accelerator框架处理同步
            return None

    if do_evaluate and evaluation_npy_dir and os.path.isdir(evaluation_npy_dir) and val_ratio > 0: # Only setup eval on main process initially
        print(f"✅ Using independent evaluation dataset: {evaluation_npy_dir}")
        val_dataset = NanoporeSignalDataset(shards_dir=evaluation_npy_dir,logic_chunk_size=dataset_logic_chunk_size)
        actual_val_size = max(1, int(val_ratio * len(val_dataset)))
        np.random.seed(42)
        indices = np.random.choice(len(val_dataset), size=actual_val_size, replace=False)
        val_subset = torch.utils.data.Subset(val_dataset, indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=device_micro_batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=True,
            drop_last=True
        )
        # 关键：必须 prepare，让 Accelerate 分发数据分片
        val_loader = accelerator.prepare(val_loader)
    # ========================
    # Model & Optimizer & Scheduler Preparation
    # ========================
    if codebook_type == "VQ" and model_type == 0:
        model = Nanopore_Tokenizer_Model_V0(
            codebook_size=codebook_size,
            codebook_decay=codebook_decay,
            codebook_emadc=codebook_emadc,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            cnn_type=cnn_type,
            init_codebook_path=init_codebook_path,
            cnn_checkpoint_path = cnn_checkpoint_path,
            freeze_cnn = freeze_cnn,
            learnable_codebook=learnable_codebook,
        )
    elif codebook_type == "VQ_distill" and model_type == 1:
        model = Nanopore_Tokenizer_Model_V1(
            codebook_size=codebook_size,
            codebook_decay=codebook_decay,
            codebook_emadc=codebook_emadc,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            cnn_type=cnn_type,
            init_codebook_path=init_codebook_path,
            cnn_checkpoint_path = cnn_checkpoint_path,
            freeze_cnn = freeze_cnn,
            learnable_codebook=learnable_codebook,
            teacher_model_path=teacher_model_path,
        )
    elif codebook_type == "VQ_distill_BERT" and model_type == 2:
        model = Nanopore_Tokenizer_Model_V2(
            codebook_size=codebook_size,
            codebook_decay=codebook_decay,
            codebook_emadc=codebook_emadc,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            cnn_type=cnn_type,
            init_codebook_path=init_codebook_path,
            cnn_checkpoint_path = cnn_checkpoint_path,
            freeze_cnn = freeze_cnn,
            learnable_codebook=learnable_codebook,
            teacher_model_path=teacher_model_path,
        )
    else:
        print("error model type. exit")
        return 
    # No need to manually call .to(device) or wrap with DDP, Accelerate handles it
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # Prepare model, optimizer, and scheduler with Accelerate
    model, optimizer = accelerator.prepare(model, optimizer)


    # Calculate total training steps correctly
    # len(train_dataloader) after prepare gives micro-batch count per epoch
    total_training_global_steps = num_epochs * len(train_dataloader) // accumulation_steps 

    # Scheduler setup (Accelerate requires it to be prepared after prepare(optimizer))
    scheduler = None
    # total_training_global_steps = int(len(train_dataloader) * num_epochs / global_batch_size) # <--- Remove or comment out old calculation

    if accelerator.is_main_process and lr_scheduler_type != "constant":
        print(f"📈 Using LR scheduler: {lr_scheduler_type}, warmup_steps={warmup_steps}")
        print(f"📈 Calculated total_training_global_steps: {total_training_global_steps}") # Optional: Add print to confirm
        print(f"📈 Calculated num_epochs: {num_epochs}") # Optional: Add print to confirm
        print(f"📈 Calculated accumulation_steps: {accumulation_steps}") # Optional: Add print to confirm
        print(f"调试信息:")
        print(f"  len(train_dataloader) = {len(train_dataloader)}")
        print(f"  accumulation_steps = {accumulation_steps}")
        print(f"  num_epochs = {num_epochs}")
        print(f"  计算的总步数 = {total_training_global_steps}")

    # 我们指定的warmup_steps是scheduler参数更新的次数，而不是真正的steps，所以

    if lr_scheduler_type != "constant":

        # 因为accelerate的调度和并行度有关，每次调度其实同时运行了多次
        # 所以要乘以rank数


        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        
        # Ensure warmup_steps does not exceed total steps to prevent errors
        actual_warmup_steps = min(warmup_steps, total_training_global_steps)
        main_steps = max(1, total_training_global_steps - actual_warmup_steps)

        scheduler_actual_warmup_steps = actual_warmup_steps * accelerator.num_processes
        scheduler_main_steps = main_steps * accelerator.num_processes

        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=warmup_start_factor, 
            end_factor=warmup_end_factor, 
            total_iters=scheduler_actual_warmup_steps
        )
        
        if lr_scheduler_type == "cosine":
            eta_min = lr * main_scheduler_end_factor # e.g., 5e-5 * 1e-6 = 5e-11
            main_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=scheduler_main_steps, 
                eta_min=eta_min
            )
        elif lr_scheduler_type == "linear":
            # Adjust end factor relative to warmup end factor for LinearLR continuity
            relative_end_factor = max(1e-8, min(1.0, main_scheduler_end_factor / warmup_end_factor)) if warmup_end_factor != 0 else main_scheduler_end_factor
            main_scheduler = LinearLR(
                optimizer, 
                start_factor=1.0, 
                end_factor=relative_end_factor, 
                total_iters=scheduler_main_steps
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")

        # Combine warmup and main scheduler
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[scheduler_actual_warmup_steps]
        )
        
        # Prepare scheduler with Accelerate
        scheduler = accelerator.prepare(scheduler)
    # ========================
    # Resume from Checkpoint
    # ========================
    start_epoch = start_spoch = start_global_step = 0
    if checkpoint_path and os.path.isdir(checkpoint_path):
        # 1. 首先检查 metadata.json 是否存在且完整
        metadata = load_checkpoint_metadata(checkpoint_path)
        if metadata:
            # 定义恢复训练所必需的元数据键
            REQUIRED_METADATA_KEYS = {'epoch', 'spoch', 'global_step'}
            # 2. 检查 metadata 是否包含所有必需的键
            missing_keys = REQUIRED_METADATA_KEYS - metadata.keys()
            if not missing_keys:
                # 所有必需的键都存在
                try:
                    if accelerator.is_main_process:
                        print(f"📥 Loading checkpoint state and metadata from: {checkpoint_path}")
                    # 4. 解析所有必需的元数据信息
                    # 由于前面已确认键存在，这里可以直接访问
                    start_epoch = metadata['epoch'] + 1
                    start_spoch = metadata['spoch'] + 1
                    start_global_step = metadata['global_step']
                    model_type = metadata['model_type']
                    dynamic_commitment_weight = metadata.get('dynamic_commitment_weight', commitment_weight)
                    if accelerator.is_main_process:
                        print(f"✅ Successfully resumed from epoch {start_epoch}, spoch {start_spoch}, global_step {start_global_step}")
                    # 3. 加载加速器状态 (模型, 优化器等)
                    accelerator.load_state(checkpoint_path)
                    # 标记为加载成功
                    load_successful = True 
                except Exception as e:
                    # 加速器状态加载失败
                    if accelerator.is_main_process:
                        print(f"❌ Failed to load checkpoint state from {checkpoint_path}, despite valid metadata. Error: {e}. Starting from scratch.")
            else:
                # metadata.json 存在，但缺少必要的键
                if accelerator.is_main_process:
                    print(f"❌ metadata.json exists but is missing required keys: {missing_keys}. Cannot resume safely. Starting from scratch.")
        else:
            # metadata.json 文件不存在
            if accelerator.is_main_process:
                print(f"❌ metadata.json not found in {checkpoint_path}. Cannot verify checkpoint integrity. Starting from scratch.")
    else:
        if accelerator.is_main_process:
            print("⚠️ No checkpoint path provided or path is not a directory. Starting from scratch.") 




    if  accelerator.is_main_process: 
        print(f"Gradient Accumulation Steps: {accumulation_steps}")
        print(f"Num Processes (GPUs): {accelerator.num_processes}")
        print(f"actual_warmup_steps: {actual_warmup_steps}")

    # ========================
    # Gradient Accumulation Setup (Handled by Accelerator)
    # ========================
    # Calculation moved to before Accelerator init for logging
    # The accumulation_steps value is passed to Accelerator during initialization.
    # No further manual setup needed here.
    # Print configuration AFTER accelerator.prepare
    # ========================
    # Training Loop
    # ========================
    model.train()
    global_step = start_global_step
    spoch = start_spoch
    total_steps = len(train_dataloader) * num_epochs
    total_global_steps = total_training_global_steps
    total_spochs = total_steps // update_loss_weight_every

    distill_loss = None
    # 在训练循环开始前
    if accelerator.is_main_process:
        print(f"\n=== 调度器详细调试 ===")
        print(f"配置值:")
        print(f"  lr: {lr}")
        print(f"  warmup_start_factor: {warmup_start_factor}")
        print(f"  warmup_end_factor: {warmup_end_factor}")
        print(f"  warmup_steps (配置): {warmup_steps}")
        print(f"  actual_warmup_steps (计算): {actual_warmup_steps}")
        
        # 检查调度器的 warmup 部分
        if hasattr(scheduler, '_schedulers'):
            warmup_sched = scheduler._schedulers[0]
            print(f"\nWarmup调度器:")
            print(f"  类型: {type(warmup_sched)}")
            print(f"  start_factor: {warmup_sched.start_factor}")
            print(f"  end_factor: {warmup_sched.end_factor}")
            print(f"  total_iters: {warmup_sched.total_iters}")
        
        print(f"\n初始学习率: {scheduler.get_last_lr()[0]:.6e}")
        print("=== 调试结束 ===\n")
    log_dict = {}

    wandb_kldiv = 0.0
    wandb_total_tokens = 0
    wandb_codebook_usage = 0
    wandb_codebook_used  = 0
    wandb_codebook_top1_ratio = 0 
    wandb_codebook_top3_ratio = 0
    wandb_codebook_top5_ratio = 0
    wandb_codebook_top7_ratio = 0
    wandb_codebook_top9_ratio = 0
    wandb_codebook_top10_ratio = 0 
    wandb_codebook_entropy = 0
 
    wandb_kldiv_1 = 0.0
    wandb_total_tokens_1 = 0
    wandb_codebook_usage_1 = 0
    wandb_codebook_used_1  = 0
    wandb_codebook_top1_ratio_1 = 0 
    wandb_codebook_top3_ratio_1 = 0
    wandb_codebook_top5_ratio_1 = 0
    wandb_codebook_top7_ratio_1 = 0
    wandb_codebook_top9_ratio_1 = 0
    wandb_codebook_top10_ratio_1 = 0 
    wandb_codebook_entropy_1 = 0

    wandb_codebook_max_entropy = 0
    wandb_eval_recon_loss = 0
    wandb_eval_comit_loss = 0

    if use_dynamic_commitment_weight:
        init_commitment_weight = commitment_weight
        dynamic_commitment_weight = init_commitment_weight
    else:
        init_commitment_weight = commitment_weight
        dynamic_commitment_weight = init_commitment_weight

    target_commitment_weight = init_commitment_weight 

    runtime_dict = {
        "target_commitment_weight":target_commitment_weight
    } 
    write_runtime_json(runtime_dict)

    train_start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for the sampler if applicable (handled by Accelerate)
        train_dataloader.sampler.set_epoch(epoch) if hasattr(train_dataloader.sampler, 'set_epoch') else None

        # Zero grad at the beginning of an epoch (or more precisely, before the first accumulation cycle)
        # Accelerate's optimizer handles the zero_grad call internally when needed,
        # but for explicit control and clarity, especially with manual accumulation, it's good practice.
        # However, Accelerate often calls zero_grad automatically *before* the first backward() in a new step cycle.
        # The safest place is right before the first backward() call in a new accumulation cycle.
        # We'll handle this inside the loop based on the accumulation logic.
        # With accelerator.accumulate, zero_grad is handled automatically at the start of each accum cycle.
        for step, batch in enumerate(train_dataloader):
            # Use Accelerate's accumulate context manager
            with accelerator.accumulate(model):
                # 这里面的代码在每个 micro-batch 都会执行
                # 包括前向传播、反向传播
                # No need to manually move batch to device
                x = batch
                # Forward pass
                if codebook_type == "VQ" and model_type == 0:
                    recon, indices, break_loss, loss_breakdown = model(x)
                    recon_loss = F.mse_loss(recon, x)
                    comit_loss = loss_breakdown.commitment
                    diver_loss = loss_breakdown.codebook_diversity
                    ortho_loss = loss_breakdown.orthogonal_reg
                    total_loss = recon_loss + comit_loss * dynamic_commitment_weight
                elif codebook_type == "VQ_distill" and model_type == 1:
                    recon, indices, break_loss, loss_breakdown, distill_loss = model(x)
                    recon_loss = F.mse_loss(recon, x)
                    comit_loss = loss_breakdown.commitment
                    diver_loss = loss_breakdown.codebook_diversity
                    ortho_loss = loss_breakdown.orthogonal_reg
                    total_loss = recon_loss + comit_loss * dynamic_commitment_weight + distill_loss * 1.0
                elif codebook_type == "VQ_distill_BERT" and model_type == 2:
                    recon, indices, break_loss, loss_breakdown, distill_loss = model(x)
                    recon_loss = F.mse_loss(recon, x)
                    comit_loss = loss_breakdown.commitment
                    diver_loss = loss_breakdown.codebook_diversity
                    ortho_loss = loss_breakdown.orthogonal_reg
                    total_loss = recon_loss + comit_loss * dynamic_commitment_weight + distill_loss * 1.0
                elif codebook_type == "RVQ":
                    recon, indices, all_loss, all_codes = model(x)
                    recon_loss = F.mse_loss(recon, x)
                    comit_loss = torch.tensor(0.0)
                    diver_loss = torch.tensor(0.0)
                    ortho_loss = torch.tensor(0.0)


                # 💡 ACTUAL LOSS: Fixed weights. DWA is NOT applied here.
                # total_loss = recon_loss + comit_loss * dynamic_commitment_weight

                # Scale the loss by the number of accumulation steps for averaging
                # Accelerate's backward function handles gradient scaling for mixed precision automatically
                # The loss is automatically scaled by 1/accumulation_steps inside the context if needed.
                accelerator.backward(total_loss) # Pass the unscaled loss

                # 梯度裁剪：必须在 step 之前，且在 accumulate 块内
                if accelerator.sync_gradients and gradient_clipping is not None:
                    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping)
                # 只在梯度累积完成后执行优化器步骤
                #if accelerator.sync_gradients:
                # 执行梯度裁剪（如果启用）
                #if gradient_clipping is not None:
                #    accelerator.clip_grad_norm_(model.parameters(), gradient_clipping)
                # 使用 Accelerate 的方法执行优化器步骤
                optimizer.step() 
                scheduler.step()
                optimizer.zero_grad()
                
                     # 重要：这里不需要调用 optimizer.zero_grad()，Accelerate 会自动处理
            # Determine if it's time to update weights (based on effective steps)
            # We use accelerator.sync_gradients to know when the optimizer step was just performed
            
            # --- Post-update operations (logging, evaluation, checkpointing) ---
            if accelerator.sync_gradients:
                # These happen after a full parameter update
                global_step += 1 # Increment global step for every micro-step processed
                g_recon = recon_loss.item() # Get value from last micro-step
                g_comit = comit_loss.item()
                g_ortho = ortho_loss.item()
                g_diver = diver_loss.item()
                g_total = total_loss.item()
                if distill_loss is not None:
                    g_distill = distill_loss.item()
                else:
                    g_distill = 0.0

                # Log metrics (main process only)
                if accelerator.is_main_process:
                    current_lr = optimizer.param_groups[0]['lr']
                    # 为了避免log10(0)的问题，通常会给一个小的epsilon值
                    epsilon = 1e-8
                    # 计算log10值，确保非负值
                    g_recon_log10 = math.log10(g_recon + epsilon) if g_recon > 0 else math.log10(epsilon)
                    g_comit_log10 = math.log10(g_comit + epsilon) if g_comit > 0 else math.log10(epsilon)
                    g_recon_per_comit = g_recon/(g_comit + epsilon)
                    lr_log10 = math.log10(current_lr + epsilon)
                    expected_commitment_weight = 0
                    if use_dynamic_commitment_weight and global_step > commitment_weight_freeze_steps:
                        #expected_commitment_weight = target_commitment_weight
                        expected_commitment_weight = g_recon_per_comit/commitment_weight_rpc 
                        if dynamic_commitment_weight < expected_commitment_weight:
                            dynamic_commitment_weight += (expected_commitment_weight - dynamic_commitment_weight)*commitment_weight_lr

                    log_and_save(
                        epoch=epoch,
                        global_step=global_step,
                        total_epochs=num_epochs,
                        total_global_steps=total_global_steps,
                        train_start_time=train_start_time,
                        epoch_total_steps=len(train_dataloader),
                        avg_recon_loss=g_recon,
                        avg_total_loss=g_total,
                        avg_comit_loss=g_comit,
                        avg_diver_loss=g_diver,
                        avg_ortho_loss=g_ortho,
                        avg_rperc_ratio=g_recon_per_comit,
                        codebook_usage=wandb_codebook_usage, # Placeholder, updated later if eval runs
                        loss_csv_path=loss_csv_path,
                        lr=current_lr,
                        accelerator=accelerator # Pass accelerator instance
                    )
                    # Prepare log dict for WandB
                    log_dict.update({
                        "train/recon_loss": g_recon,
                        "train/recon_loss_log10": g_recon_log10,
                        "train/comit_loss": g_comit,
                        "train/comit_loss_log10": g_comit_log10,
                        "train/ortho_loss": g_ortho,
                        "train/diver_loss": g_diver,
                        "train/total_loss": g_total,
                        "train/distill_loss": g_distill,
                        "comit/dynamic_commit_weight": dynamic_commitment_weight,
                        "comit/expected_commit_weight": expected_commitment_weight,
                        "train/recon_per_comit": g_recon_per_comit,
                        "evaluate/total_tokens": wandb_total_tokens, # Placeholder
                        "evaluate/recon_loss": wandb_eval_recon_loss, # Placeholder
                        "evaluate/comit_loss": wandb_eval_comit_loss, # Placeholder

                        "codebook/kldiv": wandb_kldiv, # Placeholder
                        "codebook/usage": wandb_codebook_usage, # Placeholder
                        "codebook/used_codes": wandb_codebook_used, # Placeholder
                        "codebook/entropy": wandb_codebook_entropy, # Placeholder
                        "codebook/max_entropy": wandb_codebook_max_entropy, # Placeholder
                        "codebook/top1_ratio": wandb_codebook_top1_ratio, # Placeholder
                        "codebook/top3_ratio": wandb_codebook_top3_ratio, # Placeholder
                        "codebook/top5_ratio": wandb_codebook_top5_ratio, # Placeholder
                        "codebook/top7_ratio": wandb_codebook_top7_ratio, # Placeholder
                        "codebook/top9_ratio": wandb_codebook_top9_ratio, # Placeholder
                        "codebook/topx_ratio": wandb_codebook_top10_ratio, # Placeholder
 
                        "codebook1/kldiv": wandb_kldiv_1, # Placeholder
                        "codebook1/usage": wandb_codebook_usage_1, # Placeholder
                        "codebook1/used_codes": wandb_codebook_used_1, # Placeholder
                        "codebook1/entropy": wandb_codebook_entropy_1, # Placeholder
                        "codebook1/top1_ratio": wandb_codebook_top1_ratio_1, # Placeholder
                        "codebook1/top3_ratio": wandb_codebook_top3_ratio_1, # Placeholder
                        "codebook1/top5_ratio": wandb_codebook_top5_ratio_1, # Placeholder
                        "codebook1/top7_ratio": wandb_codebook_top7_ratio_1, # Placeholder
                        "codebook1/top9_ratio": wandb_codebook_top9_ratio_1, # Placeholder
                        "codebook1/topx_ratio": wandb_codebook_top10_ratio_1, # Placeholder
   
                        "learning_rate": current_lr,
                        "lr_log10": lr_log10,
                        "epoch": epoch + 1,
                        "global_step": global_step, # Useful for plotting against global steps
                    })
                    accelerator.log(log_dict, step=global_step) # <--- 添加这一行
                # --- Evaluation ---
                # Run evaluation based on spoch
                if global_step % evaluate_every_spoch == 0:
                    accelerator.wait_for_everyone() # Sync all processes before eval
                    try:
                        #kl_div_0, used_code_n_0, usage_ratio_0, total_tokens_0,
                        #top1_ratio_0, top3_ratio_0, top5_ratio_0, top7_ratio_0, top9_ratio_0, top10_ratio_0,
                        #entropy_val_0,
                        #kl_div_1, used_code_n_1, usage_ratio_1, total_tokens_1,
                        #top1_ratio_1, top3_ratio_1, top5_ratio_1, top7_ratio_1, top9_ratio_1, top10_ratio_1,
                        #entropy_val_1,
                        #max_entropy,
                        #global_recon_loss,
                        #global_comit_loss

                        result = evaluate_codebook_metrics_v3()
                        if accelerator.is_main_process and result is not None:
                            (
                               wandb_kldiv,
                               wandb_codebook_used,
                               wandb_codebook_usage, 
                               wandb_total_tokens, 
                               wandb_codebook_top1_ratio, 
                               wandb_codebook_top3_ratio,
                               wandb_codebook_top5_ratio, 
                               wandb_codebook_top7_ratio, 
                               wandb_codebook_top9_ratio,
                               wandb_codebook_top10_ratio, 
                               wandb_codebook_entropy, 
 
                               wandb_kldiv_1,
                               wandb_codebook_used_1,
                               wandb_codebook_usage_1, 
                               wandb_total_tokens_1, 
                               wandb_codebook_top1_ratio_1, 
                               wandb_codebook_top3_ratio_1,
                               wandb_codebook_top5_ratio_1, 
                               wandb_codebook_top7_ratio_1, 
                               wandb_codebook_top9_ratio_1,
                               wandb_codebook_top10_ratio_1, 
                               wandb_codebook_entropy_1, 
                            
                               wandb_codebook_max_entropy,
                               wandb_eval_recon_loss,
                               wandb_eval_comit_loss
                               ) = result
                            print(f"Effective Step {global_step} - Codebook Usage: {wandb_codebook_usage:.2%} wandb_kldiv: {wandb_kldiv}")
                    except Exception as e:
                            print(f"Error during evaluation: {e}")

                    try:
                        runtime_json = read_runtime_json() 
                        if runtime_json is not None:
                            target_commitment_weight = runtime_json["target_commitment_weight"] 
                    except Exception as e:
                        print(f"Error during read runtime_json: {e}")

                    accelerator.wait_for_everyone() # Sync all processes after eval

                # Periodic checkpointing (main process only)
                if accelerator.is_main_process and global_step  % save_checkpoint_every_spoch == 0:
                    ckpt_path = f"{output_model_path}.step{global_step}.pth"
                    save_full_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler, # Pass scheduler state
                        epoch=epoch,
                        spoch=spoch,
                        global_step=global_step,
                        cnn_type=cnn_type,
                        model_type=model_type,
                        dynamic_commitment_weight=dynamic_commitment_weight,
                        accelerator=accelerator # Pass accelerator instance
                    )
                # --- End of if accelerator.sync_gradients block ---
                # --- End of with accelerator.accumulate(model) block ---
                # optimizer.step() and optimizer.zero_grad() are called automatically by Accelerate inside the context
                # when accelerator.sync_gradients becomes True (i.e., at the end of an accumulation cycle).


    # Final save (main process only)
    if accelerator.is_main_process:
        ckpt_path = f"{output_model_path}.final.pth"
        save_full_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=num_epochs - 1,
            spoch=spoch,
            global_step=global_step,
            cnn_type=cnn_type,
            model_type=model_type,
            dynamic_commitment_weight=dynamic_commitment_weight,
            accelerator=accelerator
        )
    # Clean up Accelerator resources
    accelerator.end_training()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Nanopore VQ Tokenizer with Accelerate")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Pass the configuration dictionary values to vqe_train
    # Added new Accelerate-specific arguments
    # 从嵌套结构中读取配置
    vqe_train(
        # 数据配置
        train_npy_dir=config.get("data", {}).get("train_npy_dir"),
        evaluation_npy_dir=config.get("data", {}).get("evaluation_npy_dir"),
        num_workers=config.get("data", {}).get("num_workers", 8),
        val_ratio=config.get("data", {}).get("val_ratio", 0.1),
        prefetch_factor=config.get("data", {}).get("prefetch_factor", 128),
        dataset_logic_chunk_size=config.get("data", {}).get("dataset_logic_chunk_size", 6000),
        chunk_size=config.get("data", {}).get("chunk_size", 12000),
        
        # 模型配置
        model_type=config.get("model", {}).get("model_type", 1),
        codebook_size=config.get("model", {}).get("codebook_size", 8192),
        cnn_type=config.get("model", {}).get("cnn_type", 1),
        learnable_codebook=config.get("model", {}).get("learnable_codebook", True),
        init_codebook_path=config.get("model", {}).get("init_codebook_path", ""),
        cnn_checkpoint_path=config.get("model", {}).get("cnn_checkpoint_path", ""),
        freeze_cnn=config.get("model", {}).get("freeze_cnn", 0),
        codebook_type=config.get("model", {}).get("codebook_type", "VQ"),
        teacher_model_path=config.get("model", {}).get("teacher_model_path", ""),
        
        # 训练配置
        lr=config.get("training", {}).get("lr", 3e-4),
        num_epochs=config.get("training", {}).get("num_epochs", 10),
        global_batch_size=config.get("training", {}).get("global_batch_size", 256),
        device_micro_batch_size=config.get("training", {}).get("device_micro_batch_size", 16),
        mixed_precision=config.get("training", {}).get("mixed_precision", "bf16"),
        gradient_clipping=config.get("training", {}).get("gradient_clipping", 1.0),
        cpu=config.get("training", {}).get("cpu", False),
        
        # 损失配置
        commitment_weight=config.get("loss", {}).get("commitment_weight", 0.25),
        use_dynamic_commitment_weight=config.get("loss", {}).get("use_dynamic_commitment_weight", True),
        commitment_weight_freeze_steps=config.get("loss", {}).get("commitment_weight_freeze_steps", 20000),
        commitment_weight_lr=config.get("loss", {}).get("commitment_weight_lr", 0.01),
        commitment_weight_rpc=config.get("loss", {}).get("commitment_weight_rpc", 1),
        codebook_diversity_loss_weight=config.get("loss", {}).get("codebook_diversity_loss_weight", 0.0),
        orthogonal_reg_weight=config.get("loss", {}).get("orthogonal_reg_weight", 0.0),
        
        # 学习率调度配置
        lr_scheduler_type=config.get("lr_scheduler", {}).get("type", "cosine"),
        warmup_steps=config.get("lr_scheduler", {}).get("warmup_steps", 100),
        warmup_start_factor=config.get("lr_scheduler", {}).get("warmup_start_factor", 1e-5),
        warmup_end_factor=config.get("lr_scheduler", {}).get("warmup_end_factor", 1.0),
        main_scheduler_end_factor=config.get("lr_scheduler", {}).get("main_scheduler_end_factor", 1e-5),
        
        # 检查点配置
        output_model_path=config.get("checkpoint", {}).get("output_model_path", "nanopore_vq_tokenizer.pth"),
        loss_csv_path=config.get("checkpoint", {}).get("loss_csv_path", "train_loss.csv"),
        save_checkpoint_every_spoch=config.get("checkpoint", {}).get("save_checkpoint_every_spoch", 10),
        update_loss_weight_every=config.get("checkpoint", {}).get("update_loss_weight_every", 10),
        
        # 评估配置
        do_evaluate=config.get("evaluation", {}).get("do_evaluate", False),
        evaluate_every_spoch=config.get("evaluation", {}).get("evaluate_every_spoch", 10),
        
        # WandB配置
        use_wandb=config.get("wandb", {}).get("use_wandb", True),
        wandb_project=config.get("wandb", {}).get("project", "nanopore_vq"),
        wandb_name=config.get("wandb", {}).get("name", "default_run"),
        
        # 其他参数
        codebook_decay=config.get("codebook_decay", 0.99),  # 注意：这个参数在之前的配置中没有
        codebook_emadc=config.get("codebook_emadc", 2),
        checkpoint_path=config.get("checkpoint_path"),

        seed=config.get("seed", 42),
    )

if __name__ == "__main__":
    main()
