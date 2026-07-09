"""
PoreCodec 纳米孔信号量化Tokenizer训练主程序
支持多路径二进制 memmap 混合洗牌分布式训练、OLMo 风格显式检查点控制与防御性环境校验
"""

import argparse
import os
import yaml
import json
import torch
import logging
import numpy as np
from dataclasses import asdict
from datetime import timedelta

# 从原生 PyTorch 引入分布式组件与基础 Dataset 架构
import torch.distributed as dist
from torch.utils.data import Subset

# 🌟 引入 Transformers 标准训练参数组件
from transformers import TrainingArguments

# 严格对齐你重构后的强类型配置与组件结构
from config.train_config import TrainConfig
from trainer import PoreTrainer
from dataset import PoreSignalDataset  # 🌟 名字对齐最终版类名
from wrappers.pore_rsq_wrapper import PoreRSQWrapper
from wrappers.pore_vq_wrapper import PoreVQWrapper
from monitor.codebook_monitor import CodebookMonitor
from callbacks.codebook_callback import CodebookCallback

# 找到你的 import 区域，添加这一行：
from modeling_pore_codec import PoreRSQCodec, PoreRSQCodecConfig
from modeling_pore_vq_codec import PoreVQCodec, PoreVQCodecConfig

# 初始化全局日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train")


def safe_barrier():
    """安全进程屏障：只有在分布式环境初始化后才进行同步，防止单卡挂起"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def train(cfg: TrainConfig):
    # 获取当前进程的分布式角色
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # ===========================================================================
    # [1] 优雅地格式化打印所有训练参数 (仅在主节点打印)
    # ===========================================================================
    if rank == 0:
        logger.info("=" * 40 + " [TRAINING CONFIGURATION] " + "=" * 40)
        cfg_json = json.dumps(asdict(cfg), indent=4, sort_keys=True, ensure_ascii=False)
        logger.info(f"\n{cfg_json}")
        logger.info("=" * 104)

    # ===========================================================================
    # [2] 模仿 OLMo: 显式配置覆盖安全性校验 (Overwrite Guard)
    # ===========================================================================
    config_save_path = os.path.join(cfg.save_folder, "config.yaml")
    save_overwrite = getattr(cfg, "save_overwrite", False)

    if os.path.exists(config_save_path) and not save_overwrite:
        raise RuntimeError(
            f"❌ [OLMo Guard] Output directory already contains an existing config at '{config_save_path}'. "
            "To prevent overwriting historical experiments, change `save_folder` or set `save_overwrite: true` in your YAML."
        )

    safe_barrier()

    if rank == 0:
        os.makedirs(cfg.save_folder, exist_ok=True)
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(cfg), f, allow_unicode=True)

    safe_barrier()

    # ===========================================================================
    # [3] 显式准备双向 memmap 数据集
    # ===========================================================================
    logger.info(f"📦 [Rank {rank}] Preparing Pure Memmap Datasets...")

    # 3.1 训练集：对齐 PoreSignalDataset 入参
    train_dataset = PoreSignalDataset(
        shard_paths=cfg.train_data.paths,              # 消费嵌套的 train_data.paths 列表
        logic_chunk_size=cfg.train_data.chunk_size,
        buffer_size=cfg.train_data.buffer_size,       # 20000 内存滑窗缓冲区大小
        memmap_dtype=cfg.train_data.memmap_dtype,
        shuffle_buffer=True,                          # 训练集明确开启局部洗牌
        rank=rank,
        world_size=world_size,
        seed=cfg.seed
    )


    val_dataset = PoreSignalDataset(
        shard_paths=cfg.eval_data.paths,               # 消费嵌套的 eval_data.paths 列表
        logic_chunk_size=cfg.eval_data.chunk_size,
        buffer_size=0,                                # 验证集传入 0，禁止任何 Buffer Shuffle
        memmap_dtype=cfg.eval_data.memmap_dtype,
        shuffle_buffer=False,                         # 关闭洗牌
        rank=rank,
        world_size=world_size,
        is_repeat=False,
        seed=cfg.seed
    )

    if rank == 0:
        logger.info(f"✅ Pure deterministic evaluation dataset ready. Total sample size: {len(val_dataset)}")

    safe_barrier()

    # ===========================================================================
    # [4] 初始化模型
    # ===========================================================================
    if rank == 0:
        logger.info(f"Building Nanopore VQE Model Branch: [{cfg.model.model_type}]...")


    safe_barrier()

    # ===========================================================================
    # [5] 全局 Batch 守恒对齐与标准 TrainingArguments 封装
    # ===========================================================================
    per_device_train_batch = cfg.device_train_microbatch_size
    per_device_eval_batch = cfg.device_eval_batch_size
    grad_acc_steps = max(1, cfg.global_train_batch_size // (per_device_train_batch * world_size))

    if rank == 0:
        logger.info(f"📐 Global Batch Size Alignment Matrix:")
        logger.info(f"   -> Target Global Batch: {cfg.global_train_batch_size}")
        logger.info(f"   -> Active GPU Counts: {world_size}")
        logger.info(f"   -> Auto Calculated Gradient Accumulation Steps: {grad_acc_steps}")

    # 🌟 核心修改：将参数严格映射为 Hugging Face 标准的 TrainingArguments 类，干掉自定义无效键
    training_args = TrainingArguments(
        output_dir=cfg.save_folder,
        per_device_train_batch_size=per_device_train_batch,
        per_device_eval_batch_size=per_device_eval_batch,
        gradient_accumulation_steps=grad_acc_steps,
        
        # 优化器与学习率对齐
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        adam_epsilon=cfg.optimizer.eps,
        max_grad_norm=cfg.max_grad_norm,
       
        # Trainer 在运行 train() 方法时，内部会计算总的 max_steps。它的逻辑优先级是：
        # 如果设置了 max_steps：这是硬性截止条件，Trainer 会时刻盯着当前的 global_step，一旦 global_step >= max_steps，立刻无条件终止。
        # 如果没设置 max_steps：Trainer 才会去通过 num_train_epochs 和数据集大小计算出需要跑多少步。
        max_steps=cfg.max_duration,
        # 显式移除 Epoch 限制，让它依赖 max_steps 退出
        # 如果 num_train_epochs=1：
        # 在数据集视角：你是在告诉 Trainer：“我的数据集逻辑上是一次完整的遍历”。
        # 在训练调度视角：由于 max_steps 已经被显式指定为 20000，Trainer 的调度器发现 20000 步远比 1 个 Epoch 对应的步数要大得多。
        # 最终效果：Trainer 会直接忽略“Epoch 结束”这一事件作为停止依据，而完全听从 max_steps 的指挥。
        # 如果你的 IterableDataset 是一个“无限流”（使用了 while True），当你设置 num_train_epochs=1 时，
        # Trainer 跑完第一轮数据流后，发现 global_step 才跑了 5000 步，距离你设定的 20000 步还远，
        # 它不会因为 Epoch 跑完了而停止，而是会自动重置数据迭代器，继续从头读取流式数据，直到 max_steps 达到 20000。
        num_train_epochs=1,
        warmup_steps=cfg.scheduler.t_warmup,
        lr_scheduler_type=cfg.scheduler.name,  # 确保这里传入符合 HF 规范的字符串，如 "cosine"
        
        # 混合精度支持
        fp16=(cfg.precision == "fp16"),
        bf16=(cfg.precision == "bf16"),
        
        # 运行期控制
        save_steps=cfg.save_interval,
        eval_steps=cfg.eval_interval,
        eval_strategy="steps",            # 必须开启步数评估策略，否则 eval_steps 不生效
        
        # 🌟 高性能数据加载硬件参数完美注入标准键名
        dataloader_num_workers=cfg.train_data.num_workers,
        dataloader_pin_memory=cfg.train_data.pin_memory,
        # 🌟 强制要求：关闭数据截断
        # 确保数据流哪怕被读取完，Trainer 也会立即开始从头重新遍历数据，而不是停止
        dataloader_drop_last=False,
       
        # 
        #remove_unused_columns=False,
        # 监控指标看板
        report_to="wandb" if rank == 0 else "none",
        logging_steps=10,                      # 没必要憋太久，每 10 步吐一次训练 Loss

        # 
        ddp_find_unused_parameters=True,

        label_names=["signal"],  # 👈 核心秘籍！告诉 Trainer，我的 "signal" 就是标签！
        # 🌟 显式指定 Hugging Face Trainer 的 run_name。
        #
        # 如果不设置 run_name：
        # Trainer 会默认将 run_name 设置为 output_dir，
        # WandB 因此会打印如下警告：
        #
        #   "The run_name is currently set to the same value as
        #    TrainingArguments.output_dir..."
        #
        # 虽然可以通过：
        #
        #   os.environ["WANDB_NAME"] = cfg.wandb.name
        #
        # 来指定 WandB 页面显示的名称，但如果这个环境变量是在
        # TrainingArguments(...) 创建之后才设置，Trainer 内部已经把
        # run_name 固定为 output_dir，WandB 仍然会产生上述 Warning。
        #
        # 因此，最规范、最符合 Hugging Face 官方设计的方式是：
        #
        #     run_name = cfg.wandb.name
        #
        # 这样 Trainer、WandB 以及最终网页显示的实验名称都会保持一致，
        # 同时无需依赖 WANDB_NAME 环境变量，也不会再出现该 Warning。
        run_name=cfg.wandb.name,
    )
    
    # 🌟 针对 WandB 环境变量在实例化前进行显式注入
    if rank == 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        #os.environ["WANDB_RUN_ID"] = cfg.wandb.name
        # 🌟 核心修改：干掉 WANDB_RUN_ID，改用 WANDB_NAME, 不指定 ID = 每次都是新实验：因为环境里没有 WANDB_RUN_ID，WandB 每次重启训练时，都会在后台自动抓阄生成一个绝对唯一的、随机的 ID（比如 ax78df92）。
        # 指定 Name = 名字随你叫：虽然每次 ID 不同，但因为它们共享同一个 WANDB_NAME="PoreCodec_rerun"，WandB 会把它们在网页端全部显示为 PoreCodec_rerun。
        os.environ["WANDB_NAME"] = cfg.wandb.name

    # ===========================================================================
    # [6] 模仿 OLMo: 显式 load_path 检查点控制与细粒度状态恢复
    # ===========================================================================
    resume_from_checkpoint = None
    weight_only_load_path = None
    load_path = getattr(cfg, "load_path", None)

    if load_path and str(load_path).strip():
        load_path = str(load_path).strip()
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"❌ [OLMo Guard] `load_path` was specified as '{load_path}', but it does not exist!")

        reset_optimizer = getattr(cfg, "reset_optimizer_state", False)
        if reset_optimizer:
            logger.info(f"🔄 [WEIGHT_ONLY_LOAD] Loading model weights from '{load_path}' but RESETTING optimizer/scheduler states.")
            weight_only_load_path = load_path
            resume_from_checkpoint = None
        else:
            logger.info(f"🔄 [FULL_RESUME] OLMo-style explicit resume triggered. Full pipeline recovery from: '{load_path}'")
            resume_from_checkpoint = load_path
    else:
        logger.info("🆕 [FRESH_START] `load_path` is empty or null. Launching a brand new run.")

    safe_barrier()

    # ===========================================================================
    # [7] 实例化 Trainer 并启动分布式流式训练
    # ===========================================================================
    if cfg.model.model_type == "rsq":
        # 1. 构建配置对象
        codec_config = PoreRSQCodecConfig(
            fsq_levels=cfg.model.rsq.levels,
            codebook_size=cfg.model.codebook_size,
            codebook_nqtz=cfg.model.codebook_nqtz,
            cnn_type=cfg.model.rsq.cnn_type
        )
        # 2. 实例化基础 Codec 模型
        raw_codec = PoreRSQCodec(codec_config)
        # 3. 封装预训练包装器 (ModelForPretraining)
        model = PoreRSQWrapper(raw_codec)

        # 2. 关键：手动实例化 Monitor 和 Callback
        # 假设你的量化层在 model.codec.vq
        monitor = CodebookMonitor(codebook_size=cfg.model.codebook_size)
        # 注意：这里我们使用 model.codec.vq 作为 hook 的 target_layer
        codebook_cb = CodebookCallback(monitor=monitor, target_layer=model.codec.vq)

        #callbacks = [codebook_cb]
    elif cfg.model.model_type in ("vq", "vq_distill"):
        teacher_model_path = cfg.model.vq.teacher_model_path
        if cfg.model.model_type == "vq_distill" and not teacher_model_path:
            raise ValueError("❌ `model_type: vq_distill` requires `model.vq.teacher_model_path`.")

        codec_config = PoreVQCodecConfig(
            codebook_size=cfg.model.codebook_size,
            codebook_decay=cfg.model.vq.codebook_decay,
            codebook_emadc=cfg.model.vq.codebook_emadc,
            commitment_weight=cfg.loss_weights.commitment_weight,
            codebook_diversity_loss_weight=cfg.loss_weights.codebook_diversity_loss_weight,
            orthogonal_reg_weight=cfg.loss_weights.orthogonal_reg_weight,
            cnn_type=cfg.model.vq.cnn_type,
            learnable_codebook=cfg.model.vq.learnable_codebook,
            init_codebook_path=cfg.model.vq.init_codebook_path,
            cnn_checkpoint_path=cfg.model.vq.cnn_checkpoint_path,
            freeze_cnn=cfg.model.vq.freeze_cnn,
            teacher_model_path=teacher_model_path,
        )
        raw_codec = PoreVQCodec(codec_config)
        model = PoreVQWrapper(
            raw_codec,
            commitment_weight=cfg.loss_weights.commitment_weight,
            codebook_diversity_loss_weight=cfg.loss_weights.codebook_diversity_loss_weight,
            orthogonal_reg_weight=cfg.loss_weights.orthogonal_reg_weight,
            distill_loss_weight=cfg.model.vq.distill_loss_weight,
        )
    else:
        raise ValueError(f"❌ Unsupported model architecture type: {cfg.model.model_type}")

    if weight_only_load_path:
        possible_weight_paths = [
            os.path.join(weight_only_load_path, "pytorch_model.bin"),
            os.path.join(weight_only_load_path, "model.safetensors"),
        ]
        loaded = False
        target_model = model.codec if hasattr(model, "codec") else model
        for p_path in possible_weight_paths:
            if os.path.exists(p_path):
                if p_path.endswith(".bin"):
                    state_dict = torch.load(p_path, map_location="cpu")
                else:
                    from safetensors.torch import load_file

                    state_dict = load_file(p_path, device="cpu")
                target_model.load_state_dict(state_dict, strict=True)
                logger.info(f"✅ Successfully injected weights into model from {p_path}")
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"❌ Could not find model weights file inside {weight_only_load_path} to perform weight-only resume.")



    trainer = PoreTrainer(
        cfg=cfg,
        model=model,
        args=training_args,            # 🌟 此时传入的是真正、合规的 TrainingArguments 实例对象
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #callbacks=callbacks # 🌟 传入 Trainer，实现即插即用
    )


    if rank == 0:
        logger.info("🚀 Starting production-ready memmap pipeline via StreamRSQTrainer...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train Nanopore VQ Tokenizer via StreamRSQTrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30)
        )
        logger.info(f"卫星网络协同就绪: 进程组握手成功. Global Rank: {os.environ['RANK']}, Local Rank: {local_rank}")

    train(cfg)


if __name__ == "__main__":
    main()
