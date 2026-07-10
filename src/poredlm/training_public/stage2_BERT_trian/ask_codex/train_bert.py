"""
PoreBERT 纳米孔生物大模型 MLM 预训练主程序
支持离线 Token ID 流式加载、动态 Masking 与 OLMo 风格显式检查点控制
"""

import argparse
import os
import yaml
import json
import torch
import logging
from dataclasses import asdict
from datetime import timedelta
import torch.nn as nn
# 分布式组件与基础 Dataset 架构
import torch.distributed as dist
from torch.utils.data import Subset
from itertools import islice  # 1. 记得导入
# Transformers 标准组件
from transformers import TrainingArguments

# 将 AutoConfig.from_pretrained("bert-base-uncased") 等替换为 Megatron 架构
from transformers import (
    MegatronBertConfig, MegatronBertForMaskedLM,
    AutoConfig, BertForMaskedLM, 
    RobertaConfig, RobertaForMaskedLM,
    DebertaV2Config, DebertaV2ForMaskedLM,
    ElectraConfig, ElectraForMaskedLM
)
# 严格对齐配置与组件结构
#from src.config.train_config import TrainBERTConfig
from src.config.train_bert_config import TrainBERTConfig # 确保导入路径正确

#from src.dataset import FlowMapDataset 
from flowmap import FlowMapDataset
from src.wrappers.pore_bert_wrapper import PoreBERTWrapper
from src.trainer_bert import PoreBertTrainer  # 假设你新建了对应的 Trainer
from src.config.constants import SPECIAL_TOKENS, TOKEN_OFFSET, VOCAB_SIZE
from src.models.simple_bert import SimpleBERTForMaskedLM, SimpleBERTConfig # 导入你的模型
# 初始化全局日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_bert")


def safe_barrier():
    """安全进程屏障：只有在分布式环境初始化后才进行同步，防止单卡挂起"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def pore_bert_collate_fn(batch):
    """
    🌟 优化版 BERT 专属数据整理器 (Collate Function)
    职责：
    1. 严格校验：确保数据源头已包含 BOS 和 EOS，严禁非法的边界截断。
    2. 偏移映射：对原始信号 Token 应用 TOKEN_OFFSET，隔离特殊 Token 空间。
    3. 结构重构：在序列最前端显式注入 [CLS] 标记，构建 [CLS][BOS]...[EOS] 完整输入。

    ⚠️ 开发者重要提示 (Attention Masking Policy)：
    - 本函数生成的序列结构为: [CLS, BOS, Signal_Token_1, ..., Signal_Token_N, EOS]
    - 在后续的 Wrapper 或 Trainer 中，必须对特殊 Token (CLS, BOS, EOS) 采取特殊的 Attention 遮掩逻辑：
      1. CLS/BOS/EOS 必须始终参与 Self-Attention 计算 (Attention Mask 设为 1)。
      2. 严禁对 [CLS], [BOS], [EOS] 进行 MLM 掩码 (即不能将它们替换为 [MSK])。
      3. 若在微调阶段处理变长序列，请确保 Padding Token (ID=0) 所在的 Attention 位置被设为 0。
    """
    processed_signals = []

    for b in batch:
        raw_signal = b["data"]
        sample_id = b.get('id', 'Unknown')
        if False:
            # 统一提取预览数据
            head = raw_signal[:10]
            tail = raw_signal[-10:]
            print(f"ID: {sample_id} | Head: {head} | Tail: {tail}")

        # 1. 严格边界校验
        # 确保数据完整性：第0位必须是 BOS，最后一位必须是 EOS
        if raw_signal[0] != SPECIAL_TOKENS["BOS"] or raw_signal[-1] != SPECIAL_TOKENS["EOS"]:
            raise ValueError(
                f"🚨 [数据完整性校验失败] 信号 ID {b.get('id', 'Unknown')} 的边界符异常！"
                f"期望首尾为 {SPECIAL_TOKENS['BOS']}/{SPECIAL_TOKENS['EOS']}，"
                f"实际读取到 {raw_signal[0]}/{raw_signal[-1]}"
            )
        
        # 2. 应用偏移并重构序列
        # 取出原始信号部分 (不包含首尾)
        raw_content = raw_signal[1:-1]

        # 偏移映射：[raw_id + 128]
        mapped_content = raw_content.to(torch.long) + TOKEN_OFFSET

        # 3. 构建标准序列: [CLS] + [BOS] + [信号] + [EOS]
        final_signal = torch.cat([
            torch.tensor([SPECIAL_TOKENS["CLS"]], dtype=torch.long),
            torch.tensor([SPECIAL_TOKENS["BOS"]], dtype=torch.long),
            mapped_content,
            torch.tensor([SPECIAL_TOKENS["EOS"]], dtype=torch.long)
        ])

        processed_signals.append(final_signal)

    # 4. 堆叠并生成 Mask
    token_ids = torch.stack(processed_signals)

    # 物理掩码：只有 ID=0 的 PAD 位置会被置为 0，其余包括特殊 Token 全部参与计算
    attention_mask = (token_ids != SPECIAL_TOKENS["PAD"]).long()

    # 5. 安全性越界安检
    max_id = token_ids.max().item()
    # 校验上限：根据你的词表大小进行调整
    if max_id >= VOCAB_SIZE:
        raise ValueError(f"🚨 [严重越界] 发现非法的 Token ID: {max_id}！这会击穿 Embedding 层。")

    return {
        "data": token_ids,
        "attention_mask": attention_mask
        #"chunk_id": chunk_ids
    }




def train(cfg: TrainBERTConfig):
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
        logger.info("=" * 40 + " [BERT TRAINING CONFIGURATION] " + "=" * 40)
        cfg_json = json.dumps(asdict(cfg), indent=4, sort_keys=True, ensure_ascii=False)
        logger.info(f"\n{cfg_json}")
        logger.info("=" * 109)

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
    # [3] 显式准备双向 memmap 数据集 (完美复用底层逻辑)
    # ===========================================================================
    logger.info(f"📦 [Rank {rank}] Preparing Pre-Tokenized Memmap Datasets for BERT...")

    train_dataset = FlowMapDataset(
        shard_paths=cfg.train_data.paths,
        buffer_size=cfg.train_data.buffer_size,
        memmap_dtype=cfg.train_data.memmap_dtype, # 这里填 "uint16" 或 "float32" 均可
        shuffle_buffer=False,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        data_name=cfg.train_data.data_name,
        memmap_cache_capacity = 1024,
        verbose = True,
    )

    val_dataset = FlowMapDataset(
        shard_paths=cfg.eval_data.paths,
        buffer_size=0,  # 验证集严格禁止 Shuffle Buffer
        memmap_dtype=cfg.eval_data.memmap_dtype,
        shuffle_buffer=False,
        rank=rank,
        world_size=world_size,
        is_repeat=False,
        seed=cfg.seed,
        data_name=cfg.eval_data.data_name,
        memmap_cache_capacity = 16
    )


    if rank == 0:
        logger.info(f"✅ Pure deterministic evaluation dataset ready. Total sample size: {len(val_dataset)}")

    safe_barrier()

    # ===========================================================================
    # [4] 依赖注入模式：初始化 BERT 基座与 Wrapper
    # ===========================================================================
    if rank == 0:
        logger.info(f"🧬 Building Biological Foundation Model Branch: [PoreBERT]")

    # -------------------------------------------------------------------
    # 1. 强制校验防线 (Strict Configuration Guard)
    # -------------------------------------------------------------------
    required_model_keys = [
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
        "bert_model_type"
    ]

    missing_keys = []
    for key in required_model_keys:
        # 检查属性是否存在，且值不为 None
        if not hasattr(cfg.model, key) or getattr(cfg.model, key) is None:
            missing_keys.append(key)

    if missing_keys:
        error_msg = f"🚨 [配置校验失败] YAML 配置文件中的 [model] 模块缺少以下必填参数: {missing_keys}。严禁使用默认值，请补全配置！"
        if rank == 0:
            logger.error(error_msg)
        raise ValueError(error_msg) # 使用 raise 确保 DDP 多卡环境下所有进程被同步切断

    # -------------------------------------------------------------------
    # 2. 安全提取参数 (此处已绝对安全，无需 getattr 的兜底值)
    # -------------------------------------------------------------------
    vocab_size = cfg.model.vocab_size
    hidden_size = cfg.model.hidden_size
    num_hidden_layers = cfg.model.num_hidden_layers
    num_attention_heads = cfg.model.num_attention_heads
    intermediate_size = cfg.model.intermediate_size
    max_position_embeddings = cfg.model.max_position_embeddings
    bert_model_type = cfg.model.bert_model_type

    # 动态构建 HuggingFace BERT Config
    
    if bert_model_type == "bert-base-uncased":
        hf_bert_config = AutoConfig.from_pretrained(bert_model_type)
        hf_bert_config.update({
            "vocab_size": VOCAB_SIZE, 
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "max_position_embeddings": max_position_embeddings,
            })
        # 实例化裸 BERT 语言模型
        bert_base_model = BertForMaskedLM(hf_bert_config)
    elif bert_model_type == "megatron-bert":
        hf_bert_config = MegatronBertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        # 实例化 MegatronBert，它天然是 Pre-LN 的！
        bert_base_model = MegatronBertForMaskedLM(hf_bert_config)
    elif bert_model_type == "deberta-v3":
        # DeBERTa V3 天然具备优秀的解耦注意力机制，且架构上更倾向于 Pre-LN
        hf_bert_config = DebertaV2Config(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            # DeBERTa 的特殊配置：开启 Positional Embedding 增强
            position_biased_input=False, 
        )
        bert_base_model = DebertaV2ForMaskedLM(hf_bert_config)

    elif bert_model_type == "roberta":
        # RoBERTa 默认是 Post-LN，若要强制 Pre-LN，通常需要修改类实现或调整配置
        hf_bert_config = RobertaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        bert_base_model = RobertaForMaskedLM(hf_bert_config)

    elif bert_model_type == "electra":
        # ELECTRA 架构适合判别式任务，同样可以通过 Config 灵活控制
        hf_bert_config = ElectraConfig(
            vocab_size=VOCAB_SIZE,
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
        )
        bert_base_model = ElectraForMaskedLM(hf_bert_config)
    elif bert_model_type == "simple":
        # 1. 创建符合 SimpleBERT 的 Config 对象
        hf_bert_config = SimpleBERTConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,             # 对应 d_model
            num_hidden_layers=num_hidden_layers, # 对应 layers
            num_attention_heads=num_attention_heads, 
            max_position_embeddings=max_position_embeddings,
            pad_token_id=0 # 假设你的 PAD 索引
        )
        
        # 2. 实例化模型 (使用重构后的类名)
        bert_base_model = SimpleBERTForMaskedLM(hf_bert_config)
    # ======= ⚙️ 完美复用已有 logger 的配置打印 (仅主节点) =======
    if rank == 0:
        if hasattr(hf_bert_config, "to_json_string"):
            config_json = hf_bert_config.to_json_string()
        elif isinstance(hf_bert_config, dict):
            config_json = json.dumps(hf_bert_config, indent=4, ensure_ascii=False)
        else:
            config_json = str(hf_bert_config)

        logger.info("\n" + "="*60 + "\n⚙️  [HF BERT CONFIG MATRIX VERIFICATION]\n" + "="*60)
        for line in config_json.splitlines():
            logger.info(f"   -> {line}")
        logger.info("="*60 + "\n")
    # ========================================================

    # -------------------------------------------------------------------
    # 3. 校验 MLM 配置并提取 Mask 概率 (修复之前的遗留 Bug)
    # -------------------------------------------------------------------
    if not hasattr(cfg, "mlm_config") or not hasattr(cfg.mlm_config, "mlm_probability"):
        error_msg = "🚨 [配置校验失败] YAML 配置缺少 [mlm_config] 模块或 [mlm_probability] 参数！"
        if rank == 0:
            logger.error(error_msg)
        raise ValueError(error_msg)
        
    mask_prob = cfg.mlm_config.mlm_probability


    # 依赖注入：将纯粹的 BERT 传入我们的 MLM Wrapper 中进行动态打 Mask
    model = PoreBERTWrapper(
        bert_model=bert_base_model,
        vocab_size=VOCAB_SIZE,
        msk_token_id = SPECIAL_TOKENS["MSK"],
        pad_token_id = SPECIAL_TOKENS["PAD"],
        mask_prob=mask_prob
    )

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

        # 步数与调度
        max_steps=cfg.max_duration,
        num_train_epochs=1,
        warmup_steps=cfg.scheduler.t_warmup,
        lr_scheduler_type=cfg.scheduler.name, 

        # 混合精度支持
        fp16=(cfg.precision == "fp16"),
        bf16=(cfg.precision == "bf16"),

        # 运行期控制
        save_steps=cfg.save_interval,
        eval_steps=cfg.eval_interval,
        eval_strategy="steps",
        # 高性能数据加载
        dataloader_num_workers=cfg.train_data.num_workers,
        dataloader_pin_memory=cfg.train_data.pin_memory,
        dataloader_drop_last=False,
        # 🌟 关键：设置预取因子 (Prefetch Factor)
        # 每个 worker 预加载的样本数，默认通常是 2。
        # 如果你的数据加载瓶颈严重，可以尝试设为 4 或更高
        dataloader_prefetch_factor=cfg.train_data.prefetch_factor,

        # 监控指标看板
        report_to="wandb" if rank == 0 else "none",
        logging_steps=10, 

        ddp_find_unused_parameters=False, # BERT 结构简单，建议设为 False 提升 DDP 速度

        # 🚨 核心修改：BERT Wrapper 自己会生成 labels 并在内部计算 Loss
        # 所以我们不要让 Trainer 从 Dataset 里找 label
        label_names=[],
        # 🚨 必须添加这一行，禁止 Trainer 删掉你 Dataset 里的数据
        remove_unused_columns=False,
        # 如果你在预训练阶段，评估时只关心 Loss（通过 Loss 算 Perplexity），根本不需要计算具体的准确率或 F1，可以直接告诉 Trainer 别传任何 Logits：
        prediction_loss_only=True,
    )

    if rank == 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        os.environ["WANDB_NAME"] = cfg.wandb.name + "_BERT"

    # ===========================================================================
    # [6] 模仿 OLMo: 显式 load_path 检查点控制
    # ===========================================================================
    resume_from_checkpoint = None
    load_path = getattr(cfg, "load_path", None)

    if load_path and str(load_path).strip():
        load_path = str(load_path).strip()
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"❌ [OLMo Guard] `load_path` was specified as '{load_path}', but it does not exist!")

        reset_optimizer = getattr(cfg, "reset_optimizer_state", False)
        if reset_optimizer:
            logger.info(f"🔄 [WEIGHT_ONLY_LOAD] Loading BERT weights from '{load_path}' but RESETTING optimizer/scheduler states.")
            # 兼容 BERT 模型加载逻辑
            from transformers import PreTrainedModel
            # 略微调整为适合 BERT Wrapper 的权重加载...
            # 这里如果涉及 state_dict，确保直接给 wrapper 或者 bert_base_model load
            resume_from_checkpoint = None
        else:
            logger.info(f"🔄 [FULL_RESUME] OLMo-style explicit resume triggered. Full pipeline recovery from: '{load_path}'")
            resume_from_checkpoint = load_path
    else:
        logger.info("🆕 [FRESH_START] `load_path` is empty or null. Launching a brand new BERT pretraining run.")

    safe_barrier()

    # ===========================================================================
    # [7] 实例化 BERT Trainer 并启动
    # ===========================================================================
    
    trainer = PoreBertTrainer(
        cfg=cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=pore_bert_collate_fn, # 🌟 将自定义的类型转换与 Masking 对齐函数挂载！
    )

    if rank == 0:
        logger.info("🚀 Starting production-ready MLM pipeline via PoreBertTrainer...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Train Nanopore Biological BERT Model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = TrainBERTConfig.from_yaml(args.config)

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
