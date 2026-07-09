import os
from typing import List, Optional
from dataclasses import dataclass, field
from omegaconf import OmegaConf

# ===========================================================================
# 🚀 1. 模型特化分支配置类 (Model Spec Branches)
# ===========================================================================

@dataclass
class RSQConfig:
    cnn_type: int = 12
    levels: List[int] = field(default_factory=lambda: [5, 5, 5, 5])
    num_quantizers: int = 2

@dataclass
class ModelConfig:
    model_type: str = "rsq"
    # 将分支特化参数聚合进来
    rsq: RSQConfig = field(default_factory=RSQConfig)
    
    # 通用量化底座参数
    codebook_size: int = 625
    codebook_dim: int = 4
    codebook_nqtz: int = 2

# ===========================================================================
# 📦 2. 数据管线特化配置类 (Data Pipeline Specs)
# ===========================================================================

@dataclass
class TrainDataConfig:
    paths: List[str] = field(default_factory=list)
    chunk_size: int = 6000
    memmap_dtype: str = "float32"
    buffer_size: int = 20000
    num_workers: int = 8
    drop_last: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 8
    persistent_workers: bool = True

@dataclass
class EvalDataConfig:
    paths: List[str] = field(default_factory=list)
    chunk_size: int = 6000
    val_ratio: float = 0.1
    memmap_dtype: str = "float32"
    buffer_size: int = 0
    num_workers: int = 2
    drop_last: bool = False
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False

# ===========================================================================
# 📊 3. 辅助组件配置类 (WandB, Loss, Optimizer, Scheduler)
# ===========================================================================

@dataclass
class WandbConfig:
    entity: str = "YourWandbEntity"
    project: str = "nanopore_vq"
    name: str = ""

@dataclass
class LossWeightsConfig:
    commitment_weight: float = 0.25
    commitment_weight_lr: float = 0.01
    use_dynamic_commitment_weight: bool = True
    commitment_weight_freeze_steps: int = 20000
    commitment_weight_rpc: int = 1
    codebook_diversity_loss_weight: float = 0.0
    orthogonal_reg_weight: float = 0.0
    update_loss_weight_every: int = 10

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    eps: float = 1.0e-8

@dataclass
class SchedulerConfig:
    name: str = "cosine_with_warmup"
    t_warmup: int = 100
    warmup_start_factor: float = 1.0e-5
    warmup_end_factor: float = 1.0
    main_scheduler_end_factor: float = 1.0e-5

# ===========================================================================
# 🌍 4. 顶层全局主配置类 (Master Training Config)
# ===========================================================================

@dataclass
class TrainConfig:
    # 全局控制
    run_name: str = "PoreCodec_V12_Multitask_Run"
    seed: int = 6198
    dry_run: bool = False

    # 子模块实例化
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train_data: TrainDataConfig = field(default_factory=TrainDataConfig)
    eval_data: EvalDataConfig = field(default_factory=EvalDataConfig)

    # 分布式 Batch 与流控矩阵
    save_folder: str = "save/porecodec_output"
    save_overwrite: bool = True
    save_interval: int = 10
    eval_interval: int = 10
    max_duration: int = 100000
    
    global_train_batch_size: int = 256
    device_train_microbatch_size: int = 16
    device_eval_batch_size: int = 16  # 后续解析时会自动承接引用

    precision: str = "amp_bf16"
    max_grad_norm: float = 1.0
    load_path: Optional[str] = None
    reset_optimizer_state: bool = False

    # 🌟 核心工程方法：从 YAML 文件安全加载并实例化强类型 Dataclass
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"❌ 未找到配置文件: {yaml_path}")
        
        # 1. 使用 OmegaConf 加载 YAML 并自动解析里面的 ${run_name} 和 ${device_train_microbatch_size} 动态引用
        loaded_cfg = OmegaConf.load(yaml_path)
        
        # 2. 将基础设施级基类作为 Schema 结构
        schema = OmegaConf.structured(cls)
        
        # 3. 强行融合并校验（如果 YAML 填错了字段名，这里会直接抛出编译期错误，严防训练在线崩溃）
        merged_cfg = OmegaConf.merge(schema, loaded_cfg)
        
        # 4. 导出为原生的强类型 Dataclass 对象
        return OmegaConf.to_object(merged_cfg)
