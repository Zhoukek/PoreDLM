import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, AutoConfig, AutoModel
from modeling_pore_codec import PoreRSQCodec, PoreRSQCodecConfig
from config.train_config import TrainConfig
from collections import deque
logger = logging.getLogger(__name__)


# ===========================================================================
# [生产级 Trainer] 融合异常探针与自动包装逻辑
# ===========================================================================
class PoreTrainer(Trainer):
    def __init__(self, cfg: TrainConfig, model, **kwargs):

        super().__init__(model=model, **kwargs)
        self.cfg = cfg

        # 动态获取异常 Loss 探针阈值
        self.anomaly_loss_threshold = getattr(cfg, "anomaly_loss_threshold", 1.0)

        # 🌟 锁定原始 dataset 句柄仅用于异常时的【零拷贝反查】
        self._raw_train_dataset = kwargs.get("train_dataset", None)

        # 🟢 动态统计探针配置
        self.loss_window_size = getattr(cfg, "loss_window_size", 10)         # 记录过去 100 步
        self.loss_std_multiplier = getattr(cfg, "loss_std_multiplier", 3.0)   # 5倍标准差触发
        self.loss_min_steps = getattr(cfg, "loss_min_steps", 10)              # 至少积累 10 步才开始动态计算
        self._loss_history = deque(maxlen=self.loss_window_size)              # 环形队列自动淘汰旧数据

    # ===========================================================================
    # 核心拦截器 & 损失计算引擎 (Compute Loss Engine)
    # ---------------------------------------------------------------------------
    # 设计目标:
    # 1. 契约剥离: 将不属于模型 forward 签名的 meta 数据 (如 id) 提前安全剥离。
    # 2. 健壮提取: 兼容 Dict, Tuple, Dataclass 等多种模型输出格式，确保 Train/Eval 均能拿到 Loss。
    # 3. 异常监控: 在训练阶段提供零拷贝的 Loss 飙升（Loss Spike）追溯探针。
    # 4. 防御兜底: 防止由于模型 Eval 逻辑缺陷导致 Trainer 崩溃或静默错误。
    # ===========================================================================
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算单步损失，并提供生产级的异常追溯能力。
        
        Args:
            model (nn.Module): 当前正在训练或评估的模型对象 (可能是 DDP wrap 后的).
            inputs (Dict[str, Tensor]): 当前 Batch 的输入数据字典.
            return_outputs (bool): 是否需要同时返回 outputs (HuggingFace 内部机制需要).
            kwargs: 接收来自外部调用者的其他未定参数，保证签名兼容性.
            
        Returns:
            Union[Tensor, Tuple[Tensor, Any]]: 单个 Loss 张量，或包含 (loss, outputs) 的元组.
        """
        # 🔍 状态嗅探: 判断当前是训练模式还是评估模式 (非常重要，用于防御逻辑)
        is_training = model.training

        # -----------------------------------------------------------------------
        # [阶段 1] 数据清洗与契约剥离 (Input Sanitization)
        # -----------------------------------------------------------------------
        # 💥 核心拦截: `id` 用于追踪，但不属于模型的 forward() 参数。
        # 使用 pop 而不是 del，保证键不存在时不会 KeyError
        chunk_ids = inputs.pop("id", None)

        # -----------------------------------------------------------------------
        # [阶段 2] 模型前向传播 (Forward Pass)
        # -----------------------------------------------------------------------
        # 此时 inputs 已被净化，完全符合模型 forward 签名契约
        outputs = model(**inputs)

        # -----------------------------------------------------------------------
        # [阶段 3] 弹性损失提取 (Robust Loss Extraction)
        # -----------------------------------------------------------------------
        loss = None
        
        # 策略 A: 字典类型 (推荐标准) - 兼容标准 "loss" 或业务自定义 "recon_loss"
        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("recon_loss", None))
        
        # 策略 B: HF 原生 Dataclass 输出 (如 ModelOutput)
        elif hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
            
        # 策略 C: 元组或列表类型 (传统的 PyTorch 模型输出)
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            loss = outputs[0]

        # -----------------------------------------------------------------------
        # [阶段 4] 防御性闭环与状态断言 (Defensive Fallback & Logging)
        # -----------------------------------------------------------------------
        if loss is None:
            if not is_training:
                # 🔴 Eval 阶段缺失 Loss: 给出详尽的排查信息，并返回 0 梯度张量防止 DDP 崩溃
                logger.error(
                    f"🚨 [Eval 阶段异常] 模型在 Eval 模式下未返回 Loss！\n"
                    f" - 模型输出类型: {type(outputs)}\n"
                    f" - 模型输出结构: {list(outputs.keys()) if isinstance(outputs, dict) else 'Non-Dict'}\n"
                    f" 👉 请检查 model.forward() 逻辑，确保 'if not self.training' 分支下依然计算并返回了 loss！"
                )
                # 动态获取所在设备，构造一个占位的标量 Tensor
                device = next(model.parameters()).device
                loss = torch.tensor(0.0, device=device, requires_grad=False)
            else:
                # 🔴 Train 阶段缺失 Loss: 致命错误，无法反向传播，直接抛出异常阻断训练
                raise ValueError("🚨 致命错误: compute_loss 在训练模式下未能从模型输出中提取到 Loss！")

        # 确保 loss 是单一标量 (防止某些模型返回了未经 mean() 处理的 [B] shape 损失)
        if loss.dim() > 0:
            loss = loss.mean()

        # 探针仅在训练阶段激活 (Eval 阶段 Loss 大通常是因为没训好，不需要物理级告警)
        # -----------------------------------------------------------------------
        # [阶段 5] 工业级动态异常 Loss 追溯探针 (Dynamic Anomaly Loss Probe)
        # -----------------------------------------------------------------------
        if is_training:
            current_loss_val = loss.item()
            is_anomaly = False
            
            # 1. 静态兜底拦截 (处理冷启动阶段的直接爆炸)
            if current_loss_val > self.anomaly_loss_threshold:
                is_anomaly = True
                trigger_reason = f"超过静态绝对阈值 ({self.anomaly_loss_threshold})"

            # 2. 动态统计拦截 (基于历史 Z-Score)
            elif len(self._loss_history) >= self.loss_min_steps:
                # 转为 Tensor 以利用 PyTorch 底层 C++ 极速计算统计量
                history_tensor = torch.tensor(list(self._loss_history), dtype=torch.float32)
                mean_loss = history_tensor.mean().item()
                # 加上 1e-5 防止 loss 完全收敛导致 std 为 0 时的除零或误触
                std_loss = history_tensor.std().item() + 1e-5 
                
                dynamic_threshold = mean_loss + (self.loss_std_multiplier * std_loss)
                
                if current_loss_val > dynamic_threshold:
                    is_anomaly = True
                    trigger_reason = f"超过动态阈值 {dynamic_threshold:.4f} (均值: {mean_loss:.4f}, Std: {std_loss:.4f})"

            # 3. 告警与溯源逻辑
            if is_anomaly:
                if chunk_ids is not None:
                    try:
                        cpu_chunk_ids = chunk_ids.detach().cpu().numpy().tolist()
                        # 防止 batch 极大导致刷屏，截取前 10 个嫌疑 ID
                        bad_chunk_ids_str = ", ".join(map(str, cpu_chunk_ids[:10]))
                        if len(cpu_chunk_ids) > 10:
                            bad_chunk_ids_str += f" ... (共 {len(cpu_chunk_ids)} 个)"

                        logger.warning(
                            f"🧨 [Loss 动态阻击] 训练 Step Loss 突增至: {current_loss_val:.4f} | "
                            f"触发原因: {trigger_reason} | "
                            f"嫌疑 Chunk_IDs: [{bad_chunk_ids_str}]"
                        )

                        # 利用常驻内存的 dataset 句柄进行零拷贝反查 (仅查询第一个作为代表)
                        if getattr(self, "_raw_train_dataset", None) is not None:
                            real_source_file = self._raw_train_dataset.get_source_path_by_id(cpu_chunk_ids[0])
                            logger.warning(f"📦 物理源文件定位: {real_source_file}")
                            
                    except Exception as e:
                        logger.debug(f"⚠️ [探针降级] 异常追溯日志打印失败，已忽略: {e}")
            else:
                # 🟢 防投毒设计：只有当前 Loss 正常时，才加入历史队列更新基线。
                # 否则巨大的尖峰会抬高均值和标准差，导致后续的异常检测失效。
                self._loss_history.append(current_loss_val)

        # -----------------------------------------------------------------------
        # [阶段 6] 遵循 Hugging Face Trainer 契约返回结果
        # -----------------------------------------------------------------------
        return (loss, outputs) if return_outputs else loss

    # ===========================================================================
    # 精简版评估引擎：直接通过基类进行标准的验证集 Loss 收集
    # 在 trainer.py 的 train() 方法中，evaluation 并不是随机触发的，而是通过 args（TrainingArguments）中定义的策略进行调度。
    # args.eval_steps: 如果策略是 "steps"，则每隔多少个 global_step 执行一次。
    # args.evaluation_strategy: 可以设置为 "no", "steps", 或 "epoch"。
    # 回调机制: Trainer 使用 TrainerControl 对象来决定是否在特定时刻调用 self.evaluate()。
    # 当你调用 trainer.evaluate() 或者训练循环触发评估时，实际上运行的是以下核心流程：
    # 1. 准备阶段:
    #       将模型设置为评估模式 (model.eval())。
    #       关闭梯度计算 (torch.no_grad())。
    #       准备 eval_dataloader（通常从 get_eval_dataloader() 获取）。
    # 2. 前向传播 (Forward Pass):
    #       遍历数据加载器，将数据送入模型。
    #       计算损失函数或其他指定的 metrics.
    # 3. 计算指标 (Compute Metrics):
    #       如果定义了 compute_metrics 参数，Trainer 会收集模型的输出（logits/labels），并将其传入你自定义的评估函数。

    # 4. 结果返回:
    #       返回一个包含损失和各项指标的字典（如你日志中看到的 eval_snr_loss, eval_runtime 等）。
    # ===========================================================================
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        利用 Hugging Face 原生轻量化评估链路，仅提取和同步核心的重建损失
        """
        logger.info(f"⚙️ 触发纳米孔 Tokenizer 纯净版重建损失评估...")

        # 1. 顺着原生 Trainer 的内建逻辑执行评估（自动处理多卡 DDP 的验证 Loss 聚合）
        # 原生 evaluation_loop 会自动调用 compute_loss 并对 eval_loss 求均值
        # super().evaluate: （即 transformers.Trainer.evaluate）
        # 此时程序跳出你的 trainer.py，进入 transformers 库源码中。
        # 原生 Trainer 开始调用 self.evaluation_loop。
        # 原生 Trainer 内部又会调用 self.compute_loss。
        # 关键点：由于 Python 的动态绑定（Dynamic Binding），当原生 Trainer 内部调用 self.compute_loss 时，它实际上会调用你在 PoreTrainer 中重写的 compute_loss。
        # 返回结果: super().evaluate 将计算好的 metrics 字典返回给你的 PoreTrainer.evaluate。
        # 你的后续逻辑: 你拿到这个字典，对其进行 recon_loss 的映射和 snr_loss 的补全，最后返回给训练框架。
        # 这一步其实已经拿到了真实的 eval_loss
        # # 🔴 关键点：当你调用 super().evaluate() 时，Hugging Face 内部已经执行了自我日志上报！
        # 也就是在这行代码执行的底层，它已经调用了 self.log(metrics) 把数据发给 WandB 了。
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        print(f"DEBUG: metrics from super().evaluate: {metrics}")

        # 2. 为确保完美兼容你外面的 WandB 看板和下游打点逻辑，显式对齐键名
        # 把原生的 eval_loss 直接映射为你需要的 eval_recon_loss
        # 你在这里对 metrics 做的任何追加、修改（比如加上 eval_recon_loss，或者重置 snr_loss 为 0.0）
        # WandB 都**完全看不到**，因为负责上传的列车早就开走了！
        # 
        if f"{metric_key_prefix}_loss" in metrics:
            raw_eval_loss = metrics[f"{metric_key_prefix}_loss"]
            metrics[f"{metric_key_prefix}_recon_loss"] = raw_eval_loss

            # 同时也保留一个占位符以对齐老代码行为（防止其他回调因找不到字段而溃散）
            metrics[f"{metric_key_prefix}_snr_loss"] = 0.0
            metrics[f"{metric_key_prefix}_codebook_max_entropy"] = 0.0

        # 这里 return 的字典，只是为了给最后的主进程终端打印用的。
        return metrics
    
    def evaluation_loop(self, dataloader, *args, **kwargs):
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        
        # 错误做法
        # 只在 rank 0 进行指标收集和更新，避免重复计算或冲突
        # if self.is_world_process_zero():
        # Hugging Face Trainer 的多卡评估机制采用的是 分布式全收集（All-Gather） 策略：
        # 分工计算: 4张显卡（Rank 0, 1, 2, 3）各自平分验证集的数据，并各自在内部跑完前向传播。
        # 局部统计: 每张卡跑完后，都会生成一个属于自己这部分数据的 EvalLoopOutput 对象，里面包含各自卡的 metrics 字典。
        # 全局聚合 (Reduce): Trainer 在 evaluation_loop 的最后阶段（也就是 return output 之后，在 evaluate 函数内部），会调用分布式通信命令（如 all_gather），把所有卡的 metrics 字典收集起来做均值或汇总
        # 1. 执行原生评估循环
        # 2. ⚠️ 移除 self.is_world_process_zero() 限制！
        # 让每张卡都算出自己那部分数据的指标，确保所有卡的 output.metrics 字典结构完全一致
        model_to_call = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_to_call, "get_metrics"):
            extra_metrics = model_to_call.get_metrics()
            # 将额外指标更新到输出字典中
            output.metrics.update({f"eval_{k}": v for k, v in extra_metrics.items()})
        return output


    def save_model(self, output_dir=None, _internal_call=False):
        """🌟 核心脱壳保存：确保只保存内部的物理模型，剥离训练层"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if hasattr(self.model, "codec"):
            self.model.codec.save_pretrained(output_dir)
            logger.info(f"✅ [脱壳保存] 物理模型已导出至: {output_dir}")
        else:
            super().save_model(output_dir, _internal_call)
