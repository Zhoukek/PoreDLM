
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import Trainer, AutoConfig, AutoModel
from src.config.train_config import TrainConfig
from collections import deque

logger = logging.getLogger(__name__)


# ===========================================================================
# [生产级 BERT Trainer] 融合异常探针、解壳保存与语言模型评估指标
# ===========================================================================
class PoreBertTrainer(Trainer):
    def __init__(self, cfg: TrainConfig, model, **kwargs):
        super().__init__(model=model, **kwargs)
        self.cfg = cfg

        # 动态获取异常 Loss 探针阈值（BERT 初始 MLM Loss 通常在 4.0~8.0 之间，可根据 cfg 灵活调整）
        self.anomaly_loss_threshold = getattr(cfg, "anomaly_loss_threshold", 10.0)

        # 🌟 锁定原始 dataset 句柄仅用于异常时的【零拷贝反查】
        self._raw_train_dataset = kwargs.get("train_dataset", None)

        # 🟢 动态统计探针配置
        self.loss_window_size = getattr(cfg, "loss_window_size", 100)         # 记录过去 100 步
        self.loss_std_multiplier = getattr(cfg, "loss_std_multiplier", 3.0)   # 3倍标准差触发
        self.loss_min_steps = getattr(cfg, "loss_min_steps", 15)              # 至少积累 15 步才开始动态计算
        self._loss_history = deque(maxlen=self.loss_window_size)              # 环形队列自动淘汰旧数据

    # ===========================================================================
    # 核心拦截器 & 损失计算引擎 (Compute Loss Engine)
    # ---------------------------------------------------------------------------
    # 设计目标:
    # 1. 契约剥离: 安全剥离不属于 BERT 语义前向传播的 meta 数据 (如 id)。
    # 2. 健壮提取: 完美兼容 MLM 字典输出、标准 Dataclass 结构或传统 Tuple。
    # 3. 异常监控: 延续 Z-Score 算法，在 BERT 训练出现 Loss Spike 时精准狙击。
    # 4. 防御兜底: 防止评估阶段因 DDP 梯度对齐缺陷引发进程挂起。
    # ===========================================================================
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算单步损失，并提供生产级的异常追溯能力。
        """
        # 🔍 状态嗅探: 判断当前是训练模式还是评估模式
        is_training = model.training

        # -----------------------------------------------------------------------
        # [阶段 1] 数据清洗与契约剥离 (Input Sanitization)
        # -----------------------------------------------------------------------
        # 💥 核心拦截: `id` 用于分布式流式加载器的嫌疑溯源，必须从 inputs 中 pop 出来
        chunk_ids = inputs.pop("id", None)

        # -----------------------------------------------------------------------
        # [阶段 2] 模型前向传播 (Forward Pass)
        # -----------------------------------------------------------------------
        # 此时 inputs 已净化，完全符合 BERT 模型或其 Wrapper 的 forward 签名
        outputs = model(**inputs)

        # -----------------------------------------------------------------------
        # [阶段 3] 弹性损失提取 (Robust Loss Extraction)
        # -----------------------------------------------------------------------
        loss = None

        # 策略 A: 字典类型 - 兼容 HuggingFace 标准 "loss" 或自定义 "mlm_loss"
        if isinstance(outputs, dict):
            loss = outputs.get("loss", outputs.get("mlm_loss", None))

        # 策略 B: HF 原生 Dataclass 输出 (如 MaskedLMOutput / ModelOutput)
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
                    f"🚨 [BERT Eval 异常] 模型在 Eval 模式下未返回任何有效 Loss！\n"
                    f" - 模型输出类型: {type(outputs)}\n"
                    f" - 模型输出结构: {list(outputs.keys()) if isinstance(outputs, dict) else 'Non-Dict'}\n"
                    f" 👉 请检查 model.forward() 逻辑，确保验证集前向传播依然计算并输出了 loss！"
                )
                device = next(model.parameters()).device
                loss = torch.tensor(0.0, device=device, requires_grad=False)
            else:
                # 🔴 Train 阶段缺失 Loss: 致命错误，直接中断训练
                raise ValueError("🚨 致命错误: compute_loss 在 BERT 训练模式下未能成功提取到 Loss！")

        # 确保 loss 是单一标量 (防止多卡或 batch 未做 mean 处理)
        if loss.dim() > 0:
            loss = loss.mean()

        # -----------------------------------------------------------------------
        # [阶段 5] 工业级动态异常 Loss 追溯探针 (Dynamic Anomaly Loss Probe)
        # -----------------------------------------------------------------------
        if is_training:
            current_loss_val = loss.item()
            is_anomaly = False
            trigger_reason = ""

            # 1. 静态兜底拦截
            if current_loss_val > self.anomaly_loss_threshold:
                is_anomaly = True
                trigger_reason = f"超过静态绝对阈值 ({self.anomaly_loss_threshold})"

            # 2. 动态 Z-Score 统计拦截
            elif len(self._loss_history) >= self.loss_min_steps:
                history_tensor = torch.tensor(list(self._loss_history), dtype=torch.float32)
                mean_loss = history_tensor.mean().item()
                std_loss = history_tensor.std().item() + 1e-5  # 防止 std 为 0

                dynamic_threshold = mean_loss + (self.loss_std_multiplier * std_loss)

                if current_loss_val > dynamic_threshold:
                    is_anomaly = True
                    trigger_reason = f"超过动态阈值 {dynamic_threshold:.4f} (均值: {mean_loss:.4f}, Std: {std_loss:.4f})"

            # 3. 告警与物理溯源
            if is_anomaly:
                if chunk_ids is not None:
                    try:
                        cpu_chunk_ids = chunk_ids.detach().cpu().numpy().tolist()
                        bad_chunk_ids_str = ", ".join(map(str, cpu_chunk_ids[:10]))
                        if len(cpu_chunk_ids) > 10:
                            bad_chunk_ids_str += f" ... (共 {len(cpu_chunk_ids)} 个)"

                        logger.warning(
                            f"🧨 [BERT Loss 动态狙击] 训练 Step Loss 发生突增！当前值: {current_loss_val:.4f} | "
                            f"触发原因: {trigger_reason} | "
                            f"嫌疑 Chunk_IDs: [{bad_chunk_ids_str}]"
                        )

                        # 零拷贝物理源文件定位
                        if getattr(self, "_raw_train_dataset", None) is not None:
                            real_source_file = self._raw_train_dataset.get_source_path_by_id(cpu_chunk_ids[0])
                            logger.warning(f"📦 物理源文件安全定位: {real_source_file}")

                    except Exception as e:
                        logger.debug(f"⚠️ [探针降级] 异常追溯日志打印失败，已忽略: {e}")
            else:
                # 🟢 防投毒设计：只有当前 Loss 正常时，才将其计入历史基线
                self._loss_history.append(current_loss_val)
            
            # 4. 健壮性防御：检查是否输出了无效 Loss
            if loss is None:
                raise ValueError("🚨 [Fatal] compute_loss 未能提取到有效的 mlm_loss")

            # 5. ⚠️ [核心策略实施]：确保模型已通过 Mask 排除特殊 Token
            # 此时 inputs["attention_mask"] 已经是 collate_fn 中正确生成的 (1 为参与计算)
            # 你的 Wrapper 模型内部必须确保 labels 中将 CLS/BOS/EOS 设为 -100


        # -----------------------------------------------------------------------
        # [阶段 6] 遵循 Hugging Face Trainer 契约返回结果
        # -----------------------------------------------------------------------
        return (loss, outputs) if return_outputs else loss

    # ===========================================================================
    # 轻量化评估引擎：内建语言模型专用指标计算 (如 Perplexity)
    # ===========================================================================
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        利用 Hugging Face 原生轻量化评估链路，同步核心损失并计算语言模型专用的困惑度 (PPL)
        """
        logger.info(f"⚙️ 触发纳米孔 BERT 模型纯净版掩码损失与指标评估...")

        # =======================================================================
        # 🕵️‍♂️ 【新增：抓包铁证埋点】
        # =======================================================================
        try:
            # 1. 尝试获取经 Trainer 底层处理好、即将交给验证循环的 DataLoader
            # 注：eval_dataset 如果为 None，Trainer 会自动使用初始化时传入的 eval_dataset
            test_loader = self.get_eval_dataloader(eval_dataset)

            # 2. 探查数据集包装层级
            wrapped_dataset = test_loader.dataset

            logger.warning("================= 🕵️‍♂️ HF TRAINER LENGTH INVESTIGATION =================")
            # 打印你的原生 dataset 长度
            if hasattr(wrapped_dataset, "dataset") and hasattr(wrapped_dataset.dataset, "__len__"):
                logger.warning(f"🔍 [你的原生 Dataset 长度]: {len(wrapped_dataset.dataset)}")

            # 打印被 HF 包装成 Shard 后的长度
            if hasattr(wrapped_dataset, "__len__"):
                logger.warning(f"🔍 [HF 包装后 Shard 长度]: {len(wrapped_dataset)}")

            # 打印进度条最终看见的总步数 (DataLoader 长度)
            logger.warning(f"🔍 [进度条最终画格子的总步数 (DataLoader)]: {len(test_loader)}")
            logger.warning("======================================================================")

        except Exception as e:
            logger.debug(f"⚠️ [抓包探针提示] 长度捕获跳过，原因: {e}")
        # =======================================================================
        # 1. 顺着原生 Trainer 的内建逻辑执行评估（多卡 DDP 验证 Loss 会自动聚合）
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        print(f"DEBUG: metrics from super().evaluate: {metrics}")

        # 2. 针对 BERT 模型追加计算困惑度 (Perplexity)
        if f"{metric_key_prefix}_loss" in metrics:
            raw_eval_loss = metrics[f"{metric_key_prefix}_loss"]
            try:
                # 使用 min(loss, 20) 防止初始未收敛时 exp() 导致 float 溢出报错
                metrics[f"{metric_key_prefix}_perplexity"] = math.exp(min(raw_eval_loss, 20))
            except Exception:
                metrics[f"{metric_key_prefix}_perplexity"] = float("inf")

        return metrics

    def evaluation_loop(self, dataloader, *args, **kwargs):
        """
        多卡全收集（All-Gather）指标更新复写，确保各卡数据结构绝对一致，防御 DDP 挂起
        """
        output = super().evaluation_loop(dataloader, *args, **kwargs)

        # 剥离 DDP 封装层，安全嗅探模型内建的附加指标（如准确率、掩码命中的统计等）
        model_to_call = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_to_call, "get_metrics"):
            extra_metrics = model_to_call.get_metrics()
            # 动态将底层自定义指标合并到输出字典中
            output.metrics.update({f"eval_{k}": v for k, v in extra_metrics.items()})
            
        return output


    def save_model_old(self, output_dir: str = None, _internal_call: bool = False):
        """
        🌟 增强版核心保存：带结构探针功能，先打印所有属性，再执行防御性解绑保存。
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # 1. 安全脱去 DDP 外壳
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # =======================================================================
        # 🕵️‍♂️ 【新增：模型架构与属性深度探针】
        # =======================================================================
        logger.warning("================= 🕵️‍♂️ POREBERT MODEL STRUCTURE INVESTIGATION =================")
        try:
            # A. 打印顶层对象的所有直接属性（过滤掉内置的双下划线属性）
            all_attributes = [attr for attr in dir(model_to_save) if not attr.startswith("__")]
            logger.warning(f"🔍 [当前顶层模型的直接属性列表 (dir)]:\n{all_attributes}\n")

            # B. 打印顶层模型的第一层子模块名称 (named_children)
            # 这能直接暴露出类似 self.bert = ... 或者 self.custom_net = ... 的变量名
            child_modules = [name for name, _ in model_to_save.named_children()]
            logger.warning(f"🔍 [当前顶层模型的第一层子模块名 (named_children)]:\n{child_modules}\n")

            # C. 打印 state_dict 的前 10 个 Key，直观观察权重路径
            if hasattr(model_to_save, "state_dict"):
                sd_keys = list(model_to_save.state_dict().keys())
                logger.warning(f"🔍 [模型权重 state_dict 前 10 个 Key 示例]:\n{sd_keys[:10]}")
        except Exception as probe_err:
            logger.error(f"⚠️ [结构探针执行失败]: {probe_err}")
        logger.warning("======================================================================")
        # =======================================================================

        # 2. 递归或多层嗅探可能存在的 cls.predictions 层
        target_prediction_layer = None
       # [优先选择]: 利用你已经实现的 HF 标准接口 (最稳妥，无论你将来怎么改结构，只要实现了这个接口就能找到)
        if hasattr(model_to_save, "get_output_embeddings"):
             target_prediction_layer = model_to_save.get_output_embeddings() 
        elif hasattr(model_to_save, "cls") and hasattr(model_to_save.cls, "predictions"):
            target_prediction_layer = model_to_save.cls.predictions
        elif hasattr(model_to_save, "generator_lm_head"):
            target_prediction_layer = model_to_save.generator_lm_head
        elif hasattr(model_to_save, "bert") and hasattr(model_to_save.bert, "cls") and hasattr(model_to_save.bert.cls, "predictions"):
            target_prediction_layer = model_to_save.bert.cls.predictions
        elif hasattr(model_to_save, "bert_model") and hasattr(model_to_save.bert_model, "cls") and hasattr(model_to_save.bert_model.cls, "predictions"):
            target_prediction_layer = model_to_save.bert_model.cls.predictions
        elif hasattr(model_to_save, "model") and hasattr(model_to_save.model, "cls") and hasattr(model_to_save.model.cls, "predictions"):
            target_prediction_layer = model_to_save.model.cls.predictions

        # 3. 动态应用“克隆解绑”防御
        if target_prediction_layer is not None and hasattr(target_prediction_layer, "decoder"):
            logger.info("🛡️ [PoreBertTrainer] 成功捕获到 BERT Prediction Head，启动 Safetensors 冲突防御性解绑...")
            
            decoder_weight = target_prediction_layer.decoder.weight
            decoder_bias = target_prediction_layer.decoder.bias

            # 制造独立的内存副本
            target_prediction_layer.decoder.weight = torch.nn.Parameter(decoder_weight.detach().clone())
            target_prediction_layer.decoder.bias = torch.nn.Parameter(decoder_bias.detach().clone())

            try:
                # 统一调用父类默认的保存逻辑
                super().save_model(output_dir, _internal_call)
                logger.info(f"✅ [PoreBertTrainer] 模型已成功安全保存至: {output_dir}")
            finally:
                # ⚠️ 极为重要：恢复原始的共享指针引用
                target_prediction_layer.decoder.weight = decoder_weight
                target_prediction_layer.decoder.bias = decoder_bias
                if hasattr(model_to_save, "tie_weights"):
                    model_to_save.tie_weights()
        else:
            # 如果不属于上述任何 BERT 结构，或者不含共享权重，走标准原生保存
            logger.warning("⚠️ [PoreBertTrainer] 未探测到预期的 BERT Head 结构，将执行原生保存流程。")
            super().save_model(output_dir, _internal_call)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """
        🌟 增强版核心保存：支持标准 BERT 与 ELECTRA 架构的防御性解绑保存。
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        # 1. 安全脱去 DDP 外壳
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # =======================================================================
        # 🕵️‍♂️ 【保留你原来的结构探针日志打印逻辑】
        # =======================================================================
        logger.warning("================= 🕵️‍♂️ POREBERT MODEL STRUCTURE INVESTIGATION =================")
        try:
            all_attributes = [attr for attr in dir(model_to_save) if not attr.startswith("__")]
            child_modules = [name for name, _ in model_to_save.named_children()]
            if hasattr(model_to_save, "state_dict"):
                sd_keys = list(model_to_save.state_dict().keys())
                logger.warning(f"🔍 [模型权重 state_dict 前 10 个 Key 示例]:\n{sd_keys[:10]}")
        except Exception as probe_err:
            logger.error(f"⚠️ [结构探针执行失败]: {probe_err}")
        logger.warning("======================================================================")

        # 2. 🚨 增强型多架构 Head 嗅探 (直接定位包含 weight 的目标 Linear 层)
        target_layer = None

        # [架构 A]: 标准 BERT / RoBERTa (寻找 cls.predictions.decoder)
        if hasattr(model_to_save, "cls") and hasattr(model_to_save.cls, "predictions"):
            target_layer = model_to_save.cls.predictions.decoder
        elif hasattr(model_to_save, "bert") and hasattr(model_to_save.bert, "cls") and hasattr(model_to_save.bert.cls, "predictions"):
            target_layer = model_to_save.bert.cls.predictions.decoder
        elif hasattr(model_to_save, "bert_model") and hasattr(model_to_save.bert_model, "cls") and hasattr(model_to_save.bert_model.cls, "predictions"):
            target_layer = model_to_save.bert_model.cls.predictions.decoder

        # [架构 B]: ELECTRA (根据报错信息寻找 generator_lm_head)
        elif hasattr(model_to_save, "generator_lm_head"):
            target_layer = model_to_save.generator_lm_head
        elif hasattr(model_to_save, "bert") and hasattr(model_to_save.bert, "generator_lm_head"):
            target_layer = model_to_save.bert.generator_lm_head
        elif hasattr(model_to_save, "bert_model") and hasattr(model_to_save.bert_model, "generator_lm_head"):
            target_layer = model_to_save.bert_model.generator_lm_head

        # 3. 动态应用“克隆解绑”防御
        if target_layer is not None and hasattr(target_layer, "weight"):
            logger.info("🛡️ [PoreBertTrainer] 成功捕获到 Prediction Head，启动 Safetensors 冲突防御性解绑...")

            # 提取原有的权重和偏置指针
            orig_weight = target_layer.weight
            orig_bias = getattr(target_layer, "bias", None)

            # 制造独立的内存副本
            target_layer.weight = torch.nn.Parameter(orig_weight.detach().clone())
            if orig_bias is not None:
                target_layer.bias = torch.nn.Parameter(orig_bias.detach().clone())

            try:
                # 统一调用父类默认的保存逻辑
                super().save_model(output_dir, _internal_call)
                logger.info(f"✅ [PoreBertTrainer] 模型已成功安全保存至: {output_dir}")
            finally:
                # ⚠️ 极为重要：恢复原始的共享指针引用
                target_layer.weight = orig_weight
                if orig_bias is not None:
                    target_layer.bias = orig_bias

                # 尝试触发 HuggingFace 内置的绑定逻辑以防万一
                if hasattr(model_to_save, "tie_weights"):
                    model_to_save.tie_weights()
                elif hasattr(model_to_save, "bert") and hasattr(model_to_save.bert, "tie_weights"):
                    model_to_save.bert.tie_weights()
        else:
            # 如果依然没找到，走原生保存
            logger.warning("⚠️ [PoreBertTrainer] 未探测到预期的 BERT/ELECTRA Head 结构，将执行原生保存流程。")
            super().save_model(output_dir, _internal_call)
