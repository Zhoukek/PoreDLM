import math
import logging
from transformers import TrainerCallback, TrainerState, TrainerArguments, TrainerControl

logger = logging.getLogger(__name__)

class PoreWandBCallback(TrainerCallback):
    """
    纳米孔模型专属的 WandB 监控插件。
    完全不污染训练主流程，由 Trainer 在特定时间节点自动调用。
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def on_log(self, args: TrainerArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        每当控制台/日志准备打印（logging_steps）时，Trainer 会把当前计算出的指标塞进 logs 字典传给你。
        我们在这个节点做复杂的数学换算。
        """
        # 确保只有分布式训练的主进程才处理日志，且 logs 不为空
        if not state.is_world_process_zero or logs is None:
            return

        # 1. 从 Trainer 算好的原生 outputs 中提取关键 Loss (需要在模型的 forward 或 compute_loss 中返回)
        # 注意：HF 默认会把 loss 存在 "loss" 里，如果你在模型里额外返回了字典，也会被塞进 logs
        g_recon = logs.get("loss_recon", 0.0)
        g_comit = logs.get("loss_comit", 0.0)
        epsilon = 1e-8

        # 2. 动态计算你特有的纳米孔业务指标，直接塞回 logs 字典
        # 🌟 重点：塞进 logs 字典后的参数，会被 Trainer 内部的 WandbCallback 自动捕获并上传！
        if g_recon > 0:
            logs["train/recon_loss_log10"] = math.log10(g_recon + epsilon)
            logs["train/recon_per_comit"] = g_recon / (g_comit + epsilon)
        if g_comit > 0:
            logs["train/comit_loss_log10"] = math.log10(g_comit + epsilon)

        # 3. 记录一些从 cfg 或状态中获取的特殊超参
        logs["comit/dynamic_commit_weight"] = getattr(self.cfg, "commitment_weight", 0.0)
        logs["train/global_step"] = state.global_step
        logs["train/epoch"] = state.epoch

    def on_evaluate(self, args: TrainerArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """
        每当评估（Evaluation）结束时自动调用。
        你可以在这里动态解析任意层数的 Codebook 指标。
        """
        if not state.is_world_process_zero or metrics is None:
            return

        # 假设你在模型的 evaluate 或 get_eval_dataloader 流程中，把多层码本指标塞进了 metrics
        # 我们可以用一个优雅的循环自动解析 codebook0, codebook1, codebook2...
        # 这样以后你哪怕从 2 层 RSQ 变成 4 层，这里的一行代码都不需要动！
        for key in list(metrics.keys()):
            if "codebook" in key:
                # 规范化命名，直接让 HF 原生的 WandB 顺手带上天
                metrics[f"eval_{key}"] = metrics.pop(key)
