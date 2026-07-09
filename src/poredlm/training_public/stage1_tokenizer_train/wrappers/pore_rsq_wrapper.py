import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple
from modeling_pore_codec import PoreRSQCodec
class PoreRSQWrapper(nn.Module):
    def __init__(self, codec: PoreRSQCodec):
        super().__init__()
        self.codec = codec
    
    def forward(self, signal: torch.Tensor, **kwargs):
        # 🌟 极简自监督架构：输入 signal 直接作为重构目标，无需额外标签字段
        # 🚨 核心修复：防止 Conv1d 将 Batch Size (16) 误判为 Channel
        # HF Trainer 传进来的 signal 是 [B, 6000] 比如 [16, 6000]
        # 我们必须显式补齐通道维度，变为 [B, 1, 6000]
        # 在深度学习架构中，Wrapper（包装器）的唯一存在意义就是做“胶水层”和“适配器”。
        # 上游（Trainer） 塞过来的是标准的、通用的 Batch 数据。
        # 下游（Core Model / Codec） 期望的是特定维度（如必须有通道维）的张量。
        # Wrapper 的职责 就是在中间把两边的标准对齐。它只把数据直接透传给 self.codec(signal)，而没有在入口处对数据的 Shape 进行防御性校验和对齐（例如确保一定是 3D 的 [B, C, L]），这就是设计上的疏忽。
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        # 进入核心编码器解码器
        #recon, level_indices = self.codec(signal, forward_mode="dropout")
        recon, level_indices = self.codec(signal)

        # 计算 Loss：直接使用原始 signal 作为对比基准
        # 注意：此处要求 recon 的维度必须与补齐后的 signal 保持一致
        loss = F.mse_loss(recon, signal)

        self.last_indices = level_indices

        # 缓存用于指标计算的张量，避免每次都要重新计算
        self.last_recon = recon.detach()
        self.last_target = signal.detach() # 保留 last_target 命名以便 get_metrics 逻辑复用

        return {"loss": loss, "recon": recon} # 返回 recon 以便后续指标计算

    @torch.no_grad() # 显式标记，防止内存泄漏
    def get_metrics(self):
        metrics = {}

        # 1. 计算 Codebook Usage (码本使用率)
        # 假设 codec 中有一个 quantizer 组件
        if hasattr(self.codec, "quantizer"):
            # 获取当前 batch 的所有码本索引
            indices = self.codec.quantizer.last_indices
            num_codes = self.codec.quantizer.codebook_size

            # 计算唯一出现的索引数量
            used_codes = torch.unique(indices).numel()
            # 💡 工业优化：转为纯 float 防止多卡 DDP 计算图挂载显存引发 OOM
            metrics["codebook_usage"] = float(used_codes / num_codes)

        # 2. 计算 SNR (信噪比)
        # SNR = 10 * log10(信号功率 / 噪声功率)
        # 噪声功率 = MSE (recon - signal)
        if hasattr(self, "last_recon") and hasattr(self, "last_target"):
            signal_power = torch.mean(self.last_target ** 2)
            noise_power = F.mse_loss(self.last_recon, self.last_target)
            # 💡 修改点 A：将计算出来的真实信噪比，直接对齐绑定到你外面的看板键名 "snr_loss" 上
            metrics["snr_loss"] = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
            
        """架构相关的指标收集器"""
        # 💡 修改点 B：使用 setdefault 安全占位。如果上面没有算出 snr_loss，才给它默认值 0.0；
        # 如果上面算出来了，绝对不会覆盖你辛辛苦苦计算的结果！
        metrics.setdefault("snr_loss", 0.0)
        metrics.setdefault("codebook_max_entropy", 0.0)
        
        # 如果 codec 中有更深层的状态（如码本利用率），可以在此提取

        return metrics
