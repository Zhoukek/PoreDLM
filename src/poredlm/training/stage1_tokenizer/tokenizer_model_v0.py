import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize

from typing import Tuple, Dict
from bonito.util import load_model 

from models import BERTEncoder
from poredlm.models import SignalCNN

# 强制使用标准实现
torch.backends.cuda.enable_flash_sdp(False)  # 禁用 flash SDP
torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用 memory efficient
torch.backends.cuda.enable_math_sdp(True)  # 使用数学实现


class Nanopore_Tokenizer_Model_V0(nn.Module):
    """
    Nanopore VQ Tokenizer for Direct RNA Sequencing (130 bps, 4 kHz)

    支持多种 CNN 架构配置，通过 `cnn_type` 切换：
        - cnn_type=0: 大容量非严格对称模型（默认）
        - cnn_type=1: 小容量严格对称模型（通道数 1→16→32→64）

    设计目标通用：
        - 感受野 ≈ 33 采样点（≈1 个 RNA 碱基）
        - 总下采样率 = 5×（每碱基 ≈6 个 tokens）
        - 输出 codebook_dim 维 latent，直接用于 VQ
        - Decoder 在 cnn_type=1 时严格对称于 encoder

    适用于：VQ tokenizer + LLM basecalling pipeline
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_decay: float = 0.99,
        codebook_emadc: int = 2,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        cnn_type: int = 0,
        learnable_codebook: bool= True,
        init_codebook_path: str = None,
        freeze_cnn: bool = False,
        cnn_checkpoint_path: str = None,
    ):
        super().__init__()

        self.cnn_model = SignalCNN(cnn_type=cnn_type)
        
        d_model = self.cnn_model.out_channels *1  # 自动设置为CNN输出维度
        
        codebook_dim = d_model

        # 设置 codebook_dim 根据 cnn_type
        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.codebook_size = codebook_size
        self.cnn_stride = self.cnn_model.stride
        self.RF = self.cnn_model.RF


        print(f"codebook_dim:{codebook_dim}")

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if learnable_codebook == True:
            ema_update = False
        else:
            ema_update = True

        self.vq = VectorQuantize(
            dim=d_model,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=codebook_decay,
            threshold_ema_dead_code=codebook_emadc,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_max_codes=256,
            orthogonal_reg_active_codes_only=True,
            learnable_codebook=learnable_codebook,
            ema_update = ema_update,
        )
        
        # 如果有初始codebook路径，加载它
        if init_codebook_path:
            self._load_init_codebook(init_codebook_path)
        # 如果有CNN检查点路径，加载权重
        if cnn_checkpoint_path:
            self._load_cnn_weights(cnn_checkpoint_path, freeze_cnn)
 

        if rank == 0:
            self._print_vq_config()

    def _load_cnn_weights(self, cnn_checkpoint_path, freeze_cnn=False):
        """从检查点加载CNN权重"""
        try:
            import os
            import torch

            if not os.path.isfile(cnn_checkpoint_path):
                print(f"⚠️ CNN checkpoint文件不存在: {cnn_checkpoint_path}")
                return

            print(f"📥 从 {cnn_checkpoint_path} 加载CNN权重")

            # 加载检查点
            cnn_ckpt = torch.load(cnn_checkpoint_path, map_location='cpu',weights_only=False)
            cnn_state_dict = cnn_ckpt.get('model_state_dict', cnn_ckpt)
            print("预训练模型权重键 (前几个):", list(cnn_state_dict.keys())[:5])
            print("当前模型权重键 (前几个):", list(self.state_dict().keys())[:5])
            # 预训练模型权重键 (前几个): ['encoder.0.mean_conv.weight', 'encoder.0.std_conv.weight', 'encoder.1.weight', 'encoder.1.bias', 'encoder.1.running_mean']
            # 当前模型权重键 (前几个): ['cnn_model.encoder.0.mean_conv.weight', 'cnn_model.encoder.0.std_conv.weight', 'cnn_model.encoder.1.weight', 'cnn_model.encoder.1.bias', 'cnn_model.encoder.1.running_mean']
            # --- 添加以下逻辑 ---
            # 假设当前模型的 encoder 部分是通过 'cnn_model.encoder' 这个属性访问的
            # 我们需要将预训练权重的 'encoder.xxxx' 映射到 'cnn_model.encoder.xxxx'
            mapped_cnn_state_dict = {}
            for k, v in cnn_state_dict.items():
                if k.startswith('encoder.'): # 如果原始键是以 'encoder.' 开头
                    new_k = 'cnn_model.' + k # 将其映射为 'cnn_model.encoder.xxxx'
                    mapped_cnn_state_dict[new_k] = v
                else:
                    # 如果不是以 'encoder.' 开头（例如 decoder 或其他部分），可以选择跳过或也进行相应映射
                    pass # 或者继续处理其他部分，如果需要的话
                        # 只加载encoder和decoder的权重
                        # encoder_decoder_keys = [k for k in cnn_state_dict.keys() if k.startswith(('encoder.', 'decoder.'))]
             # 现在使用映射后的字典
            cnn_state_dict = mapped_cnn_state_dict

            # 原来的筛选逻辑现在应该能找到匹配项了
            # 注意这里也改为 'cnn_model.encoder.'
            encoder_decoder_keys = [k for k in cnn_state_dict.keys() if k.startswith(('cnn_model.encoder.'))]
            if not encoder_decoder_keys:
                print(f"⚠️ 在checkpoint中未找到encoder/decoder权重")
                return
           # --- 添加结束 ---

            # 获取当前模型状态
            model_state = self.state_dict()
            loaded_keys = []

            for key in encoder_decoder_keys:
                if key in model_state and cnn_state_dict[key].shape == model_state[key].shape:
                    print(f"加载参数:{key}")
                    model_state[key] = cnn_state_dict[key]
                    loaded_keys.append(key)

            # 加载权重
            self.load_state_dict(model_state, strict=False)
            #print(f"✅ 加载了 {len(loaded_keys)} 个encoder/decoder参数")
            print(f"✅ 加载了 {len(loaded_keys)} 个encoder参数")

            # 冻结参数（如果需要）
            freeze_cnt = 0
            if freeze_cnn:
                #print("🔒 冻结encoder和decoder参数")
                print("🔒 冻结encoder参数")
                for name, param in self.named_parameters():
                    #if name.startswith(('encoder.', 'decoder.')):
                    #if name.startswith(('encoder.')):
                    if name.startswith(('cnn_model.encoder.')):      # <- 修改为新的前缀
                        freeze_cnt +=1
                        param.requires_grad = False
                        print(f"冻结参数:{name}")
            print(f"✅ 冻结了 {freeze_cnt} 个encoder参数")
        except Exception as e:
            print(f"❌ 加载CNN权重失败: {e}")


    # 在 vq_model.py 中修改 _load_init_codebook 方法
    def _load_init_codebook(self, init_codebook_path):
        """从numpy文件加载初始codebook - 修复内存布局问题"""
        try:
            import numpy as np
            import os
            
            if not os.path.isfile(init_codebook_path):
                print(f"⚠️ Codebook文件不存在: {init_codebook_path}")
                return
            
            # 直接加载numpy文件
            init_codebook = np.load(init_codebook_path)
            print(f"📥 加载codebook: {init_codebook.shape}")
            
            # 如果形状是2D，添加一个维度变成3D
            if len(init_codebook.shape) == 2:
                init_codebook = init_codebook[np.newaxis, :, :]
                print(f"  -> 调整为3D: {init_codebook.shape}")
            
            # 转换为tensor - 使用与模型参数相同的设备
            device = self.vq._codebook.embed.device
            init_codebook_tensor = torch.from_numpy(init_codebook).float().to(device)
            
            # 关键修复：确保内存布局一致
            # 使用 contiguous() 确保内存连续
            init_codebook_tensor = init_codebook_tensor.contiguous()
            
            # 获取原始参数的引用
            embed_param = self.vq._codebook.embed
            
            # 如果是Parameter，直接修改data
            if isinstance(embed_param, nn.Parameter):
                with torch.no_grad():
                    # 确保目标也是连续的
                    embed_param.data = embed_param.data.contiguous()
                    # 复制数据
                    embed_param.data.copy_(init_codebook_tensor)
            else:
                # 如果是buffer，直接赋值但保持内存布局
                self.vq._codebook.embed = init_codebook_tensor.contiguous()
            
            print(f"✅ Codebook初始化成功")
            print(f"   最终形状: {self.vq._codebook.embed.shape}")
            print(f"   内存连续: {self.vq._codebook.embed.is_contiguous()}")
            
        except Exception as e:
            print(f"❌ 加载初始codebook失败: {e}")
            import traceback
            traceback.print_exc()


    def _print_vq_config(self) -> None:
        """打印 VQ 配置信息（仅 rank 0）"""
        print("Intialized VectorQuantize with the following hyperparameters:")
        print(f"  codebook_size: {self.codebook_size}")
        print(f"  kmeans_init: True")
        print(f"  kmeans_iters: 10")
        print(f"  decay: 0.99")
        print(f"  threshold_ema_dead_code: 2")
        #print(f"  commitment_weight: {self.vq.commitment_weight}")
        #print(f"  codebook_diversity_loss_weight: {self.vq.codebook_diversity_loss_weight}")
        #print(f"  orthogonal_reg_weight: {self.vq.orthogonal_reg_weight}")
        #print(f"  orthogonal_reg_max_codes: 256")
        #print(f"  orthogonal_reg_active_codes_only: True")
        print(f"  cnn_type: {self.cnn_type}")
        print("-" * 60)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x (torch.Tensor): 输入信号，形状 [B, 1, T]
                (例如: B=4, T=2560 -> )

        Returns:
            recon (torch.Tensor): 重建信号，[B, 1, T]
            indices (torch.Tensor): VQ 离散 token，[B, N] (N = T // 5)
            loss (torch.Tensor): VQ 总损失（标量）
            loss_breakdown (dict): 损失分项（commitment, diversity, ortho...）
        """

        z_cnn = self.cnn_model.encode(x)

        z_permuted = z_cnn.permute(0, 2, 1)

        z_quantized_permuted, indices, loss, loss_breakdown = self.vq(
            z_permuted, # 输入连续特征
            return_loss_breakdown=True # 返回详细的损失分项
        )

        z_quantized = z_quantized_permuted.permute(0, 2, 1)

        recon = self.cnn_model.decode(z_quantized)

        target_len = x.shape[-1]  # 输入信号的原始长度 (2560)
        current_len = recon.shape[-1] # 重构信号的当前长度

        if current_len > target_len:
            # 如果重构信号过长（通常由 Padding 引起），进行裁剪
            recon = recon[..., :target_len]
        elif current_len < target_len:
            # 如果重构信号过短，进行填充 (Pad)
            # F.pad 的参数是 (左填充, 右填充)，这里只在时间轴末尾填充
            recon = F.pad(recon, (0, target_len - current_len))

        return recon, indices, loss, loss_breakdown

