import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import MaskedLMOutput

# ==============================================================================
# 1. 继承 PretrainedConfig：定义标准配置类
# ==============================================================================
class SimpleBERTConfig(PretrainedConfig):
    model_type = "pore_bert"

    def __init__(
        self,
        vocab_size=2400,
        hidden_size=512,      
        num_hidden_layers=8,  
        num_attention_heads=8,
        max_position_embeddings=512, 
        pad_token_id=0,       
        dropout=0.1,
        initializer_range=0.02, # 🌟 新增：规范化权重初始化范围，防止报错
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.initializer_range = initializer_range

# ==============================================================================
# 2. 纯粹的骨干网络 (Backbone)：不负责计算 Loss，只负责特征提取
# ==============================================================================
class SimpleMLM(nn.Module):
    def __init__(self, config: SimpleBERTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 防御性设计：如果没有传入 mask，则假设所有 token 都有效
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 🚨 核心逻辑：PyTorch 的 src_key_padding_mask 需要 BoolTensor
        # True 代表该位置是 PAD，应该被屏蔽（Mask out）。
        # 你的外部 attention_mask 是 1 为有效，0 为 PAD，所以这里取 == 0
        padding_mask = (attention_mask == 0)

        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        
        # 传入 mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)

# ==============================================================================
# 3. HF 标准的 PreTrainedModel 包装器：负责整合 Backbone、LM Head 与 Loss 计算
# ==============================================================================
class SimpleBERTForMaskedLM(PreTrainedModel):
    config_class = SimpleBERTConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. 骨干网络
        self.bert = SimpleMLM(config)

        # 2. 预测头 (LM Head)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 3. 执行 HF 内置的权重初始化
        self.post_init()

    # 🌟 新增：HF 标准接口 1 - 获取输入词向量
    def get_input_embeddings(self):
        return self.bert.token_embedding

    # 🌟 新增：HF 标准接口 2 - 设置输入词向量
    def set_input_embeddings(self, value):
        self.bert.token_embedding = value

    # 🌟 新增：HF 标准接口 3 - 获取输出预测头
    def get_output_embeddings(self):
        return self.lm_head

    # 🌟 新增：HF 标准接口 4 - 设置输出预测头
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 🌟 优化：权重初始化逻辑，使用 Config 中的 initializer_range
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        签名完全对齐 Hugging Face 标准
        """
        # 兼容老代码的入参习惯 (防止因为外部传参写成 data=... 导致报错)
        if input_ids is None and "data" in kwargs:
            input_ids = kwargs["data"]

        # 1. 获取隐藏层特征
        sequence_output = self.bert(input_ids, attention_mask)

        # 2. 通过 LM Head 得到预测 Logits
        logits = self.lm_head(sequence_output)

        # 3. 计算 Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 展平计算，确保 -100 被正确忽略
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 4. 返回标准的 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output
        )
