# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig

from .utils import NUM_CLASSES, ID2BASE

def show_layer_trainable_status(model):
    """详细打印每一层每个参数的训练状态"""
    print(f"{'Parameter Name':<60} {'Shape':<20} {'Trainable':<10}")
    print("-" * 90)
    
    for name, param in model.named_parameters():
        trainable = "Yes" if param.requires_grad else "No"
        print(f"{name:<60} {str(list(param.shape)):<20} {trainable:<10}")


class LinearCRFEncoder(nn.Module):
    def __init__(
        self,
        insize: int,
        n_base: int,
        state_len: int,
        bias: bool = True,
        scale: float | None = None,
        activation: str | None = None,
        blank_score: float | None = None,
        expand_blanks: bool = True,
        permute: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if n_base <= 0:
            raise ValueError("n_base must be >= 1.")
        if state_len < 0:
            raise ValueError("state_len must be >= 0.")
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1)
        self.linear = nn.Linear(insize, size, bias=bias)
        if activation is None:
            self.activation = None
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.permute = permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.permute is not None:
            x = x.permute(*self.permute)
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            if not scores.is_contiguous():
                scores = scores.contiguous()
            bsz, t_len, n_scores = scores.shape
            expected = self.n_base ** (self.state_len + 1)
            if n_scores != expected:
                raise ValueError("CRF score dim must match no-blank size when blank_score is set.")
            scores = scores.view(bsz, t_len, n_scores // self.n_base, self.n_base)
            scores = F.pad(scores, (1, 0), value=float(self.blank_score))
            scores = scores.view(bsz, t_len, -1)
        return scores




class LinearCTCEncoder(nn.Module):
    def __init__(
        self,
        insize: int,
        num_classes: int,
        bias: bool = True,
        scale: float | None = None,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.proj = nn.Linear(insize, num_classes, bias=bias)
        if activation is None:
            self.activation = None
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = torch.nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.proj(x)
        if self.activation is not None:
            logits = self.activation(logits)
        if self.scale is not None:
            logits = logits * self.scale
        return logits

class IdentityPreHead(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.output_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BiLSTMPreHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class TransformerPreHead(nn.Module):
    def __init__(self, model_dim: int, nhead: int = 8, dim_feedforward: int | None = None) -> None:
        super().__init__()
        ff = dim_feedforward if dim_feedforward is not None else model_dim * 4
        self.layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=1)
        self.output_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv1(y).transpose(1, 2)
        y = self.norm1(y)
        y = self.activation(y)
        y = self.dropout(y)

        y = y.transpose(1, 2)
        y = self.conv2(y).transpose(1, 2)
        y = self.norm2(y)
        y = self.activation(y)
        y = self.dropout(y)
        return residual + y


class TCNPreHead(nn.Module):
    def __init__(
        self,
        model_dim: int,
        kernel_size: int = 3,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd to keep sequence length unchanged.")
        self.blocks = nn.ModuleList(
            [
                TCNBlock(
                    channels=model_dim,
                    kernel_size=kernel_size,
                    dilation=2**layer_idx,
                    dropout=dropout,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.output_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class BasecallModel(nn.Module):
    """
    input_ids: [B, T]
    output logits_btc: [B, T, C]  (给 CTC 用)
    """
    def __init__(
        self,
        model_path: str,
        num_classes: int = NUM_CLASSES,
        hidden_layer: int = -1,          # 选哪一层 hidden_states
        learnable_fuse_last_n_layers: int = 0,  # 0=关闭；>0 时对最后 N 层做 softmax 可学习加权融合
        feature_source: str = "hidden",   # hidden | embedding
        vq_device: str = "cuda",
        vq_token_batch_size: int = 100,
        freeze_backbone: bool = False,   # ✅ 新增：是否冻结基座（默认不冻结，保持你原行为）
        reset_backbone_weights: bool = False,  # ✅ 可选：重置基座权重用于消融
        unfreeze_last_n_layers: int = 0,  # 可选：仅解冻最后 N 层（其余保持冻结）
        unfreeze_layer_start: int | None = None,
        unfreeze_layer_end: int | None = None,
        head_output_activation: str | None = None,
        head_output_scale: float | None = None,
        head_crf_blank_score: float | None = None,
        head_crf_n_base: int | None = None,
        head_crf_state_len: int | None = None,
        head_crf_expand_blanks: bool = True,
        pre_head_type: str = "none",
        pre_head_transformer_nhead: int = 8,
        head_type: str = "ctc_crf",
    ):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.learnable_fuse_last_n_layers = max(0, int(learnable_fuse_last_n_layers))
        self.feature_source = feature_source
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_last_n_layers = max(0, int(unfreeze_last_n_layers))
        self.unfreeze_layer_start = unfreeze_layer_start
        self.unfreeze_layer_end = unfreeze_layer_end
        self.tokenizer = None
        self.backbone = None
        self.vq_embedding = None

        if self.feature_source not in {"hidden", "embedding", "vq_embedding"}:
            raise ValueError(f"Unsupported feature_source: {self.feature_source}. Use 'hidden', 'embedding' or 'vq_embedding'.")
        if self.feature_source != "hidden" and self.learnable_fuse_last_n_layers > 0:
            raise ValueError("learnable_fuse_last_n_layers requires feature_source='hidden'.")
        self.layer_fuse_logits: nn.Parameter | None = None
        if self.learnable_fuse_last_n_layers > 0:
            self.layer_fuse_logits = nn.Parameter(torch.zeros(self.learnable_fuse_last_n_layers))
        if self.feature_source == "vq_embedding":
            try:
                from poregpt.tokenizers import VQETokenizer
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "feature_source='vq_embedding' requires `poregpt` to be installed in the runtime environment."
                ) from exc

            vq_tokenizer = VQETokenizer(
                model_ckpt=model_path,
                device=vq_device,
                token_batch_size=vq_token_batch_size,
            )
            codebook = vq_tokenizer._get_codebook_embed()
            if hasattr(codebook, "detach"):
                codebook = codebook.detach().cpu()
            codebook = torch.as_tensor(codebook, dtype=torch.float32)
            self.vq_embedding = nn.Embedding.from_pretrained(codebook, freeze=True)
            hidden_size = int(codebook.shape[1])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            if reset_backbone_weights:
                backbone_config = AutoConfig.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )
                self.backbone = AutoModel.from_config(
                    backbone_config,
                    trust_remote_code=True,
                )
            else:
                self.backbone = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                )

            # 省显存：关闭 cache（很多 decoder-only 默认开）
            if hasattr(self.backbone.config, "use_cache"):
                self.backbone.config.use_cache = False

            # ✅ 冻结基座（核心）
            if self.freeze_backbone or self.unfreeze_last_n_layers > 0 or unfreeze_layer_start is not None or unfreeze_layer_end is not None:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                if self.freeze_backbone and self.unfreeze_last_n_layers == 0:
                    self.backbone.eval()  # 固定 dropout 等推理行为（推荐）

            # ✅ 可选：仅解冻最后 N 层
            if self.unfreeze_last_n_layers > 0 or unfreeze_layer_start is not None or unfreeze_layer_end is not None:
                layers = self._get_transformer_layers()
                n_layers = len(layers)
                if unfreeze_layer_start is not None or unfreeze_layer_end is not None:
                    start = 0 if unfreeze_layer_start is None else int(unfreeze_layer_start)
                    end = n_layers if unfreeze_layer_end is None else int(unfreeze_layer_end)
                    if start < 0:
                        start = n_layers + start
                    if end < 0:
                        end = n_layers + end
                    if not 0 <= start <= end <= n_layers:
                        raise ValueError(f"Invalid unfreeze layer range: [{start}, {end}) with {n_layers} layers.")
                    target_layers = layers[start:end]
                else:
                    n_unfreeze = min(self.unfreeze_last_n_layers, n_layers)
                    target_layers = layers[-n_unfreeze:]
                for layer in target_layers:
                    for p in layer.parameters():
                        p.requires_grad = True

            show_layer_trainable_status(self.backbone)

            hidden_size = (
                getattr(self.backbone.config, "hidden_size", None)
                or getattr(self.backbone.config, "d_model", None)
                or getattr(self.backbone.config, "n_embd", None)
            )
            if hidden_size is None:
                raise ValueError("Cannot infer hidden_size from backbone config.")

        if num_classes is None:
            num_classes = NUM_CLASSES

        self.head_type = head_type
        self.pre_head = self._build_pre_head(
            pre_head_type=pre_head_type,
            hidden_size=hidden_size,
            transformer_nhead=pre_head_transformer_nhead,
        )

        if self.head_type == "ctc_crf":
            n_base = head_crf_n_base if head_crf_n_base is not None else (len(ID2BASE) - 1)
            if head_crf_state_len is None:
                if n_base <= 1:
                    raise ValueError("Cannot infer head_crf_state_len with n_base <= 1.")
                base = num_classes / (n_base + 1)
                state_len = math.log(base, n_base) - 1
                if not math.isclose(state_len, round(state_len)):
                    raise ValueError("Unable to infer head_crf_state_len from num_classes and n_base.")
                head_crf_state_len = int(round(state_len))
            self.base_head = LinearCRFEncoder(
                insize=self.pre_head.output_dim,
                n_base=n_base,
                state_len=head_crf_state_len,
                bias=True,
                scale=head_output_scale,
                activation=head_output_activation,
                blank_score=head_crf_blank_score,
                expand_blanks=head_crf_expand_blanks,
            )
        elif self.head_type == "ctc":
            self.base_head = LinearCTCEncoder(
                insize=self.pre_head.output_dim,
                num_classes=num_classes,
                bias=True,
                scale=head_output_scale,
                activation=head_output_activation,
            )
        else:
            raise ValueError(f"Unsupported head_type: {self.head_type}")

    @staticmethod
    def _build_pre_head(
        pre_head_type: str,
        hidden_size: int,
        transformer_nhead: int,
    ) -> nn.Module:
        if pre_head_type == "none":
            return IdentityPreHead(hidden_size)
        if pre_head_type == "bilstm":
            return BiLSTMPreHead(input_dim=hidden_size, hidden_dim=128)
        if pre_head_type == "transformer":
            if hidden_size % transformer_nhead != 0:
                raise ValueError(
                    f"hidden_size={hidden_size} must be divisible by transformer nhead={transformer_nhead}."
                )
            return TransformerPreHead(model_dim=hidden_size, nhead=transformer_nhead)
        if pre_head_type == "tcn":
            return TCNPreHead(model_dim=hidden_size)
        raise ValueError(f"Unsupported pre_head_type: {pre_head_type}")

    def _get_transformer_layers(self) -> nn.ModuleList:
        candidates = (
            ("encoder", "layer"),
            ("transformer", "h"),
            ("model", "layers"),
            ("layers",),
            ("h",),
            ("blocks",),
        )
        for path in candidates:
            obj = self.backbone
            for attr in path:
                if not hasattr(obj, attr):
                    obj = None
                    break
                obj = getattr(obj, attr)
            if obj is not None and isinstance(obj, (nn.ModuleList, list, tuple)):
                return nn.ModuleList(list(obj))
        raise ValueError("Cannot locate transformer layers for partial unfreezing.")

    @staticmethod
    def _init_backbone_weights(module: nn.Module) -> None:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    def train(self, mode: bool = True):
        """✅ 防止外面 model.train() 把 backbone 切回 train（dropout 会动）"""
        super().train(mode)
        if (
            self.freeze_backbone
            and self.unfreeze_last_n_layers == 0
            and self.unfreeze_layer_start is None
            and self.unfreeze_layer_end is None
            and self.backbone is not None
        ):
            self.backbone.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.feature_source == "vq_embedding":
            hidden = self.vq_embedding(input_ids)
        elif self.feature_source == "embedding":
            embedding_layer = self.backbone.get_input_embeddings()
            if embedding_layer is None:
                raise ValueError("Backbone does not expose input embeddings via get_input_embeddings().")
            hidden = embedding_layer(input_ids)
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states  # tuple(len = n_layers + 1)

            if self.learnable_fuse_last_n_layers > 0:
                n_hidden_layers = len(hidden_states) - 1
                if n_hidden_layers < self.learnable_fuse_last_n_layers:
                    raise ValueError(
                        "learnable_fuse_last_n_layers out of range "
                        f"(requested={self.learnable_fuse_last_n_layers}, available_hidden_layers={n_hidden_layers})."
                    )
                selected = hidden_states[-self.learnable_fuse_last_n_layers:]
                stacked = torch.stack(selected, dim=0)  # [N, B, T, H]
                weights = torch.softmax(self.layer_fuse_logits, dim=0).to(stacked.dtype)  # [N]
                hidden = torch.sum(stacked * weights.view(-1, 1, 1, 1), dim=0)
            else:
                try:
                    hidden = hidden_states[self.hidden_layer]
                except IndexError:
                    raise ValueError(
                        f"hidden_layer={self.hidden_layer} out of range "
                        f"(num hidden states = {len(hidden_states)})"
                    )

        hidden = self.pre_head(hidden)
        logits_btc = self.base_head(hidden)
        return logits_btc
