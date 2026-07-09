import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from vector_quantize_pytorch import VectorQuantize
from typing import Literal, Tuple

MODEL_TYPE = "pore_codec_vq"

class SignalCNN(nn.Module):
    """Nanopore 信号重建用纯卷积自编码器（无 VQ）。"""

    def __init__(self, cnn_type: Literal[0, 1] = 1) -> None:
        super().__init__()

        if cnn_type not in (0, 1):
            raise ValueError(f"`cnn_type` must be 0 or 1, got {cnn_type}.")

        self.cnn_type: int = cnn_type
        if cnn_type == 0:
            self._build_cnn_type0()
            self.out_channels = 768
            self.stride = 5
            self.receptive_field = 27
            self.RF = 27
        elif cnn_type == 1:
            pass
    
    def _build_cnn_type0(self) -> None:
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.SiLU(),

            nn.Conv1d(4, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.SiLU(),

            nn.Conv1d(16, 768, kernel_size=19, stride=5, padding=9, bias=False),
        )
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 768 → 16
            nn.ConvTranspose1d(
                in_channels=768,
                out_channels=16,
                kernel_size=19,
                stride=5,
                padding=9,
                output_padding=1,
                bias=False,
            ),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 16 → 4
            nn.Conv1d(16, 4, kernel_size=5, padding=2, bias=False),
            nn.SiLU(),
            
            # Inverse of encoder Layer 1: 4 → 1
            nn.Conv1d(4, 1, kernel_size=5, padding=2, bias=True)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signal to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to signal."""
        return self.decoder(z)


class PoreVQCodecConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_decay: float = 0.99,
        codebook_emadc: int = 2,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        cnn_type: int = 0,
        learnable_codebook: bool = True,
        init_codebook_path: str | None = None,
        freeze_cnn: bool = False,
        cnn_checkpoint_path: str | None = None,
        teacher_model_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.codebook_decay = codebook_decay
        self.codebook_emadc = codebook_emadc
        self.commitment_weight = commitment_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight
        self.cnn_type = cnn_type
        self.learnable_codebook = learnable_codebook
        self.init_codebook_path = init_codebook_path
        self.freeze_cnn = freeze_cnn
        self.cnn_checkpoint_path = cnn_checkpoint_path
        self.teacher_model_path = teacher_model_path


class PoreVQCodec(PreTrainedModel):
    config_class = PoreVQCodecConfig
    _no_split_modules = ["SignalCNN", "VectorQuantize"]

    def __init__(self, config: PoreVQCodecConfig):
        self.all_tied_weights_keys = {}
        self._tied_weights_keys = set()
        super().__init__(config)

        self.cnn_model = SignalCNN(cnn_type=config.cnn_type)
        d_model = self.cnn_model.out_channels

        self.codebook_dim = d_model
        self.cnn_type = config.cnn_type
        self.codebook_size = config.codebook_size
        self.cnn_stride = self.cnn_model.stride
        self.RF = self.cnn_model.RF
        self.margin_stride_count = 2
        self.teacher_model_path = config.teacher_model_path

        ema_update = not bool(config.learnable_codebook)
        self.vq = VectorQuantize(
            dim=d_model,
            codebook_size=config.codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=config.codebook_decay,
            threshold_ema_dead_code=config.codebook_emadc,
            commitment_weight=config.commitment_weight,
            codebook_diversity_loss_weight=config.codebook_diversity_loss_weight,
            orthogonal_reg_weight=config.orthogonal_reg_weight,
            orthogonal_reg_max_codes=256,
            orthogonal_reg_active_codes_only=True,
            learnable_codebook=config.learnable_codebook,
            ema_update=ema_update,
        )

        if config.init_codebook_path:
            self._load_init_codebook(config.init_codebook_path)
        if config.cnn_checkpoint_path:
            self._load_cnn_weights(config.cnn_checkpoint_path, freeze_cnn=config.freeze_cnn)
        if config.teacher_model_path:
            self._load_teacher_model(config.teacher_model_path)

    def _load_teacher_model(self, teacher_model_path: str) -> None:
        from bonito.util import load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher_model = load_model(teacher_model_path, device=device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        object.__setattr__(self, "teacher_model", teacher_model)

    def _load_cnn_weights(self, cnn_checkpoint_path: str, freeze_cnn: bool = False) -> None:
        checkpoint = torch.load(cnn_checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                mapped_state_dict[f"cnn_model.{key}"] = value
            elif key.startswith("cnn_model.encoder."):
                mapped_state_dict[key] = value

        model_state = self.state_dict()
        loadable_state = {
            key: value
            for key, value in mapped_state_dict.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        self.load_state_dict(loadable_state, strict=False)

        if freeze_cnn:
            for name, param in self.named_parameters():
                if name.startswith("cnn_model.encoder."):
                    param.requires_grad = False

    def _load_init_codebook(self, init_codebook_path: str) -> None:
        import numpy as np

        init_codebook = np.load(init_codebook_path)
        if len(init_codebook.shape) == 2:
            init_codebook = init_codebook[None, :, :]

        init_codebook_tensor = torch.from_numpy(init_codebook).float()
        init_codebook_tensor = init_codebook_tensor.to(self.vq._codebook.embed.device).contiguous()
        embed = self.vq._codebook.embed

        if isinstance(embed, nn.Parameter):
            with torch.no_grad():
                embed.data.copy_(init_codebook_tensor)
        else:
            self.vq._codebook.embed = init_codebook_tensor

    def forward(self, signal: torch.Tensor):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        z_cnn = self.cnn_model.encode(signal)
        z_permuted = z_cnn.permute(0, 2, 1)
        distill_loss = torch.tensor(0.0, device=signal.device)

        if self.teacher_model_path:
            teacher_model = self.teacher_model
            teacher_model.float()
            if next(teacher_model.parameters()).device != signal.device:
                teacher_model = teacher_model.to(signal.device)
                object.__setattr__(self, "teacher_model", teacher_model)

            with torch.no_grad():
                teacher_features = teacher_model.encoder[0](signal.float())
                teacher_features = teacher_model.encoder[1](teacher_features)
                teacher_features = teacher_model.encoder[2](teacher_features)

            teacher_features = teacher_features.permute(0, 2, 1).to(dtype=z_permuted.dtype)
            min_len = min(z_permuted.shape[1], teacher_features.shape[1])
            if z_permuted.shape[-1] != teacher_features.shape[-1]:
                raise ValueError(
                    "Teacher feature dim does not match student feature dim: "
                    f"teacher={teacher_features.shape[-1]}, student={z_permuted.shape[-1]}"
                )

            student_features = z_permuted[:, :min_len, :]
            teacher_features = teacher_features[:, :min_len, :]
            target = torch.ones(student_features.shape[0] * student_features.shape[1], device=signal.device)
            distill_loss = F.cosine_embedding_loss(
                student_features.reshape(-1, student_features.shape[-1]),
                teacher_features.reshape(-1, teacher_features.shape[-1]),
                target,
            )

        z_quantized_permuted, indices, vq_loss, loss_breakdown = self.vq(
            z_permuted,
            return_loss_breakdown=True,
        )
        z_quantized = z_quantized_permuted.permute(0, 2, 1)
        recon = self.cnn_model.decode(z_quantized)

        target_len = signal.shape[-1]
        current_len = recon.shape[-1]
        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))

        return recon, indices.to(device=signal.device, dtype=torch.long), vq_loss, loss_breakdown, distill_loss

    def encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        if signal.device != device:
            signal = signal.to(device)
        with torch.no_grad():
            _, indices, _, _, _ = self.forward(signal)
        return indices

    def decode_token(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        with torch.no_grad():
            z_quantized_permuted = self.vq.get_output_from_indices(token_ids)
            z_quantized = z_quantized_permuted.permute(0, 2, 1)
            recon = self.cnn_model.decode(z_quantized)
        return recon

    def save_pretrained(self, *args, **kwargs):
        teacher_model_path = self.config.teacher_model_path
        self.config.teacher_model_path = None
        try:
            return super().save_pretrained(*args, **kwargs)
        finally:
            self.config.teacher_model_path = teacher_model_path


AutoConfig.register(MODEL_TYPE, PoreVQCodecConfig)
AutoModel.register(PoreVQCodecConfig, PoreVQCodec)
