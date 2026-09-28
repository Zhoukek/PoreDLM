"""Continuous CNN encoder and denoising reconstruction model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig


MODEL_TYPE = "continuous_signal_cnn"


class SignalCNN(nn.Module):
    """Self-contained CNN backbone used by the continuous signal model.

    The module keeps the same parameter names and architecture as the public
    ``cnn_type=0`` backbone, but does not import the VQ implementation.
    """

    def __init__(self, cnn_type: int = 0) -> None:
        super().__init__()
        if int(cnn_type) != 0:
            raise ValueError("ContinuousSignalCNN currently supports cnn_type=0 only.")
        self.cnn_type = 0
        self.out_channels = 768
        self.stride = 5
        self.receptive_field = 27
        self.RF = 27
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(4, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(16, 768, kernel_size=19, stride=5, padding=9, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                768, 16, kernel_size=19, stride=5, padding=9,
                output_padding=1, bias=False,
            ),
            nn.SiLU(),
            nn.Conv1d(16, 4, kernel_size=5, padding=2, bias=False),
            nn.SiLU(),
            nn.Conv1d(4, 1, kernel_size=5, padding=2, bias=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class ContinuousCNNConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(self, hidden_size: int = 768, cnn_type: int = 0,
                 noise_std: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.cnn_type = int(cnn_type)
        self.noise_std = float(noise_std)


class ContinuousSignalCNN(PreTrainedModel):
    """Continuous CNN encoder without quantization or codebook layers."""

    stride = 5

    config_class = ContinuousCNNConfig
    _no_split_modules = ["SignalCNN"]

    def __init__(self, config: ContinuousCNNConfig | None = None):
        config = config or ContinuousCNNConfig()
        super().__init__(config)
        self.cnn_model = SignalCNN(cnn_type=config.cnn_type)
        if self.cnn_model.out_channels != config.hidden_size:
            raise ValueError(
                f"hidden_size={config.hidden_size} does not match SignalCNN output "
                f"channels={self.cnn_model.out_channels}."
            )
        self.hidden_size = self.cnn_model.out_channels
        self.stride = self.cnn_model.stride
        self.input_noise_std = config.noise_std

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)
        return self.cnn_model.encode(signal)

    def decode(self, latent: torch.Tensor, target_length: int) -> torch.Tensor:
        output = self.cnn_model.decode(latent)
        if output.shape[-1] > target_length:
            return output[..., :target_length]
        if output.shape[-1] < target_length:
            return F.pad(output, (0, target_length - output.shape[-1]))
        return output

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)
        noisy = signal + torch.randn_like(signal) * self.input_noise_std if self.training else signal
        latent = self.encode(noisy)
        reconstruction = self.decode(latent, signal.shape[-1])
        loss = F.smooth_l1_loss(reconstruction, signal)
        return {"loss": loss, "reconstruction": reconstruction, "latent": latent}


AutoConfig.register(MODEL_TYPE, ContinuousCNNConfig)
AutoModel.register(ContinuousCNNConfig, ContinuousSignalCNN)
