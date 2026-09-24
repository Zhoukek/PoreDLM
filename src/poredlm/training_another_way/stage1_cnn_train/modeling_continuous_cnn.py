"""Continuous CNN encoder and denoising reconstruction model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig

from poredlm.training_public.stage1_tokenizer_train.modeling_pore_vq_codec import SignalCNN


MODEL_TYPE = "continuous_signal_cnn"


class ContinuousCNNConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(self, hidden_size: int = 768, cnn_type: int = 0,
                 noise_std: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.cnn_type = int(cnn_type)
        self.noise_std = float(noise_std)


class ContinuousSignalCNN(PreTrainedModel):
    """The public VQ-CNN backbone without quantization or codebook layers.

    This reuses ``SignalCNN`` from the public VQ codec. Only its CNN
    encoder/decoder is used; the VectorQuantize module is never called.
    """

    stride = 4

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
