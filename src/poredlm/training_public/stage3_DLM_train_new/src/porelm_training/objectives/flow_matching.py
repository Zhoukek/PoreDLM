from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..config import FlowMatchingConfig
from .base import ObjectiveOutput, TrainingObjective


class FlowMatchingObjective(TrainingObjective):
    """Conditional flow matching in the Stage-2 latent space."""

    def __init__(self, config: FlowMatchingConfig):
        self.config = config

    def _sample_timesteps(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        mean: float,
        std: float,
    ) -> torch.Tensor:
        if self.config.timestep_schedule == "uniform":
            return torch.rand(count, device=device, dtype=dtype)
        if self.config.timestep_schedule == "logit_normal":
            return torch.sigmoid(torch.randn(count, device=device, dtype=dtype) * std + mean)
        raise ValueError(f"Unknown timestep schedule: {self.config.timestep_schedule}")

    def _sample_guidance_scale(
        self,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        uniform = torch.rand(count, device=device, dtype=dtype)
        lower = torch.tensor(1.0 + self.config.guidance_scale_min, device=device, dtype=dtype)
        upper = torch.tensor(1.0 + self.config.guidance_scale_max, device=device, dtype=dtype)
        return lower * torch.exp(uniform * torch.log(upper / lower)) - 1.0

    @staticmethod
    def _synchronize_boolean(value: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(value, src=0)
        return value

    @staticmethod
    def _zero_parameter_loss(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
        value = torch.zeros((), device=device)
        for parameter in module.parameters():
            value = value + parameter.float().sum() * 0.0
        return value

    def __call__(self, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> ObjectiveOutput:
        core_model = model.module if hasattr(model, "module") else model
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        condition_mask = batch["condition_mask"]
        clean = core_model.encode(input_ids, batch.get("encoder_attention_mask"))
        condition_mask_3d = condition_mask.to(clean.dtype).unsqueeze(-1)
        loss_mask = attention_mask.to(clean.dtype) * (1.0 - condition_mask.to(clean.dtype))
        batch_size, sequence_length = input_ids.shape

        timestep = self._sample_timesteps(
            batch_size,
            clean.device,
            clean.dtype,
            mean=self.config.timestep_mean,
            std=self.config.timestep_std,
        )
        timestep_3d = timestep[:, None, None]
        noise = torch.randn_like(clean) * self.config.noise_scale
        noisy = timestep_3d * clean + (1.0 - timestep_3d) * noise
        noisy = condition_mask_3d * clean + (1.0 - condition_mask_3d) * noisy
        target_velocity = (clean - noisy) / torch.clamp(
            1.0 - timestep_3d, min=self.config.timestep_epsilon
        )
        guidance_scale = self._sample_guidance_scale(
            batch_size, clean.device, clean.dtype
        )

        decoder_active = self._synchronize_boolean(
            torch.rand((), device=clean.device) < self.config.decoder_probability
        )
        if bool(decoder_active.item()):
            decoder_timestep = torch.sigmoid(
                torch.randn(
                    batch_size, sequence_length, 1, device=clean.device, dtype=clean.dtype
                )
                * self.config.decoder_timestep_std
                + self.config.decoder_timestep_mean
            )
            decoder_noise = torch.randn_like(clean) * self.config.decoder_noise_scale
            decoder_latent = decoder_timestep * clean + (1.0 - decoder_timestep) * decoder_noise
            if self.config.self_condition_probability > 0:
                decoder_latent = torch.cat([decoder_latent, torch.zeros_like(decoder_latent)], dim=-1)
            decoder_output = model(
                decoder_latent,
                torch.ones_like(timestep),
                attention_mask=attention_mask,
                guidance_scale=guidance_scale,
                decoder_active=True,
            )
            assert decoder_output.logits is not None
            logits = decoder_output.logits
            token_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                input_ids.reshape(-1),
                reduction="none",
            ).view_as(input_ids)
            loss = (token_loss * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1.0)
            loss = loss + decoder_output.prediction.float().sum() * 0.0
            if self.config.self_condition_probability <= 0:
                loss = loss + self._zero_parameter_loss(
                    core_model.denoiser.self_condition_projection, clean.device
                )
            zero = loss.detach().new_zeros(())
            return ObjectiveOutput(
                loss=loss,
                metrics={"flow_loss": zero, "decoder_loss": loss.detach(), "decoder_step": zero.new_ones(())},
            )

        self_condition = None
        if self.config.self_condition_probability > 0:
            use_self_condition = torch.rand((), device=clean.device) < self.config.self_condition_probability
            if bool(use_self_condition.item()):
                with torch.no_grad():
                    initial_output = core_model(
                        torch.cat([noisy, torch.zeros_like(noisy)], dim=-1),
                        timestep,
                        attention_mask=attention_mask,
                        guidance_scale=guidance_scale,
                    )
                self_condition = initial_output.prediction
            else:
                self_condition = torch.zeros_like(noisy)
        model_input = torch.cat([noisy, self_condition], dim=-1) if self_condition is not None else noisy
        denoiser_output = model(
            model_input,
            timestep,
            attention_mask=attention_mask,
            guidance_scale=guidance_scale,
        )
        predicted_velocity = (denoiser_output.prediction - noisy) / torch.clamp(
            1.0 - timestep_3d, min=self.config.timestep_epsilon
        )
        per_token = (predicted_velocity - target_velocity).float().pow(2).mean(dim=-1)
        loss = (per_token * loss_mask.float()).sum() / loss_mask.sum().clamp_min(1.0)
        loss = loss + self._zero_parameter_loss(
            core_model.denoiser.decoder_projection, clean.device
        )
        loss = loss + self._zero_parameter_loss(core_model.denoiser.token_head, clean.device)
        if self.config.self_condition_probability <= 0:
            loss = loss + self._zero_parameter_loss(
                core_model.denoiser.self_condition_projection, clean.device
            )
        zero = loss.detach().new_zeros(())
        return ObjectiveOutput(
            loss=loss,
            metrics={"flow_loss": loss.detach(), "decoder_loss": zero, "decoder_step": zero},
        )
