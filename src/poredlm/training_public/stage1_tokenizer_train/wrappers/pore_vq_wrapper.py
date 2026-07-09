import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_pore_vq_codec import PoreVQCodec


def _loss_item(loss_breakdown, name: str, device: torch.device) -> torch.Tensor:
    value = getattr(loss_breakdown, name, None)
    if value is None:
        return torch.tensor(0.0, device=device)
    if not isinstance(value, torch.Tensor):
        return torch.tensor(float(value), device=device)
    return value


class PoreVQWrapper(nn.Module):
    def __init__(
        self,
        codec: PoreVQCodec,
        commitment_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        distill_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.codec = codec
        self.commitment_weight = commitment_weight
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.distill_loss_weight = distill_loss_weight

    def forward(self, signal: torch.Tensor, **kwargs):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)

        recon, indices, vq_loss, loss_breakdown, distill_loss = self.codec(signal)
        recon_loss = F.mse_loss(recon, signal)

        commitment_loss = _loss_item(loss_breakdown, "commitment", signal.device)
        diversity_loss = _loss_item(loss_breakdown, "codebook_diversity", signal.device)
        orthogonal_loss = _loss_item(loss_breakdown, "orthogonal_reg", signal.device)

        loss = (
            recon_loss
            + commitment_loss * self.commitment_weight
            + diversity_loss * self.codebook_diversity_loss_weight
            + orthogonal_loss * self.orthogonal_reg_weight
            + distill_loss * self.distill_loss_weight
        )

        self.last_indices = indices.detach()
        self.last_recon = recon.detach()
        self.last_target = signal.detach()
        self.last_recon_loss = recon_loss.detach()
        self.last_commitment_loss = commitment_loss.detach()
        self.last_vq_loss = vq_loss.detach() if isinstance(vq_loss, torch.Tensor) else torch.tensor(0.0, device=signal.device)
        self.last_distill_loss = distill_loss.detach()

        return {
            "loss": loss,
            "recon": recon,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "vq_loss": vq_loss,
            "distill_loss": distill_loss,
        }

    @torch.no_grad()
    def get_metrics(self):
        metrics = {}

        if hasattr(self, "last_indices"):
            used_codes = torch.unique(self.last_indices).numel()
            metrics["codebook_usage"] = float(used_codes / self.codec.codebook_size)

        if hasattr(self, "last_recon") and hasattr(self, "last_target"):
            signal_power = torch.mean(self.last_target ** 2)
            noise_power = F.mse_loss(self.last_recon, self.last_target)
            metrics["snr_loss"] = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()

        if hasattr(self, "last_recon_loss"):
            metrics["recon_loss"] = float(self.last_recon_loss.item())
        if hasattr(self, "last_commitment_loss"):
            metrics["commitment_loss"] = float(self.last_commitment_loss.item())
        if hasattr(self, "last_distill_loss"):
            metrics["distill_loss"] = float(self.last_distill_loss.item())

        metrics.setdefault("snr_loss", 0.0)
        metrics.setdefault("codebook_usage", 0.0)
        metrics.setdefault("codebook_max_entropy", 0.0)
        return metrics
