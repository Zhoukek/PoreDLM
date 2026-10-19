import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_pore_block_mlp_vq import BlockMLPVQCodec


def _loss_item(breakdown, name, device):
    value = getattr(breakdown, name, None)
    if value is None:
        return torch.tensor(0.0, device=device)
    return value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=device)


class BlockMLPVQWrapper(nn.Module):
    """Training wrapper with the existing stage1 VQ wrapper contract."""

    def __init__(self, codec: BlockMLPVQCodec, commitment_weight=1.0,
                 codebook_diversity_loss_weight=1.0, orthogonal_reg_weight=1.0):
        super().__init__()
        self.codec = codec
        self.commitment_weight = commitment_weight
        self.codebook_diversity_loss_weight = codebook_diversity_loss_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight

    def forward(self, signal: torch.Tensor, **kwargs):
        if signal.ndim == 2:
            signal = signal.unsqueeze(1)
        recon, indices, vq_loss, breakdown, distill_loss = self.codec(signal)
        recon_loss = F.mse_loss(recon, signal)
        commitment = _loss_item(breakdown, "commitment", signal.device)
        diversity = _loss_item(breakdown, "codebook_diversity", signal.device)
        orthogonal = _loss_item(breakdown, "orthogonal_reg", signal.device)
        loss = recon_loss + self.commitment_weight * commitment
        loss = loss + self.codebook_diversity_loss_weight * diversity
        loss = loss + self.orthogonal_reg_weight * orthogonal
        self.last_indices = indices.detach()
        self.last_recon = recon.detach()
        self.last_target = signal.detach()
        self.last_recon_loss = recon_loss.detach()
        self.last_commitment_loss = commitment.detach()
        self.last_vq_loss = vq_loss.detach() if isinstance(vq_loss, torch.Tensor) else torch.tensor(0.0, device=signal.device)
        return {"loss": loss, "recon": recon, "recon_loss": recon_loss,
                "commitment_loss": commitment, "vq_loss": vq_loss,
                "distill_loss": distill_loss}

    @torch.no_grad()
    def get_metrics(self):
        metrics = {"codebook_usage": 0.0, "snr_loss": 0.0}
        if hasattr(self, "last_indices"):
            metrics["codebook_usage"] = float(torch.unique(self.last_indices).numel() / self.codec.codebook_size)
        if hasattr(self, "last_recon"):
            power = torch.mean(self.last_target ** 2)
            noise = F.mse_loss(self.last_recon, self.last_target)
            metrics["snr_loss"] = 10 * torch.log10(power / (noise + 1e-8)).item()
        metrics["recon_loss"] = float(getattr(self, "last_recon_loss", torch.tensor(0.0)).item())
        metrics["commitment_loss"] = float(getattr(self, "last_commitment_loss", torch.tensor(0.0)).item())
        return metrics
