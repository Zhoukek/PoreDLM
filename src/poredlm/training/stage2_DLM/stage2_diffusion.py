"""Stage 2 training loop for diffusion language model pretraining."""

from __future__ import annotations

from poredlm.training.runtime import bootstrap_training
from poredlm.utils.distributed import cleanup_distributed


def train_stage2(config_path: str) -> None:
    """Train denoising or flow matching over continuous signal embeddings."""

    runtime = bootstrap_training(config_path)
    try:
        raise NotImplementedError(
            "Stage 2 training is not implemented yet. "
            f"rank={runtime.context.rank}, world_size={runtime.context.world_size}, "
            f"seed={runtime.reproducibility.seed}"
        )
    finally:
        cleanup_distributed()
