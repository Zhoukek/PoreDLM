"""Stage 3 downstream fine-tuning loop."""

from __future__ import annotations

from poredlm.training.runtime import bootstrap_training
from poredlm.utils.distributed import cleanup_distributed


def train_stage3(config_path: str) -> None:
    """Fine-tune PoreDLM for basecalling or modification-aware tasks."""

    runtime = bootstrap_training(config_path)
    try:
        raise NotImplementedError(
            "Stage 3 fine-tuning is not implemented yet. "
            f"rank={runtime.context.rank}, world_size={runtime.context.world_size}, "
            f"seed={runtime.reproducibility.seed}"
        )
    finally:
        cleanup_distributed()
