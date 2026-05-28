"""CLI wrapper for Stage 2 training."""

from __future__ import annotations

import argparse

from poredlm.training.stage2_diffusion import train_stage2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_diffusion.yaml")
    args = parser.parse_args()
    train_stage2(args.config)


if __name__ == "__main__":
    main()
