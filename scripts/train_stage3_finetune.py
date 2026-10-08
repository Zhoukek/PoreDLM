"""CLI wrapper for Stage 3 fine-tuning."""

from __future__ import annotations

import argparse

from poredlm.training.stage3_finetune import train_stage3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_finetune.yaml")
    args = parser.parse_args()
    train_stage3(args.config)


if __name__ == "__main__":
    main()
