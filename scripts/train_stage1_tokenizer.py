"""CLI wrapper for Stage 1 training."""

from __future__ import annotations

import argparse

from poredlm.training.stage1_tokenizer import train_stage1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_tokenizer.yaml")
    args = parser.parse_args()
    train_stage1(args.config)


if __name__ == "__main__":
    main()
