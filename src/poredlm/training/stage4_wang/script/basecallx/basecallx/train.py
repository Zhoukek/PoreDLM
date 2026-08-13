# -*- coding: utf-8 -*-
from __future__ import annotations

from .loops import evaluate, train_one_epoch
from .runner import RunState, run_training


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
