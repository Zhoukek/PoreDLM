# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys

from .config import add_train_arguments


def main() -> None:
    parser = argparse.ArgumentParser(prog="basecallx", description="Clean dcbasecaller command wrapper.")
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = add_train_arguments(sub.add_parser("train", help="Run the refactored training pipeline."))
    eval_parser = sub.add_parser("eval", help="Forward to the compatible evaluator.")
    eval_parser.add_argument("args", nargs=argparse.REMAINDER)
    infer_parser = sub.add_parser("infer", help="Forward to the compatible inference command.")
    infer_parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command == "train":
        from .train import run_training

        command_index = sys.argv.index("train") if "train" in sys.argv else 1
        run_training(sys.argv[command_index + 1 :])
        return
    if args.command == "eval":
        from basecall.eval import main as eval_main

        sys.argv = ["basecall-eval", *args.args]
        eval_main()
        return
    if args.command == "infer":
        from basecall.infer import main as infer_main

        sys.argv = ["basecall-infer", *args.args]
        infer_main()
        return
    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
