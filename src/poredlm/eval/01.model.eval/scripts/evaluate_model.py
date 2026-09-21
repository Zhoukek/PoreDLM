#!/usr/bin/env python3
"""Run NanoRepDist's PoreDLM evaluation with this project's corpus/output defaults."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVALUATION_ROOT.parent


def arguments_with_defaults(argv: list[str]) -> list[str]:
    """Keep NanoRepDist options intact and fill only project-specific missing paths."""
    if "--help" in argv or "-h" in argv:
        return ["evaluate-model", *argv]

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--strategy", default="apple")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args, _ = parser.parse_known_args(argv)
    result = ["evaluate-model", *argv]
    if args.corpus_root is None:
        result.extend(
            ["--corpus-root", str(EVALUATION_ROOT.parent / "00.corpus" / args.strategy)]
        )
    if args.output_dir is None and args.model_dir is not None:
        model_root = args.model_dir.expanduser().resolve()
        if model_root.name == "hf_dlm":
            model_root = model_root.parent
        output = (
            EVALUATION_ROOT
            / model_root.name
            / "runs"
            / f"{datetime.now(tz=timezone.utc).astimezone():%Y%m%d}_nanorepdist_modes"
        )
        result.extend(["--output-dir", str(output)])
    return result


def main(argv: list[str] | None = None) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "script" / "NanoRepDist" / "src"))
    print(f"Using NanoRepDist from {PROJECT_ROOT / 'script' / 'NanoRepDist' / 'src'}")
    from nanorepdist.cli import main as nanorepdist_main

    nanorepdist_main(arguments_with_defaults(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
