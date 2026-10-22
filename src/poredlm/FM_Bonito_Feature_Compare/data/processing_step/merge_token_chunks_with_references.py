#!/usr/bin/env python3
"""Add per-record reference base strings to chunks jsonl.gz files."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read *_chunks.jsonl.gz files and matching *_references.npy files, "
            "then write new jsonl.gz files with each record's 'bases' field filled "
            "from the corresponding reference row. Use either single-file mode or "
            "directory batch mode."
        )
    )
    parser.add_argument(
        "--chunks-jsonl-gz",
        default=None,
        help="Input chunks jsonl.gz file, for example 250F..._chunks.jsonl.gz.",
    )
    parser.add_argument(
        "--references-npy",
        default=None,
        help="Matching references npy file, for example 250F..._references.npy.",
    )
    parser.add_argument(
        "--output-jsonl-gz",
        default=None,
        help="Output jsonl.gz file. Use a different path from the input unless --overwrite.",
    )
    parser.add_argument(
        "--chunks-dir",
        default=None,
        help="Directory containing *_chunks.jsonl.gz files for batch mode.",
    )
    parser.add_argument(
        "--references-dir",
        default=None,
        help="Directory containing matching *_references.npy files for batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for batch outputs. Output file names match the input chunks file names.",
    )
    parser.add_argument(
        "--pattern",
        default="*_chunks.jsonl.gz",
        help="Chunks glob pattern for batch mode. Defaults to *_chunks.jsonl.gz.",
    )
    parser.add_argument(
        "--keep-padding-zeros",
        action="store_true",
        help="Keep trailing 0 values in the bases string. Defaults to trimming trailing padding zeros.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--gzip-compresslevel",
        type=int,
        default=1,
        help="Gzip compression level, 1 is fastest and 9 is smallest. Defaults to 1.",
    )
    return parser.parse_args()


def reference_row_to_bases(row: np.ndarray, *, trim_padding_zeros: bool) -> str:
    values = np.asarray(row).reshape(-1)
    if trim_padding_zeros:
        nonzero = np.flatnonzero(values)
        values = values[: int(nonzero[-1]) + 1] if nonzero.size else values[:0]

    if values.size == 0:
        return ""

    return "".join(str(int(value)) for value in values)


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            yield record


def count_jsonl_gz_records(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def merge_bases(
    *,
    chunks_path: Path,
    references_path: Path,
    output_path: Path,
    trim_padding_zeros: bool,
    overwrite: bool,
    gzip_compresslevel: int,
) -> int:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if not references_path.exists():
        raise FileNotFoundError(f"References file not found: {references_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")
    if chunks_path.resolve() == output_path.resolve():
        raise ValueError("Output path must be different from input chunks path")
    if gzip_compresslevel < 0 or gzip_compresslevel > 9:
        raise ValueError("--gzip-compresslevel must be in [0, 9]")

    references = np.load(references_path, allow_pickle=False)
    if references.ndim != 2:
        raise ValueError(f"references npy must be a 2D array, got shape={references.shape}")

    num_chunks = count_jsonl_gz_records(chunks_path)
    num_references = int(references.shape[0])
    if num_chunks != num_references:
        raise ValueError(
            f"Record count mismatch: chunks={num_chunks}, references={num_references}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=gzip_compresslevel) as out:
        for record, reference_row in zip(iter_jsonl_gz(chunks_path), references):
            record["bases"] = reference_row_to_bases(
                reference_row,
                trim_padding_zeros=trim_padding_zeros,
            )
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    return num_chunks


def prefix_from_chunks_path(path: Path) -> str:
    suffix = "_chunks.jsonl.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Expected chunks file ending with {suffix}: {path}")
    return path.name[: -len(suffix)]


def iter_batch_pairs(
    *,
    chunks_dir: Path,
    references_dir: Path,
    output_dir: Path,
    pattern: str,
) -> Iterator[tuple[Path, Path, Path]]:
    chunks_files = sorted(chunks_dir.glob(pattern))
    if not chunks_files:
        raise FileNotFoundError(f"No chunks files matching {pattern!r} under {chunks_dir}")

    for chunks_path in chunks_files:
        prefix = prefix_from_chunks_path(chunks_path)
        references_path = references_dir / f"{prefix}_references.npy"
        output_path = output_dir / chunks_path.name
        yield chunks_path, references_path, output_path


def run_batch(args: argparse.Namespace) -> None:
    chunks_dir = Path(args.chunks_dir)
    references_dir = Path(args.references_dir) if args.references_dir else chunks_dir
    output_dir = Path(args.output_dir) if args.output_dir else chunks_dir / "with_bases"

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    if not references_dir.exists():
        raise FileNotFoundError(f"References directory not found: {references_dir}")

    total_records = 0
    total_files = 0
    for chunks_path, references_path, output_path in iter_batch_pairs(
        chunks_dir=chunks_dir,
        references_dir=references_dir,
        output_dir=output_dir,
        pattern=str(args.pattern),
    ):
        count = merge_bases(
            chunks_path=chunks_path,
            references_path=references_path,
            output_path=output_path,
            trim_padding_zeros=not bool(args.keep_padding_zeros),
            overwrite=bool(args.overwrite),
            gzip_compresslevel=int(args.gzip_compresslevel),
        )
        total_files += 1
        total_records += count
        print(f"[OK] {chunks_path.name} -> {output_path} ({count} records)")

    print(f"[Done] files={total_files}, records={total_records}, output_dir={output_dir}")


def run_single_file(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in (
            ("--chunks-jsonl-gz", args.chunks_jsonl_gz),
            ("--references-npy", args.references_npy),
            ("--output-jsonl-gz", args.output_jsonl_gz),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Single-file mode requires "
            + ", ".join(missing)
            + ". For batch mode, pass --chunks-dir."
        )

    count = merge_bases(
        chunks_path=Path(args.chunks_jsonl_gz),
        references_path=Path(args.references_npy),
        output_path=Path(args.output_jsonl_gz),
        trim_padding_zeros=not bool(args.keep_padding_zeros),
        overwrite=bool(args.overwrite),
        gzip_compresslevel=int(args.gzip_compresslevel),
    )
    print(f"[OK] wrote {count} records to {args.output_jsonl_gz}")


def main() -> None:
    args = parse_args()
    if args.chunks_dir:
        run_batch(args)
    else:
        run_single_file(args)


if __name__ == "__main__":
    main()
