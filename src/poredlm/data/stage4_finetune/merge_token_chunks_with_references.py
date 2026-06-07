#!/usr/bin/env python3
"""Merge Stage2 token chunks and reference arrays into Stage4 jsonl.gz files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CHUNKS_SUFFIX = "_chunks.npy"
REFERENCES_SUFFIX = "_references.npy"
REFERENCE_LENGTHS_SUFFIX = "_reference_lengths.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge *_chunks.npy with reference/*_references.npy and "
            "reference/*_reference_lengths.npy into *.jsonl.gz files."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_chunks.npy files, for example /path/to/train.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Directory containing reference npy files. Defaults to <input-dir>/reference.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output *.jsonl.gz files. Defaults to <input-dir>.",
    )
    parser.add_argument(
        "--pattern",
        default=f"*{CHUNKS_SUFFIX}",
        help=f"Chunks glob pattern. Defaults to *{CHUNKS_SUFFIX}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a reference row has nonzero values after reference_length.",
    )
    parser.add_argument(
        "--gzip-compresslevel",
        type=int,
        default=1,
        help="Gzip compression level, 1 is fastest and 9 is smallest. Defaults to 1.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to process in parallel. Defaults to 1.",
    )
    return parser.parse_args()


def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def iter_chunk_records(chunks: np.ndarray) -> Iterable[Any]:
    if chunks.ndim == 0:
        yield chunks.item()
        return
    for item in chunks:
        yield item


def normalize_chunk_record(record: Any, *, fallback_id: str) -> dict[str, str]:
    if isinstance(record, np.ndarray) and record.ndim == 0:
        record = record.item()
    elif isinstance(record, np.ndarray):
        if record.dtype.kind in {"U", "S"}:
            text = "".join(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in record.reshape(-1)
            )
            return normalize_chunk_record(text, fallback_id=fallback_id)
        if record.dtype == object:
            flat = record.reshape(-1)
            if len(flat) == 1:
                return normalize_chunk_record(flat[0], fallback_id=fallback_id)
            if all(isinstance(item, (str, bytes)) for item in flat):
                text = "".join(
                    item.decode("utf-8") if isinstance(item, bytes) else item
                    for item in flat
                )
                return normalize_chunk_record(text, fallback_id=fallback_id)
        raise TypeError(
            "chunks.npy records must already contain text strings or JSON/dict records with a 'text' key. "
            f"Got ndarray dtype={record.dtype}, shape={record.shape} for {fallback_id}."
        )

    if isinstance(record, bytes):
        record = record.decode("utf-8")

    if isinstance(record, str):
        stripped = record.strip()
        if stripped.startswith("{"):
            record = json.loads(stripped)
        else:
            return {"id": fallback_id, "text": stripped}

    if isinstance(record, dict):
        if "text" not in record:
            raise KeyError(f"Chunk record is missing 'text': {record.keys()}")
        return {
            "id": str(record.get("id", fallback_id)),
            "text": str(record["text"]),
        }

    if hasattr(record, "item"):
        item = record.item()
        if item is not record:
            return normalize_chunk_record(item, fallback_id=fallback_id)

    raise TypeError(f"Unsupported chunk record type: {type(record)}")


def reference_to_bases(reference_row: np.ndarray, reference_length: int, *, strict: bool) -> str:
    reference_length = int(reference_length)
    if reference_length < 0:
        raise ValueError(f"reference_length must be >= 0, got {reference_length}")
    if reference_length > int(reference_row.shape[0]):
        raise ValueError(
            f"reference_length={reference_length} exceeds reference width={reference_row.shape[0]}"
        )

    if strict and reference_length < int(reference_row.shape[0]):
        tail = reference_row[reference_length:]
        if np.any(tail != 0):
            raise ValueError("Found nonzero padded values after reference_length")

    bases = reference_row[:reference_length].astype(np.int64, copy=False)
    if bases.size == 0:
        return ""
    if bases.min(initial=0) >= 0 and bases.max(initial=0) <= 9:
        return np.array2string(
            bases,
            separator="",
            max_line_width=np.iinfo(np.int32).max,
            threshold=np.iinfo(np.int32).max,
        )[1:-1]
    return "".join(str(int(base)) for base in bases)


def prefix_from_chunks_path(path: Path) -> str:
    if not path.name.endswith(CHUNKS_SUFFIX):
        raise ValueError(f"Expected file ending with {CHUNKS_SUFFIX}: {path}")
    return path.name[: -len(CHUNKS_SUFFIX)]


def merge_one(
    chunks_path: Path,
    *,
    reference_dir: Path,
    output_dir: Path,
    overwrite: bool,
    strict: bool,
    gzip_compresslevel: int,
) -> tuple[Path, int]:
    prefix = prefix_from_chunks_path(chunks_path)
    references_path = reference_dir / f"{prefix}{REFERENCES_SUFFIX}"
    lengths_path = reference_dir / f"{prefix}{REFERENCE_LENGTHS_SUFFIX}"
    output_path = output_dir / f"{prefix}.jsonl.gz"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    chunks = load_npy(chunks_path)
    references = load_npy(references_path)
    reference_lengths = load_npy(lengths_path)

    num_chunks = len(chunks)
    num_references = int(references.shape[0])
    num_lengths = len(reference_lengths)
    if num_chunks != num_references or num_chunks != num_lengths:
        raise ValueError(
            f"Sample count mismatch for {prefix}: "
            f"chunks={num_chunks}, references={num_references}, reference_lengths={num_lengths}"
        )
    if references.ndim != 2:
        raise ValueError(f"references must be a 2D array, got shape={references.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=gzip_compresslevel) as handle:
        for index, (chunk_record, reference_row, reference_length) in enumerate(
            zip(iter_chunk_records(chunks), references, reference_lengths)
        ):
            fallback_id = f"{chunks_path.stem}_chunk_{index}"
            chunk = normalize_chunk_record(chunk_record, fallback_id=fallback_id)
            merged = {
                "text": chunk["text"],
                "bases": reference_to_bases(reference_row, int(reference_length), strict=strict),
            }
            handle.write(json.dumps(merged, ensure_ascii=False) + "\n")

    return output_path, num_chunks


def merge_one_from_paths(
    chunks_path: str,
    reference_dir: str,
    output_dir: str,
    overwrite: bool,
    strict: bool,
    gzip_compresslevel: int,
) -> tuple[str, int]:
    output_path, count = merge_one(
        Path(chunks_path),
        reference_dir=Path(reference_dir),
        output_dir=Path(output_dir),
        overwrite=overwrite,
        strict=strict,
        gzip_compresslevel=gzip_compresslevel,
    )
    return str(output_path), count


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir) if args.reference_dir else input_dir / "reference"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    chunks_files = sorted(input_dir.glob(args.pattern))
    if not chunks_files:
        raise FileNotFoundError(f"No chunks files matching {args.pattern!r} under {input_dir}")

    total_records = 0
    workers = max(1, int(args.workers))
    gzip_compresslevel = int(args.gzip_compresslevel)
    if gzip_compresslevel < 0 or gzip_compresslevel > 9:
        raise ValueError("--gzip-compresslevel must be in [0, 9]")

    if workers == 1:
        for chunks_path in chunks_files:
            output_path, count = merge_one(
                chunks_path,
                reference_dir=reference_dir,
                output_dir=output_dir,
                overwrite=bool(args.overwrite),
                strict=bool(args.strict),
                gzip_compresslevel=gzip_compresslevel,
            )
            total_records += count
            print(f"[OK] {chunks_path.name} -> {output_path} ({count} records)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    merge_one_from_paths,
                    str(chunks_path),
                    str(reference_dir),
                    str(output_dir),
                    bool(args.overwrite),
                    bool(args.strict),
                    gzip_compresslevel,
                ): chunks_path
                for chunks_path in chunks_files
            }
            for future in as_completed(futures):
                chunks_path = futures[future]
                output_path, count = future.result()
                total_records += count
                print(f"[OK] {chunks_path.name} -> {output_path} ({count} records)")

    print(f"[Done] files={len(chunks_files)}, records={total_records}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
