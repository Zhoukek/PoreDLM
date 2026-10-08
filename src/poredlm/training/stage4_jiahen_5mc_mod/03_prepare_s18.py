#!/usr/bin/env python3
"""Prepare the strict 200x-per-label-7mer S18 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


COMMON_COLUMNS = [
    "window_id", "site_pos0", "label", "bed_percent", "bed_coverage",
    "block_id", "ref_7mer", "fast5_raw_read_id", "source_part", "mapq",
    "signal_len", "v003_stone_tokens_raw", "v003_stone_input_ids",
    "v600_apple_tokens_raw", "v600_apple_input_ids",
    "v610_apple_tokens_raw", "v610_apple_input_ids",
]
TOKEN_ID_COLUMNS = [
    "v003_stone_input_ids", "v600_apple_input_ids", "v610_apple_input_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr19-corpus", required=True)
    parser.add_argument("--chr16-corpus", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-per-label-kmer", type=int, default=200)
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=1536,
        help="Maximum input length shared by all three model interfaces",
    )
    return parser.parse_args()


def digest_u64(value: str, person: bytes) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode(), digest_size=8, person=person).digest(),
        "big",
    )


def validate_table(table: pa.Table, name: str) -> None:
    if table.num_rows == 0:
        raise RuntimeError(f"{name} corpus is empty")
    required = set(COMMON_COLUMNS)
    if set(table.column_names) != required:
        missing = sorted(required - set(table.column_names))
        raise RuntimeError(f"{name} is missing required columns: {missing}")
    labels = table["label"].to_pylist()
    if set(labels) != {0, 1}:
        raise RuntimeError(f"{name} must contain both labels, got {sorted(set(labels))}")
    ids = table["window_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{name} contains duplicate window_id values")
    kmers = table["ref_7mer"].to_pylist()
    if any(len(kmer) != 7 or kmer[3:5] != "CG" or set(kmer) - set("ACGT") for kmer in kmers):
        raise RuntimeError(f"{name} contains a non-NNNCGNN ref_7mer")


def filter_common(table: pa.Table, common: set[str]) -> pa.Table:
    mask = pc.is_in(table["ref_7mer"], value_set=pa.array(sorted(common)))
    return table.filter(mask)


def filter_model_compatible(table: pa.Table, max_length: int) -> tuple[pa.Table, int]:
    data = table.to_pydict()
    keep = []
    for index in range(table.num_rows):
        keep.append(all(len(data[column][index]) <= max_length for column in TOKEN_ID_COLUMNS))
    mask = np.asarray(keep, dtype=bool)
    return table.filter(pa.array(mask)), int((~mask).sum())


def cap_per_label_kmer(table: pa.Table, target: int) -> pa.Table:
    data = table.to_pydict()
    order = sorted(
        range(table.num_rows),
        key=lambda index: (
            int(data["label"][index]),
            str(data["ref_7mer"][index]),
            str(data["window_id"][index]),
        ),
    )
    counts: Counter[tuple[int, str]] = Counter()
    keep = []
    for index in order:
        key = (int(data["label"][index]), str(data["ref_7mer"][index]))
        if counts[key] >= target:
            continue
        counts[key] += 1
        keep.append(index)
    keep.sort()
    return table.take(pa.array(keep, type=pa.int64()))


def metadata_table(table: pa.Table, dataset: str) -> pa.Table:
    rows = table.to_pydict()
    n = table.num_rows
    if dataset == "chr19":
        fold = [digest_u64(str(read), b"s18fold") % 5 for read in rows["fast5_raw_read_id"]]
        role = ["cv" for _ in range(n)]
    else:
        calibration = [digest_u64(str(read), b"s18split") < (1 << 63) for read in rows["fast5_raw_read_id"]]
        fold = [-1 for _ in range(n)]
        role = ["baseline" if flag and int(label) == 0 else "test" for flag, label in zip(calibration, rows["label"])]
        # A positive window on a calibration read is intentionally excluded.
        role = ["excluded_calibration_read" if flag and int(label) == 1 else value for flag, label, value in zip(calibration, rows["label"], role)]
    return pa.table({
        "dataset": [dataset] * n,
        "row_index": np.arange(n, dtype=np.int64),
        "window_id": [str(value) for value in rows["window_id"]],
        "site_pos0": np.asarray(rows["site_pos0"], dtype=np.int64),
        "label": np.asarray(rows["label"], dtype=np.int8),
        "ref_7mer": [str(value) for value in rows["ref_7mer"]],
        "fast5_raw_read_id": [str(value) for value in rows["fast5_raw_read_id"]],
        "block_id": np.asarray(rows["block_id"], dtype=np.int32),
        "fold": np.asarray(fold, dtype=np.int8),
        "role": role,
    })


def count_summary(meta: pa.Table) -> dict[str, object]:
    data = meta.to_pydict()
    labels = np.asarray(data["label"])
    kmers = np.asarray(data["ref_7mer"], dtype=object)
    summary: dict[str, object] = {"rows": meta.num_rows, "reads": len(set(data["fast5_raw_read_id"]))}
    for label in (0, 1):
        selected = labels == label
        counts = Counter(kmers[selected].tolist())
        summary[f"label_{label}_rows"] = int(selected.sum())
        summary[f"label_{label}_kmers"] = len(counts)
        summary[f"label_{label}_sites"] = len(set(np.asarray(data["site_pos0"])[selected].tolist()))
        if counts:
            summary[f"label_{label}_per_kmer_min_max"] = [min(counts.values()), max(counts.values())]
    return summary


def main() -> int:
    args = parse_args()
    if args.min_per_label_kmer < 1:
        raise ValueError("min-per-label-kmer must be positive")
    if args.max_position_embeddings < 3:
        raise ValueError("max-position-embeddings must be at least 3")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chr19 = pq.read_table(args.chr19_corpus, columns=COMMON_COLUMNS)
    chr16 = pq.read_table(args.chr16_corpus, columns=COMMON_COLUMNS)
    validate_table(chr19, "chr19")
    validate_table(chr16, "chr16")
    chr19, chr19_incompatible = filter_model_compatible(chr19, args.max_position_embeddings)
    chr16, chr16_incompatible = filter_model_compatible(chr16, args.max_position_embeddings)

    def label_counts(table: pa.Table, label: int) -> Counter[str]:
        data = table.to_pydict()
        return Counter(
            str(kmer) for kmer, value in zip(data["ref_7mer"], data["label"])
            if int(value) == label
        )

    group_counts = {
        "chr19_label0": label_counts(chr19, 0),
        "chr19_label1": label_counts(chr19, 1),
        "chr16_label0": label_counts(chr16, 0),
        "chr16_label1": label_counts(chr16, 1),
    }
    all_kmers = set().union(*(counts for counts in group_counts.values()))
    common = set.intersection(
        *(
            {kmer for kmer in all_kmers if counts[kmer] >= args.min_per_label_kmer}
            for counts in group_counts.values()
        )
    )
    if len(common) < 700:
        raise RuntimeError(
            f"Unexpectedly few common 7mers at {args.min_per_label_kmer}x: {len(common)}"
        )
    chr19_common = cap_per_label_kmer(filter_common(chr19, common), args.min_per_label_kmer)
    chr16_common = cap_per_label_kmer(filter_common(chr16, common), args.min_per_label_kmer)
    pq.write_table(chr19_common, out / "chr19_common.parquet", compression="zstd", compression_level=6, use_dictionary=True)
    pq.write_table(chr16_common, out / "chr16_common.parquet", compression="zstd", compression_level=6, use_dictionary=True)

    chr19_meta = metadata_table(chr19_common, "chr19")
    chr16_meta = metadata_table(chr16_common, "chr16")
    pq.write_table(chr19_meta, out / "chr19_split.parquet", compression="zstd", compression_level=6, use_dictionary=True)
    pq.write_table(chr16_meta, out / "chr16_split.parquet", compression="zstd", compression_level=6, use_dictionary=True)

    chr16_data = chr16_meta.to_pydict()
    role = np.asarray(chr16_data["role"], dtype=object)
    if set(np.asarray(chr16_data["fast5_raw_read_id"])[role == "baseline"]) & set(np.asarray(chr16_data["fast5_raw_read_id"])[role == "test"]):
        raise RuntimeError("chr16 baseline and test reads overlap")
    baseline_counts = Counter(
        kmer for kmer, value in zip(chr16_data["ref_7mer"], chr16_data["role"]) if value == "baseline"
    )
    test_counts = Counter(
        kmer for kmer, value in zip(chr16_data["ref_7mer"], chr16_data["role"]) if value == "test"
    )
    if min(baseline_counts.values()) < 30 or min(test_counts.values()) < 30:
        raise RuntimeError("Read-level split left an unexpectedly sparse 7mer group")

    summary = {
        "min_per_label_kmer": args.min_per_label_kmer,
        "max_position_embeddings": args.max_position_embeddings,
        "common_kmers": len(common),
        "incompatible_rows_filtered": {"chr19": chr19_incompatible, "chr16": chr16_incompatible},
        "common_7mers_sorted": sorted(common),
        "group_counts_before_filter": {
            name: {
                "kmers": len(counts),
                "rows": int(sum(counts.values())),
                "per_kmer_min_max": [min(counts.values()), max(counts.values())] if counts else [0, 0],
            }
            for name, counts in group_counts.items()
        },
        "chr19": count_summary(chr19_meta),
        "chr16": count_summary(chr16_meta),
        "chr16_baseline_negative_windows": int(sum(value == "baseline" for value in chr16_data["role"])),
        "chr16_test_windows": int(sum(value == "test" for value in chr16_data["role"])),
        "chr16_excluded_calibration_read_windows": int(sum(value == "excluded_calibration_read" for value in chr16_data["role"])),
        "chr16_baseline_per_kmer_min_max": [min(baseline_counts.values()), max(baseline_counts.values())],
        "chr16_test_per_kmer_min_max": [min(test_counts.values()), max(test_counts.values())],
        "split": "chr19 read-hash 5-fold CV; chr16 read-hash 50:50 baseline/test with calibration-read positives excluded",
        "seed_definition": "BLAKE2b digest_size=8; S18-specific hash persons; chr16 calibration iff unsigned digest < 2^63",
    }
    (out / "s18_split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "common_kmers": len(common),
        "chr19_rows": chr19_common.num_rows,
        "chr16_rows": chr16_common.num_rows,
        "chr16_baseline_negative": summary["chr16_baseline_negative_windows"],
        "chr16_test": summary["chr16_test_windows"],
        "chr16_excluded": summary["chr16_excluded_calibration_read_windows"],
        "baseline_per_kmer_min_max": summary["chr16_baseline_per_kmer_min_max"],
        "test_per_kmer_min_max": summary["chr16_test_per_kmer_min_max"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
