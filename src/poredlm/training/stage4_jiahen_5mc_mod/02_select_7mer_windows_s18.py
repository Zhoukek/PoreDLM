#!/usr/bin/env python3
"""Select a compact, deterministic 7-mer-balanced S16 tokenization cohort."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-per-label-kmer", type=int, default=100)
    parser.add_argument(
        "--reserve-per-label-kmer",
        type=int,
        default=0,
        help="Retain this many extra candidates per label/7mer for downstream compatibility filtering",
    )
    parser.add_argument("--prefix", default="s16")
    parser.add_argument("--chrom", default="chr19")
    return parser.parse_args()


def canonical_kmers() -> list[str]:
    return [
        "".join(left) + "CG" + "".join(right)
        for left in itertools.product("ACGT", repeat=3)
        for right in itertools.product("ACGT", repeat=2)
    ]


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{args.prefix}_selected_windows.parquet"
    coverage_path = out / f"{args.prefix}_7mer_coverage.tsv"
    summary_path = out / f"{args.prefix}_selection_summary.json"
    if manifest_path.exists() or coverage_path.exists() or summary_path.exists():
        raise FileExistsError("S16 selection outputs already exist; refusing to overwrite them")

    kmers = canonical_kmers()
    target = args.target_per_label_kmer
    if target < 1:
        raise ValueError("target-per-label-kmer must be positive")
    if args.reserve_per_label_kmer < 0:
        raise ValueError("reserve-per-label-kmer must be non-negative")
    selection_target = target + args.reserve_per_label_kmer
    heaps: dict[tuple[int, str], list[tuple[int, str, int, dict[str, object]]]] = {
        (label, kmer): [] for label in (0, 1) for kmer in kmers
    }
    selected: dict[tuple[int, str], dict[str, tuple[int, str, int, dict[str, object]]]] = {
        (label, kmer): {} for label in (0, 1) for kmer in kmers
    }
    availability = {(label, kmer): 0 for label in (0, 1) for kmer in kmers}
    files = sorted(candidate_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No candidate parquet shards found in {candidate_dir}")

    columns = [
        "window_id", "site_pos0", "label", "bed_coverage", "bed_percent", "block_id",
        "cov_bin", "ref_7mer", "read_id", "fast5_raw_read_id", "source_part", "mapq",
        "signal_len", "priority",
    ]
    scanned_rows = 0
    tie_breaker = 0
    for file_index, path in enumerate(files, 1):
        table = pq.ParquetFile(path).read(columns=columns)
        data = table.to_pydict()
        scanned_rows += len(data["window_id"])
        for index, window_id in enumerate(data["window_id"]):
            label = int(data["label"][index])
            kmer = str(data["ref_7mer"][index])
            key = (label, kmer)
            if key not in heaps:
                continue
            availability[key] += 1
            row = {name: data[name][index] for name in columns}
            priority = int(row["priority"])
            tie_breaker += 1
            # Max-priority item stays at heap root via negative priority.
            item = (-priority, str(window_id), tie_breaker, row)
            heap = heaps[key]
            chosen = selected[key]
            existing = chosen.get(str(window_id))
            if existing is not None:
                if item[:2] > existing[:2]:
                    chosen[str(window_id)] = item
                    heapq.heappush(heap, item)
                continue
            if len(chosen) < selection_target:
                chosen[str(window_id)] = item
                heapq.heappush(heap, item)
                continue
            while heap and chosen.get(heap[0][1]) is not heap[0]:
                heapq.heappop(heap)
            if heap and item[:2] > heap[0][:2]:
                worst = heapq.heappop(heap)
                del chosen[worst[1]]
                chosen[str(window_id)] = item
                heapq.heappush(heap, item)
        if file_index % 500 == 0 or file_index == len(files):
            print(json.dumps({"stage": "select", "shards": file_index, "total_shards": len(files), "rows": scanned_rows}), flush=True)

    selected_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for label in (0, 1):
        for kmer in kmers:
            key = (label, kmer)
            rows = [entry[3] for entry in sorted(selected[key].values(), key=lambda entry: (-entry[0], entry[1], entry[2]))]
            selected_rows.extend(rows)
            coverage_rows.append({
                "label": label,
                "bed_percent": 100 if label else 0,
                "ref_7mer": kmer,
                "available_candidate_windows": availability[key],
                "selected_windows": len(rows),
                "target_windows": target,
                "selection_target_windows": selection_target,
                "target_met": len(rows) >= target,
                "selected_unique_sites": len({int(row["site_pos0"]) for row in rows}),
                "selected_unique_raw_reads": len({str(row["fast5_raw_read_id"]) for row in rows}),
            })
    selected_rows.sort(key=lambda row: (int(row["label"]), str(row["ref_7mer"]), int(row["priority"]), str(row["window_id"])))
    pq.write_table(pa.Table.from_pylist(selected_rows), manifest_path, compression="zstd", compression_level=6, use_dictionary=True)
    with coverage_path.open("w") as handle:
        handle.write("label\tbed_percent\tref_7mer\tavailable_candidate_windows\tselected_windows\ttarget_windows\tselection_target_windows\ttarget_met\tselected_unique_sites\tselected_unique_raw_reads\n")
        for row in coverage_rows:
            handle.write("\t".join(map(str, row.values())) + "\n")

    summary = {
        "definition": f"positive-strand {args.chrom} NNNCGNN windows from candidate shards; label 0=BED 0%, label 1=BED 100%",
        "target_windows_per_label_7mer": target,
        "reserve_windows_per_label_7mer": args.reserve_per_label_kmer,
        "selection_target_windows_per_label_7mer": selection_target,
        "theoretical_windows": 2 * len(kmers) * target,
        "selected_windows": len(selected_rows),
        "label_0_selected_windows": sum(row["selected_windows"] for row in coverage_rows if row["label"] == 0),
        "label_1_selected_windows": sum(row["selected_windows"] for row in coverage_rows if row["label"] == 1),
        "label_0_kmers_present": sum(row["available_candidate_windows"] > 0 for row in coverage_rows if row["label"] == 0),
        "label_1_kmers_present": sum(row["available_candidate_windows"] > 0 for row in coverage_rows if row["label"] == 1),
        "label_0_kmers_at_target": sum(row["target_met"] for row in coverage_rows if row["label"] == 0),
        "label_1_kmers_at_target": sum(row["target_met"] for row in coverage_rows if row["label"] == 1),
        "candidate_shards": len(files),
        "candidate_rows_scanned": scanned_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
