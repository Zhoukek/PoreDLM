#!/usr/bin/env python3
"""Stream aligned calibrated signals and build S18 Apple/Stone 7-mer candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

S15_DIR = Path(__file__).resolve().parents[1] / "s15"
if str(S15_DIR) not in sys.path:
    sys.path.insert(0, str(S15_DIR))

from s15_common import (
    POREGPT_ROOT,
    BedSite,
    contiguous_window,
    cut_window,
    dump_json,
    json_line_id,
    load_feature_extractor,
    normalize_complete_signal,
    parse_bed,
    stable_u64,
    stone_normalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--bed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-dir", default="")
    parser.add_argument("--chrom", default="chr19")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--mapq-min", type=int, default=30)
    parser.add_argument("--min-bed-coverage", type=int, default=10)
    parser.add_argument("--flush-rows", type=int, default=512)
    parser.add_argument("--max-sites-per-class", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--apple-encoder",
        default=str(POREGPT_ROOT / "models/HF_RSQ742C12A511_DNAOLMO_V600/encoder"),
    )
    parser.add_argument(
        "--apple-reference-encoder",
        default=str(POREGPT_ROOT / "models/HF_RSQ741C12V523_MIXOLMO_V610/encoder"),
    )
    return parser.parse_args()


def limit_sites(candidates: dict[int, BedSite], per_class: int) -> dict[int, BedSite]:
    if not per_class:
        return candidates
    out: dict[int, BedSite] = {}
    for label in (0, 1):
        rows = sorted(
            (site for site in candidates.values() if site.label == label),
            key=lambda site: stable_u64("s18-candidate", label, site.pos0),
        )[:per_class]
        out.update({site.pos0: site for site in rows})
    return out


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", compression_level=6, use_dictionary=True)


def main() -> int:
    args = parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        raise ValueError("worker-index must be in [0, workers)")

    out = Path(args.out_dir)
    shard_dir = Path(args.candidate_dir) if args.candidate_dir else out / "candidate_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    candidates, nonzero = parse_bed(Path(args.bed), chrom=args.chrom, min_coverage=args.min_bed_coverage)
    all_candidate_count = len(candidates)
    candidates = limit_sites(candidates, args.max_sites_per_class)

    apple_dir = Path(args.apple_encoder)
    apple_reference_dir = Path(args.apple_reference_encoder)
    apple = load_feature_extractor(apple_dir)
    apple_reference = load_feature_extractor(apple_reference_dir)
    if str(getattr(apple, "strategy", "")) != "apple" or str(getattr(apple_reference, "strategy", "")) != "apple":
        raise RuntimeError("V600 and V610 feature extractors must both use strategy='apple'")

    source = Path(args.jsonl)
    total_bytes = source.stat().st_size
    shard_start = total_bytes * args.worker_index // args.workers
    shard_end = total_bytes * (args.worker_index + 1) // args.workers
    state_path = out / f"extract_{args.chrom}_worker_{args.worker_index:02d}.state.json"
    batch: list[dict[str, object]] = []
    rows_seen = reads_with_windows = windows = 0
    resume_offset = shard_start
    if args.resume and state_path.is_file():
        state = json.loads(state_path.read_text())
        resume_offset = int(state.get("next_offset", shard_start))
        rows_seen = int(state.get("records_seen", 0))
        reads_with_windows = int(state.get("candidate_reads", 0))
        windows = int(state.get("candidate_windows", 0))
    part = len(list(shard_dir.glob(f"candidate_w{args.worker_index:02d}_*.parquet")))
    apple_agreement_max = 0.0
    started = time.time()

    with source.open("rb", buffering=64 * 1024 * 1024) as handle:
        handle.seek(resume_offset)
        if resume_offset == shard_start and shard_start:
            handle.seek(shard_start)
            handle.readline()
        while True:
            line_offset = handle.tell()
            if line_offset >= shard_end:
                break
            line = handle.readline()
            if not line:
                break
            rows_seen += 1
            if args.max_records and rows_seen > args.max_records:
                break
            read_id = json_line_id(line)
            if read_id is None:
                raise RuntimeError(f"Cannot parse read_id at line {rows_seen}")
            obj = json.loads(line)
            if obj.get("strand") != "+" or int(obj.get("mapq", -1)) < args.mapq_min:
                continue

            covered = [
                candidates[pos]
                for pos in {int(row["genome_pos0"]) for row in obj["per_base"]}
                if pos in candidates
            ]
            if not covered:
                continue
            valid = []
            for site in covered:
                result = contiguous_window(obj, site, nonzero, args.mapq_min, chrom=args.chrom)
                if result is not None:
                    valid.append((site, *result))
            if not valid:
                continue

            raw = np.asarray(obj["signal"], dtype=np.float32)
            apple_full = normalize_complete_signal(apple, raw)
            reference_full = normalize_complete_signal(apple_reference, raw)
            difference = float(np.max(np.abs(apple_full - reference_full)))
            apple_agreement_max = max(apple_agreement_max, difference)
            if apple_agreement_max > 1e-5:
                raise RuntimeError(
                    "V600 and V610 Apple preprocessors disagree; refusing to mix Apple inputs "
                    f"(max_abs_difference={apple_agreement_max})"
                )
            stone_full = stone_normalize(raw)
            reads_with_windows += 1
            raw_read_id = str(obj.get("fast5_raw_read_id", read_id))
            for site, temporal, ref, query in valid:
                apple_window = cut_window(apple_full, temporal)
                stone_window = cut_window(stone_full, temporal)
                dwell = [int(row["dwell"]) for row in temporal]
                batch.append({
                    "window_id": f"{site.pos0}:{raw_read_id}",
                    "site_pos0": site.pos0,
                    "label": site.label,
                    "bed_coverage": site.bed_coverage,
                    "bed_percent": site.mod_percent,
                    "block_id": site.block_id,
                    "cov_bin": site.cov_bin,
                    "ref_7mer": ref,
                    "query_7mer": query,
                    "read_id": read_id,
                    "fast5_raw_read_id": raw_read_id,
                    "source_part": str(obj.get("source_part", "")),
                    "mapq": int(obj["mapq"]),
                    "dwell": dwell,
                    "signal_apple": apple_window.tolist(),
                    "signal_stone": stone_window.tolist(),
                    "signal_len": int(apple_window.size),
                    "priority": stable_u64("s18-read", site.pos0, raw_read_id),
                })
                windows += 1

            if len(batch) >= args.flush_rows:
                write_rows(batch, shard_dir / f"candidate_w{args.worker_index:02d}_{part:06d}.parquet")
                part += 1
                batch.clear()
                dump_json(state_path, {
                    "worker_index": args.worker_index,
                    "workers": args.workers,
                    "next_offset": handle.tell(),
                    "records_seen": rows_seen,
                    "candidate_reads": reads_with_windows,
                    "candidate_windows": windows,
                    "complete": False,
                })
            if rows_seen % 1000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(json.dumps({
                    "stage": "extract",
                    "chrom": args.chrom,
                    "records": rows_seen,
                    "windows": windows,
                    "candidate_reads": reads_with_windows,
                    "progress": round((handle.tell() - shard_start) / max(shard_end - shard_start, 1), 5),
                    "worker": args.worker_index,
                    "mb_per_s": round(handle.tell() / elapsed / 1e6, 2),
                }), flush=True)

    if batch:
        write_rows(batch, shard_dir / f"candidate_w{args.worker_index:02d}_{part:06d}.parquet")
        part += 1

    summary = {
        "source_jsonl": str(source),
        "bed": str(Path(args.bed)),
        "input_normalization": "fast5_calibrated",
        "signal_base_shift": -4,
        "chrom": args.chrom,
        "window_nt": 7,
        "window_definition": "[site-3, site+4), NNNCGNN",
        "site_strand": "+",
        "read_strand": "+",
        "mapq_min": args.mapq_min,
        "min_bed_coverage": args.min_bed_coverage,
        "all_eligible_bed_sites": all_candidate_count,
        "selected_bed_sites": len(candidates),
        "records_seen": rows_seen,
        "candidate_reads": reads_with_windows,
        "candidate_windows": windows,
        "shards": part,
        "worker_index": args.worker_index,
        "workers": args.workers,
        "byte_range": [shard_start, shard_end],
        "apple_encoder": str(apple_dir),
        "apple_reference_encoder": str(apple_reference_dir),
        "apple_v600_v610_max_abs_difference": apple_agreement_max,
        "normalization_scope": "complete calibrated JSONL signal before 7-nt window slicing",
        "stone_implementation": "PoreDLM median/MAD strategy used by the S17/S18 V003 branch",
        "seconds": time.time() - started,
    }
    dump_json(out / f"extract_{args.chrom}_worker_{args.worker_index:02d}.summary.json", summary)
    dump_json(state_path, {
        "worker_index": args.worker_index,
        "workers": args.workers,
        "next_offset": shard_end,
        "records_seen": rows_seen,
        "candidate_reads": reads_with_windows,
        "candidate_windows": windows,
        "complete": True,
    })
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
