#!/usr/bin/env python3
"""Extract selected 10-nt Apple-normalized signal windows from large corpus JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


READ_ID_RE = re.compile(br'"read_id"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--mod-jsonl", required=True)
    parser.add_argument("--unmod-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_targets(path: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    targets: dict[str, dict[str, list[dict[str, str]]]] = {"MOD": {}, "UNMOD": {}}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            targets[row["dataset"]].setdefault(row["read_id"], []).append(row)
    return targets


def extract_one_corpus(
    dataset: str,
    jsonl_path: str,
    target_map: dict[str, list[dict[str, str]]],
    output_path: str,
) -> dict[str, Any]:
    source = Path(jsonl_path)
    total_bytes = source.stat().st_size
    found_reads: set[str] = set()
    records: list[dict[str, Any]] = []
    started = time.time()

    with source.open("rb", buffering=64 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            match = READ_ID_RE.search(line[:512])
            if match is None:
                raise ValueError(f"Could not parse read_id at {source}:{line_number}")
            read_id = match.group(1).decode("utf-8")
            target_windows = target_map.get(read_id)
            if target_windows:
                obj = json.loads(line)
                if obj.get("normal_mode") != "apple":
                    raise ValueError(f"{dataset}/{read_id} is not marked apple-normalized")
                if int(obj.get("signal_base_shift")) != -4:
                    raise ValueError(f"{dataset}/{read_id} has shift={obj.get('signal_base_shift')}, expected -4")
                signal = obj["signal"]
                by_pos = {int(base["genome_pos0"]): base for base in obj["per_base"]}
                for target in target_windows:
                    start = int(target["window_start0"])
                    end = int(target["window_end0"])
                    genomic_items = [by_pos.get(pos) for pos in range(start, end)]
                    if any(item is None for item in genomic_items):
                        raise ValueError(
                            f"{dataset}/{read_id}/{target['site_id']} lacks a per_base item in {start}:{end}"
                        )
                    genomic_items = [item for item in genomic_items if item is not None]
                    signal_items = sorted(genomic_items, key=lambda item: int(item["query_pos_signal"]))
                    window_signal: list[float] = []
                    spans: list[list[int]] = []
                    for item in signal_items:
                        span_start, span_end = (int(value) for value in item["signal_span"])
                        if span_end <= span_start:
                            raise ValueError(f"Non-positive signal span in {dataset}/{read_id}")
                        spans.append([span_start, span_end])
                        window_signal.extend(signal[span_start:span_end])
                    records.append(
                        {
                            "site_id": target["site_id"],
                            "dataset": dataset,
                            "label": 1 if dataset == "MOD" else 0,
                            "read_id": read_id,
                            "chrom": obj["chrom"],
                            "site_pos0": int(target["site_pos0"]),
                            "window_start0": start,
                            "window_end0": end,
                            "strand": obj["strand"],
                            "mapq": int(obj["mapq"]),
                            "normal_mode": obj["normal_mode"],
                            "signal_base_shift": int(obj["signal_base_shift"]),
                            "window_ref_seq_genomic": "".join(str(item["ref_base"]) for item in genomic_items),
                            "window_query_seq_genomic": "".join(str(item["query_base"]) for item in genomic_items),
                            "signal_order_genome_pos0": [int(item["genome_pos0"]) for item in signal_items],
                            "query_pos_signal": [int(item["query_pos_signal"]) for item in signal_items],
                            "source_signal_spans": spans,
                            "per_base_dwell_signal_order": [int(item["dwell"]) for item in signal_items],
                            "signal_len": len(window_signal),
                            "signal": window_signal,
                        }
                    )
                found_reads.add(read_id)
            if line_number % 5000 == 0:
                elapsed = max(time.time() - started, 1e-6)
                consumed = handle.tell()
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "progress": round(consumed / total_bytes, 4),
                            "records_seen": line_number,
                            "target_reads_found": len(found_reads),
                            "mb_per_s": round(consumed / elapsed / 1e6, 1),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    missing = sorted(set(target_map) - found_reads)
    if missing:
        raise RuntimeError(f"{dataset}: {len(missing)} target reads were absent, first={missing[:3]}")
    records.sort(key=lambda row: (row["site_id"], row["dataset"], row["read_id"]))
    with Path(output_path).open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {
        "dataset": dataset,
        "source": str(source),
        "source_bytes": total_bytes,
        "target_reads": len(target_map),
        "found_reads": len(found_reads),
        "window_records": len(records),
        "seconds": round(time.time() - started, 3),
        "output": output_path,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(Path(args.targets))
    jobs = [
        ("MOD", args.mod_jsonl, targets["MOD"], str(out_dir / ".mod_signal_windows.jsonl")),
        ("UNMOD", args.unmod_jsonl, targets["UNMOD"], str(out_dir / ".unmod_signal_windows.jsonl")),
    ]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(extract_one_corpus, *job) for job in jobs]
        summaries = [future.result() for future in futures]

    records: list[dict[str, Any]] = []
    for _, _, _, output_path in jobs:
        with Path(output_path).open() as handle:
            records.extend(json.loads(line) for line in handle)
        os.remove(output_path)
    records.sort(key=lambda row: (row["site_id"], row["dataset"], row["read_id"]))

    jsonl_out = out_dir / "signal_windows.10nt.jsonl"
    with jsonl_out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    metadata_out = out_dir / "signal_windows.10nt.metadata.tsv"
    metadata_fields = [
        "site_id", "dataset", "label", "read_id", "chrom", "site_pos0", "window_start0",
        "window_end0", "strand", "mapq", "normal_mode", "signal_base_shift",
        "window_ref_seq_genomic", "window_query_seq_genomic", "signal_len",
    ]
    with metadata_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=metadata_fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in metadata_fields})

    summary = {
        "processes": summaries,
        "records": len(records),
        "all_apple_normalized": all(record["normal_mode"] == "apple" for record in records),
        "all_shift_minus4": all(record["signal_base_shift"] == -4 for record in records),
        "signal_len_min": min(record["signal_len"] for record in records),
        "signal_len_max": max(record["signal_len"] for record in records),
        "signal_len_mean": sum(record["signal_len"] for record in records) / len(records),
        "orientation": "signal concatenated in ascending query_pos_signal (sequencing/signal order)",
    }
    (out_dir / "signal_window_extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
