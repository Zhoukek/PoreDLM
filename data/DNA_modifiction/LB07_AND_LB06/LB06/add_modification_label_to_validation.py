#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_INPUT = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/"
    "validation_fullapple_token1600.jsonl.gz"
)
DEFAULT_OUTPUT = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/"
    "LB07_AND_LB06/LB06/stage2_fullapple_token1600/validation/"
    "validation_fullapple_token1600_modlabel.jsonl.gz"
)
DEFAULT_MODIFIED_BASE_POSITIONS = "14,33,52,71,90,109,128"


def parse_positions(value: str) -> list[int]:
    positions = [int(part.strip()) for part in value.split(",") if part.strip()]
    if any(pos <= 0 for pos in positions):
        raise ValueError(f"Base positions are 1-based and must be positive, got {positions}")
    return positions


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def count_jsonl_records(path: Path) -> int:
    with open_text(path, "rt") as handle:
        return sum(1 for line in handle if line.strip())


def base_span_to_token_span(
    start_sample: int,
    end_sample: int,
    *,
    samples_per_token: int,
    token_count: int,
) -> tuple[int, int]:
    token_start = max(0, int(start_sample) // samples_per_token)
    token_end = min(token_count, int(math.ceil(float(end_sample) / samples_per_token)))
    token_end = max(token_start, token_end)
    return token_start, token_end


def add_label_to_item(
    item: dict,
    *,
    modified_base_positions: list[int],
    samples_per_token: int,
) -> tuple[dict, dict]:
    meta = item.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"item id={item.get('id')!r} has non-dict meta.")

    spans = meta.get("base_sample_spans_rel")
    if not isinstance(spans, list):
        raise ValueError(f"item id={item.get('id')!r} has no meta.base_sample_spans_rel list.")

    original_token_len = int(meta.get("original_token_len", 0))
    if original_token_len <= 0:
        raise ValueError(f"item id={item.get('id')!r} has invalid original_token_len={original_token_len!r}.")

    labels = [0] * original_token_len
    modification_token_spans: list[list[int]] = []
    missing_base_positions: list[int] = []

    for base_position in modified_base_positions:
        base_index = base_position - 1
        if base_index < 0 or base_index >= len(spans):
            missing_base_positions.append(base_position)
            continue

        span = spans[base_index]
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(
                f"item id={item.get('id')!r} has invalid span at base position "
                f"{base_position}: {span!r}"
            )
        token_start, token_end = base_span_to_token_span(
            int(span[0]),
            int(span[1]),
            samples_per_token=samples_per_token,
            token_count=original_token_len,
        )
        if token_end <= token_start:
            continue
        for token_index in range(token_start, token_end):
            labels[token_index] = 1
        modification_token_spans.append([base_position, token_start, token_end])

    meta["modification_label"] = labels
    meta["modification_base_positions_1based"] = modified_base_positions
    meta["modification_samples_per_token"] = samples_per_token
    meta["modification_token_spans"] = modification_token_spans
    if missing_base_positions:
        meta["modification_missing_base_positions_1based"] = missing_base_positions

    stats = {
        "label_ones": int(sum(labels)),
        "missing_base_count": len(missing_base_positions),
    }
    return item, stats


def process_jsonl(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    modified_base_positions: list[int],
    samples_per_token: int,
    count_total_first: bool,
) -> dict:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = count_jsonl_records(input_jsonl) if count_total_first else None

    stats = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "modified_base_positions_1based": modified_base_positions,
        "samples_per_token": samples_per_token,
        "total_records": 0,
        "written_records": 0,
        "failed_records": 0,
        "total_label_ones": 0,
        "records_with_missing_base_positions": 0,
    }

    start_time = time.time()
    with open_text(input_jsonl, "rt") as fin, open_text(output_jsonl, "wt") as fout:
        iterator = fin
        if tqdm is not None:
            iterator = tqdm(fin, total=total, desc="Adding modification_label", ncols=120)

        for line_number, line in enumerate(iterator, start=1):
            if not line.strip():
                continue
            stats["total_records"] += 1
            try:
                item = json.loads(line)
                item, item_stats = add_label_to_item(
                    item,
                    modified_base_positions=modified_base_positions,
                    samples_per_token=samples_per_token,
                )
            except Exception as exc:
                stats["failed_records"] += 1
                raise type(exc)(f"{input_jsonl} line {line_number}: {exc}") from exc

            stats["total_label_ones"] += int(item_stats["label_ones"])
            if item_stats["missing_base_count"] > 0:
                stats["records_with_missing_base_positions"] += 1
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats["written_records"] += 1

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["records_per_second"] = stats["written_records"] / max(elapsed, 1e-6)

    stats_path = output_jsonl.with_name(output_jsonl.name + ".stats.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")
    print(f"Stats: {stats_path}")
    print(f"Written records: {stats['written_records']}")
    print(f"Total positive token labels: {stats['total_label_ones']}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add meta.modification_label to LB06 fixed-token jsonl.gz based on "
            "meta.base_sample_spans_rel and known modified base positions."
        )
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT)
    parser.add_argument("--modified-base-positions", default=DEFAULT_MODIFIED_BASE_POSITIONS)
    parser.add_argument("--samples-per-token", type=int, default=5)
    parser.add_argument("--no-count-total-first", action="store_true")
    args = parser.parse_args()

    if args.samples_per_token <= 0:
        raise ValueError("--samples-per-token must be positive.")

    process_jsonl(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        modified_base_positions=parse_positions(args.modified_base_positions),
        samples_per_token=args.samples_per_token,
        count_total_first=not args.no_count_total_first,
    )


if __name__ == "__main__":
    main()
