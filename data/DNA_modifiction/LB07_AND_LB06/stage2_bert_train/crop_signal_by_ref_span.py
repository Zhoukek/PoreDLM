from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with open_text(path, "rt") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield line_no, obj


def is_null_span(span: Any) -> bool:
    return (
        isinstance(span, list)
        and len(span) == 2
        and span[0] is None
        and span[1] is None
    )


def normalize_non_null_span(span: Any, *, line_no: int, index: int) -> tuple[int, int] | None:
    if is_null_span(span):
        return None
    if not isinstance(span, list) or len(span) != 2:
        raise ValueError(f"line {line_no}: base_sample_span_ref[{index}] must be [start, end] or [null, null]")

    start, end = span
    if start is None or end is None:
        raise ValueError(f"line {line_no}: base_sample_span_ref[{index}] has only one null side: {span}")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"line {line_no}: base_sample_span_ref[{index}] must use integer positions: {span}")
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"line {line_no}: invalid base_sample_span_ref[{index}] range: {span}")
    return start, end


def crop_one_record(obj: dict[str, Any], *, line_no: int) -> tuple[dict[str, Any], int]:
    signal = obj.get("signal")
    spans = obj.get("base_sample_span_ref")

    if not isinstance(signal, list):
        raise ValueError(f"line {line_no}: signal must be a list")
    if not isinstance(spans, list):
        raise ValueError(f"line {line_no}: base_sample_span_ref must be a list")

    non_null_spans: list[tuple[int, int]] = []
    for index, span in enumerate(spans):
        normalized = normalize_non_null_span(span, line_no=line_no, index=index)
        if normalized is not None:
            non_null_spans.append(normalized)

    if not non_null_spans:
        raise ValueError(f"line {line_no}: base_sample_span_ref has no non-null signal ranges")

    crop_start = min(start for start, _ in non_null_spans)
    crop_end = max(end for _, end in non_null_spans)
    if crop_end > len(signal):
        raise ValueError(
            f"line {line_no}: max span end {crop_end} exceeds signal length {len(signal)}"
        )

    new_spans: list[list[int | None]] = []
    for span in spans:
        normalized = normalize_non_null_span(span, line_no=line_no, index=len(new_spans))
        if normalized is None:
            new_spans.append([None, None])
            continue
        start, end = normalized
        new_spans.append([start - crop_start, end - crop_start])

    cropped_signal = signal[crop_start:crop_end]
    saved_signal_len = len(cropped_signal)

    out_obj = dict(obj)
    out_obj["signal"] = cropped_signal
    out_obj["base_sample_span_ref"] = new_spans
    out_obj["signal_len"] = saved_signal_len
    out_obj["signal_crop_start"] = crop_start
    out_obj["signal_crop_end"] = crop_end
    return out_obj, saved_signal_len


def process_jsonl(input_jsonl: Path, output_jsonl: Path, stats_json: Path | None) -> dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if stats_json is not None:
        stats_json.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "total_records": 0,
        "written_records": 0,
        "skipped_records": 0,
        "skip_reasons": {},
        "max_saved_signal_len": 0,
        "max_saved_signal_read_id": None,
        "min_saved_signal_len": None,
    }

    with open_text(output_jsonl, "wt") as fout:
        for line_no, obj in iter_jsonl(input_jsonl):
            stats["total_records"] += 1
            try:
                out_obj, saved_signal_len = crop_one_record(obj, line_no=line_no)
            except ValueError as exc:
                stats["skipped_records"] += 1
                reason = str(exc).split(": ", 1)[-1]
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                continue

            fout.write(json.dumps(out_obj, ensure_ascii=False, separators=(",", ":")) + "\n")

            stats["written_records"] += 1
            if stats["min_saved_signal_len"] is None:
                stats["min_saved_signal_len"] = saved_signal_len
            else:
                stats["min_saved_signal_len"] = min(stats["min_saved_signal_len"], saved_signal_len)
            if saved_signal_len > stats["max_saved_signal_len"]:
                stats["max_saved_signal_len"] = saved_signal_len
                stats["max_saved_signal_read_id"] = out_obj.get("read_id")

    if stats_json is not None:
        with stats_json.open("w", encoding="utf-8") as fout:
            json.dump(stats, fout, ensure_ascii=False, indent=2)
            fout.write("\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop each record's signal to the continuous range covered by "
            "base_sample_span_ref, then rebase non-null spans from zero."
        )
    )
    parser.add_argument("--input-jsonl", required=True, type=Path, help="Input .jsonl or .jsonl.gz file")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output .jsonl or .jsonl.gz file")
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional stats JSON path. Defaults to <output-jsonl>.stats.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_json = args.stats_json
    if stats_json is None:
        output_name = args.output_jsonl.name
        if output_name.endswith(".jsonl.gz"):
            stats_json = args.output_jsonl.with_name(output_name[: -len(".jsonl.gz")] + ".stats.json")
        elif output_name.endswith(".jsonl"):
            stats_json = args.output_jsonl.with_suffix(".stats.json")
        else:
            stats_json = args.output_jsonl.with_name(output_name + ".stats.json")

    stats = process_jsonl(args.input_jsonl, args.output_jsonl, stats_json)
    print(f"written_records: {stats['written_records']}")
    print(f"skipped_records: {stats['skipped_records']}")
    print(f"max_saved_signal_len: {stats['max_saved_signal_len']}")
    print(f"max_saved_signal_read_id: {stats['max_saved_signal_read_id']}")
    print(f"stats_json: {stats_json}")


if __name__ == "__main__":
    main()
