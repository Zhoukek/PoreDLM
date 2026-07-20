#!/usr/bin/env python3
"""Select fully modified chr19 sites with complete 10-nt signal-window coverage."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
QUERY_CONSUMING = frozenset("MIS=X")
MATCHING = frozenset("M=X")


@dataclass(frozen=True)
class Alignment:
    dataset: str
    read_id: str
    covered_start: int
    covered_end: int
    strand: str
    mapq: int
    cigar: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", required=True)
    parser.add_argument("--mod-alignments", required=True)
    parser.add_argument("--unmod-alignments", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--window-nt", type=int, default=10)
    parser.add_argument("--shift", type=int, default=-4)
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--min-site-distance", type=int, default=1000)
    return parser.parse_args()


def load_bed_100_percent(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, str]] = []
    with opener(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11 or fields[0] != "chr19" or float(fields[10]) != 100.0:
                continue
            rows.append(
                {
                    "chrom": fields[0],
                    "start0": fields[1],
                    "end0": fields[2],
                    "bed_name": fields[3],
                    "bed_score": fields[4],
                    "bed_strand": fields[5],
                    "bed_coverage": fields[9],
                    "percent_modified": fields[10],
                }
            )
    rows.sort(key=lambda row: int(row["start0"]))
    positions = np.fromiter((int(row["start0"]) for row in rows), dtype=np.int64)
    if positions.size == 0:
        raise RuntimeError("No chr19 BED sites with percent_modified=100 were found")
    return positions, rows


def load_alignments(path: Path) -> list[Alignment]:
    records: list[Alignment] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            records.append(
                Alignment(
                    dataset=row["dataset"],
                    read_id=row["read_id"],
                    covered_start=int(row["genome_start0"]),
                    covered_end=int(row["genome_end0"]),
                    strand=row["strand"],
                    mapq=int(row["mapq"]),
                    cigar=row["cigar"],
                )
            )
    return records


def valid_match_blocks(aln: Alignment, shift: int) -> list[tuple[int, int]]:
    ops = [(int(length), op) for length, op in CIGAR_RE.findall(aln.cigar)]
    if not ops or "".join(f"{n}{op}" for n, op in ops) != aln.cigar:
        raise ValueError(f"Malformed CIGAR for {aln.read_id}: {aln.cigar}")
    query_len = sum(length for length, op in ops if op in QUERY_CONSUMING)
    shift_abs = abs(shift)
    if shift != -shift_abs:
        raise ValueError("This corpus validation currently expects a negative signal-base shift")
    if aln.strand == "+":
        valid_q_start, valid_q_end = 0, query_len - shift_abs
    elif aln.strand == "-":
        valid_q_start, valid_q_end = shift_abs, query_len
    else:
        raise ValueError(f"Unexpected strand {aln.strand!r}")

    query_pos = 0
    ref_offset = 0
    relative: list[tuple[int, int]] = []
    for length, op in ops:
        if op in MATCHING:
            left = max(query_pos, valid_q_start)
            right = min(query_pos + length, valid_q_end)
            if left < right:
                relative.append(
                    (ref_offset + left - query_pos, ref_offset + right - query_pos)
                )
        if op in QUERY_CONSUMING:
            query_pos += length
        if op in "MDN=X":
            ref_offset += length
    if not relative:
        return []

    raw_ref_start = aln.covered_start - relative[0][0]
    blocks = [(raw_ref_start + start, raw_ref_start + end) for start, end in relative]
    if blocks[0][0] != aln.covered_start or blocks[-1][1] != aln.covered_end:
        raise ValueError(
            f"CIGAR/top-level span mismatch for {aln.read_id}: "
            f"derived {blocks[0][0]}-{blocks[-1][1]}, "
            f"recorded {aln.covered_start}-{aln.covered_end}"
        )
    return blocks


def window_depths(
    positions: np.ndarray,
    alignments: list[Alignment],
    left_nt: int,
    right_nt: int,
    shift: int,
) -> np.ndarray:
    difference = np.zeros(positions.size + 1, dtype=np.int32)
    for aln in alignments:
        for start, end in valid_match_blocks(aln, shift):
            # Window [site-left_nt, site+right_nt) must fit wholly in this block.
            site_lo = start + left_nt
            site_hi_exclusive = end - right_nt + 1
            if site_lo >= site_hi_exclusive:
                continue
            lo = int(np.searchsorted(positions, site_lo, side="left"))
            hi = int(np.searchsorted(positions, site_hi_exclusive, side="left"))
            if lo < hi:
                difference[lo] += 1
                difference[hi] -= 1
    return np.cumsum(difference[:-1], dtype=np.int32)


def covering_reads(
    site_positions: set[int],
    alignments: list[Alignment],
    left_nt: int,
    right_nt: int,
    shift: int,
) -> list[dict[str, object]]:
    selected = sorted(site_positions)
    rows: list[dict[str, object]] = []
    for aln in alignments:
        blocks = valid_match_blocks(aln, shift)
        for site in selected:
            window_start = site - left_nt
            window_end = site + right_nt
            if any(start <= window_start and end >= window_end for start, end in blocks):
                rows.append(
                    {
                        "dataset": aln.dataset,
                        "read_id": aln.read_id,
                        "site_pos0": site,
                        "window_start0": window_start,
                        "window_end0": window_end,
                        "strand": aln.strand,
                        "mapq": aln.mapq,
                    }
                )
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.window_nt % 2:
        raise ValueError("window-nt must be even for this analysis")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left_nt = args.window_nt // 2
    right_nt = args.window_nt - left_nt

    positions, bed_rows = load_bed_100_percent(Path(args.bed))
    mod = load_alignments(Path(args.mod_alignments))
    unmod = load_alignments(Path(args.unmod_alignments))
    mod_depth = window_depths(positions, mod, left_nt, right_nt, args.shift)
    unmod_depth = window_depths(positions, unmod, left_nt, right_nt, args.shift)

    eligible = np.flatnonzero((mod_depth >= args.min_depth) & (unmod_depth >= args.min_depth))
    ranked = sorted(
        eligible.tolist(),
        key=lambda idx: (
            -min(int(mod_depth[idx]), int(unmod_depth[idx])),
            -(int(mod_depth[idx]) + int(unmod_depth[idx])),
            int(positions[idx]),
        ),
    )
    chosen: list[int] = []
    for idx in ranked:
        pos = int(positions[idx])
        if all(abs(pos - int(positions[other])) >= args.min_site_distance for other in chosen):
            chosen.append(idx)
        if len(chosen) == args.sites:
            break
    if len(chosen) < args.sites:
        raise RuntimeError(f"Only {len(chosen)} sites passed all criteria")

    site_rows: list[dict[str, object]] = []
    for rank, idx in enumerate(chosen, start=1):
        row: dict[str, object] = dict(bed_rows[idx])
        row.update(
            {
                "site_id": f"S{rank}",
                "window_start0": int(positions[idx]) - left_nt,
                "window_end0": int(positions[idx]) + right_nt,
                "mod_complete_window_depth": int(mod_depth[idx]),
                "unmod_complete_window_depth": int(unmod_depth[idx]),
                "min_depth": min(int(mod_depth[idx]), int(unmod_depth[idx])),
            }
        )
        site_rows.append(row)
    write_tsv(
        out_dir / "selected_5_sites.tsv",
        site_rows,
        [
            "site_id", "chrom", "start0", "end0", "bed_name", "bed_score", "bed_strand",
            "bed_coverage", "percent_modified", "window_start0", "window_end0",
            "mod_complete_window_depth", "unmod_complete_window_depth", "min_depth",
        ],
    )

    selected_positions = {int(row["start0"]) for row in site_rows}
    read_rows = covering_reads(selected_positions, mod, left_nt, right_nt, args.shift)
    read_rows.extend(covering_reads(selected_positions, unmod, left_nt, right_nt, args.shift))
    site_id_by_pos = {int(row["start0"]): str(row["site_id"]) for row in site_rows}
    for row in read_rows:
        row["site_id"] = site_id_by_pos[int(row["site_pos0"])]
    read_rows.sort(key=lambda row: (str(row["site_id"]), str(row["dataset"]), str(row["read_id"])))
    write_tsv(
        out_dir / "target_read_windows.tsv",
        read_rows,
        ["site_id", "dataset", "read_id", "site_pos0", "window_start0", "window_end0", "strand", "mapq"],
    )

    summary = {
        "bed_chr19_100_percent_sites": int(positions.size),
        "eligible_sites_complete_10nt_depth_ge_threshold": int(eligible.size),
        "selected_sites": len(site_rows),
        "min_depth": args.min_depth,
        "window_definition": "[site_pos0-5, site_pos0+5), 10 reference bases",
        "signal_base_shift": args.shift,
        "mod_alignments": len(mod),
        "unmod_alignments": len(unmod),
        "target_read_windows": len(read_rows),
    }
    (out_dir / "site_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
