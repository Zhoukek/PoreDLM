"""PoreKmer coverage-matched comparisons from checksummed score artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .metrics import TABLE_FIELDS, _validated_rows, compare_tables
from .data import current_metadata


def _load_bound_table(
    score_path: str | Path, report: Mapping[str, Any], name: str,
) -> tuple[list[dict[str, Any]], str]:
    tables = report.get("tables")
    if not isinstance(tables, Mapping) or not isinstance(tables.get(name), Mapping):
        raise ValueError(f"score report is missing bound {name} table metadata")
    descriptor = tables[name]
    relative, expected = descriptor.get("path"), descriptor.get("sha256")
    if (
        not isinstance(relative, str) or not relative or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError(f"unsafe {name} table path")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"invalid {name} table SHA-256 digest")
    root = Path(score_path).resolve().parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{name} table path escapes score artifact directory")
    # Hash and parse the same bytes, so a concurrent replacement cannot change
    # the parsed content after verification.
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read bound {name} table: {path}") from exc
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{name} table checksum mismatch")
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        current_column = "stone_median" if current_metadata(report)["normalization"] == "stone_unclamped" else "current_median"
        fields = tuple(current_column if field == "stone_median" else field for field in TABLE_FIELDS)
        if reader.fieldnames != list(fields):
            raise ValueError(f"{name} table must use columns {fields}")
        rows = []
        for row in reader:
            if set(row) != set(fields) or any(value is None for value in row.values()):
                raise ValueError(f"malformed {name} table row")
            for key in ("class_id", "n_reads", "n_events"):
                if re.fullmatch(r"[+-]?[0-9]+", row[key]) is None:
                    raise ValueError(f"{name} table {key} must be an integer")
            rows.append({
                "class_id": int(row["class_id"]),
                "kmer": row["kmer"],
                "stone_median": None if row[current_column] == "NA" else float(row[current_column]),
                "n_reads": int(row["n_reads"]),
                "n_events": int(row["n_events"]),
            })
    except (UnicodeError, csv.Error, TypeError, ValueError) as exc:
        raise ValueError(f"malformed {name} table: {exc}") from exc
    try:
        return _validated_rows(rows), expected
    except ValueError as exc:
        raise ValueError(str(exc).replace("stone_median", current_column)) from exc


def _mask_to_panel(rows: list[dict[str, Any]], panel: set[int]) -> list[dict[str, Any]]:
    return [
        dict(row) if row["class_id"] in panel else {
            **row, "stone_median": None, "n_reads": 0, "n_events": 0,
        }
        for row in rows
    ]


def shared_table_panel(
    score_paths: Sequence[str | Path], reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate every prediction on the same observed class-ID intersection.

    Tables are bound by each score report's path and SHA-256 metadata. All
    reference tables must be byte-identical. The shared panel includes only
    classes observed in *every* prediction and the common reference table.
    Original coverage statistics belong in the separate per-model reports;
    this helper's coverage describes the intentionally restricted panel.

    ``reliable_metrics`` in each model still applies that model's read support
    threshold. Its reliable class IDs are explicit and may differ by model;
    only the primary (unfiltered) metrics share an identical class panel.
    """
    if not score_paths or len(score_paths) != len(reports):
        raise ValueError("score paths and reports must be nonempty and have matching lengths")
    if any(not isinstance(report, Mapping) for report in reports):
        raise ValueError("each score report must be an object")
    current = current_metadata(reports[0])
    if any(current_metadata(report) != current for report in reports[1:]):
        raise ValueError("current normalization or units differ across score reports")
    predictions, references, reference_hashes, thresholds = [], [], [], []
    for path, report in zip(score_paths, reports):
        if not isinstance(report, Mapping):
            raise ValueError("each score report must be an object")
        predicted, _ = _load_bound_table(path, report, "predicted")
        reference, digest = _load_bound_table(path, report, "reference")
        if len(predicted) != len(reference):
            raise ValueError("prediction and reference table class counts differ")
        predictions.append(predicted)
        references.append(reference)
        reference_hashes.append(digest)
        if "min_reads" not in report:
            raise ValueError("score report must declare min_reads")
        thresholds.append(report["min_reads"])
    if any(digest != reference_hashes[0] for digest in reference_hashes[1:]):
        raise ValueError("reference tables differ across models")
    if any(rows != references[0] for rows in references[1:]):
        raise ValueError("reference table contents differ across models")
    if any(value != thresholds[0] for value in thresholds[1:]):
        raise ValueError("min_reads thresholds differ across models")

    reference = references[0]
    panel = {row["class_id"] for row in reference if row["stone_median"] is not None}
    for rows in predictions:
        panel.intersection_update(row["class_id"] for row in rows if row["stone_median"] is not None)
    restricted_reference = _mask_to_panel(reference, panel)
    return {
        "current": current,
        "class_ids": sorted(panel),
        "common_classes": len(panel),
        "models": [
            compare_tables(_mask_to_panel(rows, panel), restricted_reference, min_reads=threshold)
            for rows, threshold in zip(predictions, thresholds)
        ],
    }
