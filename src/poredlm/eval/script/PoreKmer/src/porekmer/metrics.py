"""PoreKmer event-level classification and read-balanced Stone evaluation.

Each observation must be one center event, not one signal sample or frame.
Classification proportions are in [0, 1].  Confidence intervals resample
physical reads with replacement, keeping Top1 and Top5 paired within a draw.
Tables first aggregate events within a read/class and only then across reads.
No function imputes absent classes or forces table coverage.
"""

from __future__ import annotations

import csv
import numbers
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


TABLE_FIELDS = ("class_id", "kmer", "stone_median", "n_reads", "n_events")


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be between 1 and {maximum or 'infinity'}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _labels(values: Any, num_classes: int, name: str = "labels") -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if values.size and not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer class IDs")
    if values.size and (np.any(values < 0) or np.any(values >= num_classes)):
        raise ValueError(f"{name} class IDs must be in [0, num_classes)")
    return values.astype(np.int64, copy=False)


def _read_indices(read_ids: Any, n_events: int) -> tuple[np.ndarray, int]:
    read_ids = np.asarray(read_ids, dtype=object)
    if read_ids.ndim != 1 or len(read_ids) != n_events:
        raise ValueError("read_ids must have shape [N], matching observations")
    inverse = np.empty(n_events, dtype=np.int64)
    index: dict[Any, int] = {}
    for i, value in enumerate(read_ids.tolist()):
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("read_ids must not contain empty identifiers")
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError("read_ids must contain non-empty strings or integer IDs")
        if value not in index:
            index[value] = len(index)
        inverse[i] = index[value]
    return inverse, len(index)


def _kmer(class_id: int) -> str:
    # All 5-mers in A,C,G,T lexicographic order, including toy prefixes.
    return "".join("ACGT"[(class_id >> shift) & 3] for shift in (8, 6, 4, 2, 0))


def classification_metrics(
    labels: Any,
    topk: Any,
    read_ids: Any,
    *,
    num_classes: int = 1024,
    bootstrap_reps: int = 500,
    seed: int = 260916,
) -> dict[str, Any]:
    """Evaluate event predictions, with read-cluster bootstrap uncertainty.

    ``topk`` is ordered best-first and must include at least min(5,C)
    distinct classes per event.  Missing ground-truth classes have ``recall``
    None; only ``macro_recall_all`` explicitly counts them as zero.  With no
    events all rate metrics are None.  Bootstrap intervals are None if
    disabled or if fewer than two independent reads are present.
    """
    num_classes = _positive_int(num_classes, "num_classes", maximum=1024)
    bootstrap_reps = _nonnegative_int(bootstrap_reps, "bootstrap_reps")
    seed = _nonnegative_int(seed, "seed")
    labels = _labels(labels, num_classes)
    n_events = int(labels.size)
    topk = np.asarray(topk)
    required_k = min(5, num_classes)
    if topk.ndim != 2 or topk.shape[0] != n_events:
        raise ValueError("topk must have shape [N, K], matching labels")
    if not required_k <= topk.shape[1] <= num_classes:
        raise ValueError("topk must include at least min(5, num_classes) and at most num_classes columns")
    if topk.size and not np.issubdtype(topk.dtype, np.integer):
        raise ValueError("topk must contain integer class IDs")
    if topk.size and (np.any(topk < 0) or np.any(topk >= num_classes)):
        raise ValueError("topk class IDs must be in [0, num_classes)")
    if topk.size and np.any(np.diff(np.sort(topk, axis=1), axis=1) == 0):
        raise ValueError("topk must contain distinct class IDs within each event")
    topk = topk.astype(np.int64, copy=False)
    read_index, n_reads = _read_indices(read_ids, n_events)

    correct1 = topk[:, 0] == labels
    correct5 = np.any(topk[:, :required_k] == labels[:, None], axis=1)
    support = np.bincount(labels, minlength=num_classes)
    class_correct = np.bincount(labels[correct1], minlength=num_classes)
    present = support > 0
    recall = np.divide(class_correct, support, out=np.zeros(num_classes), where=present)
    per_class = [
        {
            "label": i,
            "support": int(support[i]),
            "correct": int(class_correct[i]),
            "recall": float(recall[i]) if present[i] else None,
        }
        for i in range(num_classes)
    ]
    read_counts = np.bincount(read_index, minlength=n_reads)
    read_top1 = np.divide(
        np.bincount(read_index, weights=correct1, minlength=n_reads), read_counts
    )
    read_top5 = np.divide(
        np.bincount(read_index, weights=correct5, minlength=n_reads), read_counts
    )
    ci1 = ci5 = None
    if bootstrap_reps and n_reads >= 2:
        rng = np.random.default_rng(seed)
        sampled = np.empty((bootstrap_reps, 2), dtype=np.float64)
        # One draw at a time keeps memory bounded for millions of reads.
        for i in range(bootstrap_reps):
            draws = rng.integers(0, n_reads, size=n_reads)
            sampled[i] = (read_top1[draws].mean(), read_top5[draws].mean())
        bounds = np.quantile(sampled, [0.025, 0.975], axis=0)
        ci1, ci5 = bounds[:, 0].tolist(), bounds[:, 1].tolist()

    return {
        "unit": "center_event",
        "rate_scale": "fraction",
        "n_events": n_events,
        "n_reads": n_reads,
        "num_classes": num_classes,
        "top5_effective_k": required_k,
        "event_micro_top1": float(correct1.mean()) if n_events else None,
        "event_micro_top5": float(correct5.mean()) if n_events else None,
        "read_macro_top1": float(read_top1.mean()) if n_reads else None,
        "read_macro_top5": float(read_top5.mean()) if n_reads else None,
        "macro_recall_present": float(recall[present].mean()) if present.any() else None,
        "macro_recall_all": float(recall.mean()) if n_events else None,
        "macro_recall_all_missing_classes_count_as_zero": True,
        "supported_classes": int(present.sum()),
        "zero_recall_supported_classes": int(np.count_nonzero(present & (class_correct == 0))),
        "read_macro_top1_ci95": ci1,
        "read_macro_top5_ci95": ci5,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": seed,
        "bootstrap_unit": "physical_read",
        "bootstrap_method": "paired_top1_top5_read_resampling_percentile",
        "bootstrap_note": (
            "disabled" if not bootstrap_reps else
            "requires_at_least_two_reads" if n_reads < 2 else None
        ),
        "per_class": per_class,
    }


def table_rows(
    labels_or_predictions: Any,
    currents: Any,
    read_ids: Any,
    num_classes: int = 1024,
) -> list[dict[str, Any]]:
    """Build median-within-read then median-across-read Stone tables.

    ``currents`` must already contain one center-event current per observation.
    Neighbor-event currents must not be pooled into the center's measurement.
    """
    num_classes = _positive_int(num_classes, "num_classes", maximum=1024)
    labels = _labels(labels_or_predictions, num_classes)
    try:
        currents = np.asarray(currents, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("currents must contain finite numeric values") from exc
    if currents.ndim != 1 or len(currents) != len(labels):
        raise ValueError("currents must have shape [N], matching labels")
    if not np.isfinite(currents).all():
        raise ValueError("currents must contain finite numeric values")
    read_index, _ = _read_indices(read_ids, len(labels))
    event_counts = np.bincount(labels, minlength=num_classes)
    class_values: list[list[float]] = [[] for _ in range(num_classes)]
    # Group observations once, instead of rescanning all events per class/read.
    if labels.size:
        order = np.lexsort((read_index, labels))
        sorted_labels = labels[order]
        sorted_reads = read_index[order]
        edges = np.flatnonzero(
            (sorted_labels[1:] != sorted_labels[:-1]) |
            (sorted_reads[1:] != sorted_reads[:-1])
        ) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(labels)]))
        for start, end in zip(starts, ends):
            class_values[int(sorted_labels[start])].append(
                float(np.median(currents[order[start:end]]))
            )
    return [
        {
            "class_id": i,
            "kmer": _kmer(i),
            "stone_median": float(np.median(values)) if values else None,
            "n_reads": len(values),
            "n_events": int(event_counts[i]),
        }
        for i, values in enumerate(class_values)
    ]


def _validated_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    _positive_int(len(rows), "table row count", maximum=1024)
    by_class: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not all(field in row for field in TABLE_FIELDS):
            raise ValueError(f"table rows must contain {TABLE_FIELDS}")
        class_id = _nonnegative_int(row["class_id"], "class_id")
        if class_id >= len(rows) or class_id in by_class:
            raise ValueError("table must contain each class_id from 0 to C-1 exactly once")
        if row["kmer"] != _kmer(class_id):
            raise ValueError("kmer does not match ACGT-lexicographic 5-mer class_id")
        n_reads = _nonnegative_int(row["n_reads"], "n_reads")
        n_events = _nonnegative_int(row["n_events"], "n_events")
        value = row["stone_median"]
        if value is None:
            if n_reads or n_events:
                raise ValueError("missing stone_median must have zero n_reads and n_events")
        else:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
                raise ValueError("stone_median must be finite or None")
            if n_reads < 1 or n_events < n_reads:
                raise ValueError("observed rows require n_events >= n_reads >= 1")
        by_class[class_id] = row
    return [by_class[i] for i in range(len(rows))]


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    x = left - left.mean()
    y = right - right.mean()
    x_norm, y_norm = np.linalg.norm(x), np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        return None
    return float(np.clip(np.dot(x / x_norm, y / y_norm), -1.0, 1.0))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    if len(values):
        edges = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [len(values)]))
        for start, end in zip(starts, ends):
            ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
    return ranks


def _paired_table_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if not len(left):
        return dict(pearson=None, spearman=None, mae=None, rmse=None)
    residual = left - right
    return {
        "pearson": _pearson(left, right),
        "spearman": _pearson(_average_ranks(left), _average_ranks(right)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
    }


def compare_tables(
    pred_rows: Iterable[Mapping[str, Any]],
    ref_rows: Iterable[Mapping[str, Any]],
    *,
    min_reads: int = 1,
) -> dict[str, Any]:
    """Compare common class IDs; report hard and supported coverage separately.

    Correlations are undefined (None) for fewer than two common classes or a
    constant vector.  MAE/RMSE remain defined for a single common class.  The
    reliable subset requires ``min_reads`` in *both* independently built tables.
    """
    min_reads = _positive_int(min_reads, "min_reads")
    pred = _validated_rows(pred_rows)
    ref = _validated_rows(ref_rows)
    if len(pred) != len(ref):
        raise ValueError("prediction and reference tables must have the same total classes")
    pred_present = np.asarray([r["stone_median"] is not None for r in pred])
    ref_present = np.asarray([r["stone_median"] is not None for r in ref])
    common = pred_present & ref_present
    reliable = common & np.asarray([
        p["n_reads"] >= min_reads and r["n_reads"] >= min_reads
        for p, r in zip(pred, ref)
    ])
    left = np.asarray([p["stone_median"] for p, keep in zip(pred, common) if keep], dtype=np.float64)
    right = np.asarray([r["stone_median"] for r, keep in zip(ref, common) if keep], dtype=np.float64)
    reliable_left = np.asarray([p["stone_median"] for p, keep in zip(pred, reliable) if keep], dtype=np.float64)
    reliable_right = np.asarray([r["stone_median"] for r, keep in zip(ref, reliable) if keep], dtype=np.float64)
    return {
        "total_classes": len(pred),
        "hard_coverage": int(pred_present.sum()),
        "hard_coverage_fraction": float(pred_present.mean()),
        "reference_coverage": int(ref_present.sum()),
        "common_classes": int(common.sum()),
        "common_class_ids": np.flatnonzero(common).tolist(),
        "min_reads": min_reads,
        "reliable_common_classes": int(reliable.sum()),
        "reliable_common_class_ids": np.flatnonzero(reliable).tolist(),
        **_paired_table_metrics(left, right),
        "reliable_metrics": _paired_table_metrics(reliable_left, reliable_right),
        "aggregation": "median_of_within_read_class_medians",
        "missing_classes_imputed": False,
    }


def write_table(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a complete table using NA for missing levels; never overwrite."""
    rows = _validated_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in TABLE_FIELDS}
            output["stone_median"] = (
                "NA" if row["stone_median"] is None else format(float(row["stone_median"]), ".17g")
            )
            writer.writerow(output)
