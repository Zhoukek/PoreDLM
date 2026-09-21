"""PoreKmer: prepare conditional, event-boundary-aware k-mer datasets.

This module never estimates a physical register. Existing dense labels are copied
under an explicit unverified-register acknowledgement; schema 2 preserves imported
upstream phase calibration without claiming independent validation. Event selection
uses move boundaries and alignment-derived annotations, so these datasets are not a
signal-only, end-to-end benchmark even though feature files contain no labels.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np


SPLITS = frozenset({"train", "validation", "prediction", "reference", "test"})
INFORMATION_BOUNDARY = "known_move_boundaries_and_alignment_filtered_not_signal_only"
REGISTER_WARNING = (
    "Source start-offset labels are reused without recalibration. Historical "
    "start/center-offset semantics are unresolved; these results do not validate "
    "the physical pore register. No automatic two-base correction is applied."
)
FEATURE_KEYS = frozenset({
    "features", "centers", "center_ids", "currents", "event_starts", "event_ends",
})
LEGACY_CURRENT = {"normalization": "stone_unclamped", "units": "dimensionless"}


def current_metadata(artifact: dict[str, Any]) -> dict[str, str]:
    """Normalize current provenance; old artifacts contain unclamped Stone values."""
    if "current" not in artifact:
        return dict(LEGACY_CURRENT)
    current = artifact["current"]
    if not isinstance(current, dict):
        raise ValueError("current metadata must be an object")
    normalization, units = current.get("normalization"), current.get("units")
    if normalization not in ("input_signal", "stone_unclamped"):
        raise ValueError("current.normalization must be input_signal or stone_unclamped")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("current.units must be a nonempty string")
    return {"normalization": normalization, "units": units}


def validate_phase_register(register: Any) -> None:
    """Validate upstream phase provenance without claiming independent validation."""
    if not isinstance(register, dict) or register.get("status") != "phase_calibrated":
        raise ValueError("schema 2 requires a phase_calibrated register")
    if register.get("label_coordinate_system") != "reference":
        raise ValueError("phase-calibrated labels must use reference coordinates")
    if register.get("kmer_pattern") != "XXNXX":
        raise ValueError("phase-calibrated labels require kmer_pattern=XXNXX")
    if _integer(register.get("additional_offset_bases"), "additional_offset_bases") != 0:
        raise ValueError("additional_offset_bases must be zero; upstream phase is already applied")
    if register.get("independently_validated") is not False:
        raise ValueError("upstream phase calibration must not be labeled independently validated")
    calibrations = register.get("calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError("phase-calibrated register requires nonempty calibrations provenance")
    seen: set[tuple[str, str]] = set()
    for calibration in calibrations:
        if not isinstance(calibration, dict):
            raise ValueError("each calibration provenance entry must be an object")
        path = calibration.get("path")
        if not isinstance(path, str) or not path.strip() or "\0" in path:
            raise ValueError("calibration path must be a nonempty path string")
        digest = _hash(calibration.get("sha256"), "calibration sha256")
        identity = (path, digest)
        if identity in seen:
            raise ValueError("duplicate calibration provenance entry")
        seen.add(identity)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _safe_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("record path must be a nonempty safe relative path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError(f"unsafe relative record path: {relative!r}")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"record path escapes corpus root: {relative!r}")
    return resolved


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _checked_path(root: Path, row: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = _safe_path(root, row.get(path_key))
    expected = _hash(row.get(hash_key), hash_key)
    if file_sha256(path) != expected:
        raise ValueError(f"checksum mismatch for {row[path_key]}")
    return path


def _identities(samples: Any, *, source: bool, root: Path) -> None:
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a nonempty list")
    physical_splits: dict[str, str] = {}
    read_ids: set[str] = set()
    records: set[str] = set()
    paths: set[str] = set()
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("sample rows must be objects")
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"unsupported split: {split!r}")
        read = row.get("physical_read_id", row.get("read_id"))
        if not isinstance(read, str) or not read:
            raise ValueError("every sample requires a physical read_id")
        if not isinstance(row.get("read_id"), str) or not row["read_id"]:
            raise ValueError("every sample requires a nonempty read_id")
        if not source and "physical_read_id" in row and row["physical_read_id"] != row["read_id"]:
            raise ValueError("canonical physical_read_id must equal read_id; metric clusters cannot use aliases")
        previous = physical_splits.get(read)
        if previous is not None:
            if previous != split:
                raise ValueError(f"physical-read split leakage: {read!r}")
            raise ValueError(f"duplicate physical read violates one-chunk-per-read policy: {read!r}")
        physical_splits[read] = split
        if row["read_id"] in read_ids:
            raise ValueError(f"duplicate read_id: {row['read_id']!r}")
        read_ids.add(row["read_id"])
        record = row.get("record_id", row.get("query_name", row["read_id"]))
        if not isinstance(record, str) or not record or record in records:
            raise ValueError(f"invalid or duplicate record identity: {record!r}")
        records.add(record)
        for path_key, hash_key in (
            (("path", "npz_sha256"),) if source else
            (("features_path", "features_sha256"), ("truth_path", "truth_sha256"))
        ):
            resolved = str(_safe_path(root, row.get(path_key)))
            if resolved in paths:
                raise ValueError("duplicate record file path")
            paths.add(resolved)
            _hash(row.get(hash_key), hash_key)
        if not source:
            _integer(row.get("n_centers"), "n_centers", 1)


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a complete canonical manifest and reject cross-split read leakage."""
    path = Path(path)
    manifest = _json(path)
    version = _integer(manifest.get("schema_version"), "schema_version")
    if version not in (1, 2) or manifest.get("kind") != "kmer_event_context":
        raise ValueError("unsupported event-context manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("event-context corpus is not complete")
    _integer(manifest.get("hidden_dim"), "hidden_dim", 1)
    if manifest.get("num_classes") != 1024 or manifest.get("max_window") != 5:
        raise ValueError("expected 1024 classes and a shared five-event window")
    _hash(manifest.get("source_manifest_sha256"), "source_manifest_sha256")
    if manifest.get("information_boundary") != INFORMATION_BOUNDARY:
        raise ValueError("missing known-boundary/alignment-filtered limitation")
    register = manifest.get("register", {})
    if version == 1:
        if not isinstance(register, dict) or register.get("status") != "unverified":
            raise ValueError("schema 1 only supports explicitly unverified source registers")
        start = _integer(register.get("start_offset_bases"), "start_offset_bases")
        if register.get("center_offset_bases") != start + 2 or not register.get("warning"):
            raise ValueError("inconsistent start/center register metadata")
        if current_metadata(manifest) != LEGACY_CURRENT:
            raise ValueError("schema 1 currents must preserve legacy Stone normalization and units")
    else:
        validate_phase_register(register)
        if "current" not in manifest:
            raise ValueError("schema 2 requires explicit current metadata")
        current_metadata(manifest)
    _identities(manifest.get("samples"), source=False, root=path.parent)
    return manifest


def _int_array(value: np.ndarray, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has inconsistent length")
    return array.astype(np.int64, copy=False)


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _validate_features(arrays: dict[str, np.ndarray], n_centers: int | None = None) -> None:
    if set(arrays) != FEATURE_KEYS:
        raise ValueError("feature file must contain only the six approved feature arrays")
    features = _finite(arrays["features"], "features")
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError("features must have shape [events, hidden_dim]")
    count = len(features)
    starts = _int_array(arrays["event_starts"], "event_starts", count)
    ends = _int_array(arrays["event_ends"], "event_ends", count)
    centers = _int_array(arrays["centers"], "centers", n_centers)
    ids = _int_array(arrays["center_ids"], "center_ids", len(centers))
    currents = _finite(arrays["currents"], "currents")
    if currents.shape != (len(centers),):
        raise ValueError("currents must have one value per center")
    if np.any(starts < 0) or np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("events must have positive, ordered, nonoverlapping sample intervals")
    if np.any(centers < 2) or np.any(centers + 2 >= count) or np.any(np.diff(centers) <= 0):
        raise ValueError("center indices must be sorted, unique, and have two real neighbors on each side")
    if not np.array_equal(ids, starts[centers]):
        raise ValueError("center_ids must be central-event start sample coordinates")
    for shift in range(-2, 2):
        if np.any(ends[centers + shift] != starts[centers + shift + 1]):
            raise ValueError("a context window bridges a missing event or signal gap")


def load_features(root: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load checksummed inference features without opening the truth sidecar."""
    path = _checked_path(Path(root), row, "features_path", "features_sha256")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    _validate_features(arrays, _integer(row.get("n_centers"), "n_centers", 1))
    return arrays


def load_truth(root: Path, row: dict[str, Any]) -> np.ndarray:
    path = _checked_path(Path(root), row, "truth_path", "truth_sha256")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"labels"}:
            raise ValueError("truth file must contain only labels")
        labels = _int_array(archive["labels"], "labels", _integer(row.get("n_centers"), "n_centers", 1))
    if np.any((labels < 0) | (labels >= 1024)):
        raise ValueError("truth labels must be in [0, 1024)")
    return labels


def make_windows(features: dict[str, np.ndarray]) -> np.ndarray:
    """Return ordered [centers, 5, hidden_dim] windows; never pad or bridge gaps."""
    _validate_features(features)
    centers = _int_array(features["centers"], "centers")
    positions = centers[:, None] + np.arange(-2, 3, dtype=np.int64)[None, :]
    return np.asarray(features["features"][positions], dtype=np.float32)


def _extract_events(path: Path, geometry: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, Counter]:
    required = {
        "hidden", "labels", "weights", "purity_mask", "owned_mask", "state_ids",
        "query_positions", "reference_positions", "sample_starts", "sample_ends",
        "signal_stone_unclamped",
    }
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError(f"dense record missing arrays: {sorted(required.difference(archive.files))}")
        data = {name: archive[name] for name in required}
    hidden = _finite(data["hidden"], "hidden")
    if hidden.ndim != 2 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
        raise ValueError("hidden must be a nonempty [frames, hidden_dim] matrix")
    n = len(hidden)
    integer_names = ("labels", "state_ids", "query_positions", "reference_positions", "sample_starts", "sample_ends")
    for name in integer_names:
        data[name] = _int_array(data[name], name, n)
    for name in ("purity_mask", "owned_mask"):
        value = data[name]
        if value.shape != (n,) or value.dtype.kind not in "biu" or not np.isin(value, [0, 1]).all():
            raise ValueError(f"{name} must be a boolean frame mask")
        data[name] = value.astype(bool)
    weights = _finite(data["weights"], "weights")
    if weights.shape != (n,) or np.any((weights < 0) | (weights > 1 + 1e-6)):
        raise ValueError("weights must be finite inverse-dwell frame weights in [0, 1]")
    if np.any((data["labels"] < -1) | (data["labels"] >= 1024)):
        raise ValueError("dense labels must be -1 or in [0, 1024)")
    signal = _finite(data["signal_stone_unclamped"], "signal_stone_unclamped")
    stride = geometry["model_stride_samples"]
    if signal.ndim != 1 or len(signal) != n * stride:
        raise ValueError("signal length and hidden stride geometry disagree")
    if "chunk_samples" in geometry and _integer(geometry["chunk_samples"], "chunk_samples", 1) != len(signal):
        raise ValueError("declared chunk_samples disagrees with signal length")
    if "hidden_frames" in geometry and _integer(geometry["hidden_frames"], "hidden_frames", 1) != n:
        raise ValueError("declared hidden_frames disagrees with hidden shape")
    starts, ends = data["sample_starts"], data["sample_ends"]
    if not np.array_equal(starts, np.arange(n) * stride) or not np.array_equal(ends, starts + stride):
        raise ValueError("dense sample intervals do not follow the declared fixed grid")
    central_start, central_end = geometry["central_supervised_samples"]
    if not 0 <= central_start < central_end <= len(signal):
        raise ValueError("invalid central supervision interval")
    expected_owned = (starts >= central_start) & (ends <= central_end)
    if "central_supervised_frames" in geometry:
        expected_frames = [(central_start + stride - 1) // stride, central_end // stride]
        if geometry["central_supervised_frames"] != expected_frames:
            raise ValueError("declared central frame and sample geometry disagree")
    if not np.array_equal(data["owned_mask"], expected_owned):
        raise ValueError("owned mask disagrees with manifest central geometry")
    eligible_mask = data["purity_mask"] & data["owned_mask"] & (data["query_positions"] >= 0)
    if np.any((weights > 0) & ~eligible_mask):
        raise ValueError("positive weights violate purity/ownership/query-position contract")
    valid = (weights > 0) & eligible_mask
    if np.any(valid & ((data["labels"] < 0) | (data["reference_positions"] < 0) | (data["state_ids"] < 0))):
        raise ValueError("positive-weight valid frames need nonnegative labels and state coordinates")
    indices = np.flatnonzero(valid)
    stats: Counter = Counter(total_frames=n, valid_frames=len(indices))
    event_features, event_labels, event_queries, event_starts, event_ends = [], [], [], [], []
    if len(indices):
        valid_queries = data["query_positions"][indices]
        query_change = np.diff(valid_queries)
        if np.any(query_change < 0):
            raise ValueError("query event coordinates must be nondecreasing in signal order")
        same_query = query_change == 0
        for name in ("state_ids", "reference_positions", "labels"):
            if np.any(same_query & (np.diff(data[name][indices]) != 0)):
                raise ValueError("an event has inconsistent labels or reference/state identity")
        query_runs = np.concatenate(([0], np.flatnonzero(query_change != 0) + 1))
        query_weight_sums = np.add.reduceat(weights[indices].astype(np.float64), query_runs)
        if np.any(query_weight_sums > 1.0 + 1.1e-5):
            raise ValueError("inverse-dwell weight exceeds one for a query event; possible duplicate event")
        # Group by a run's event identity, never by class identity. Repeated
        # homopolymer labels remain distinct events; disconnected runs stay apart.
        boundaries = np.flatnonzero(
            (np.diff(indices) != 1)
            | (np.diff(data["state_ids"][indices]) != 0)
            | (np.diff(data["query_positions"][indices]) != 0)
        ) + 1
        for group in np.split(indices, boundaries):
            stats["candidate_event_runs"] += 1
            labels = data["labels"][group]
            references = data["reference_positions"][group]
            if np.any(labels != labels[0]) or np.any(references != references[0]):
                raise ValueError("an event run has inconsistent labels or reference positions")
            if not np.isclose(np.sum(weights[group], dtype=np.float64), 1.0, rtol=1e-5, atol=1e-6):
                stats["omitted_partial_event_runs"] += 1
                continue
            if not np.allclose(weights[group], 1.0 / len(group), rtol=1e-5, atol=1e-6):
                raise ValueError("complete event does not have uniform inverse-dwell weights")
            if np.any(starts[group[1:]] != ends[group[:-1]]):
                raise ValueError("event samples are not contiguous")
            # Accumulate in float64 so finite float32 inputs do not overflow
            # merely because an event has multiple frames. Storage stays f32.
            event_features.append(np.mean(hidden[group], axis=0, dtype=np.float64))
            event_labels.append(int(labels[0]))
            event_queries.append(int(data["query_positions"][group[0]]))
            event_starts.append(int(starts[group[0]]))
            event_ends.append(int(ends[group[-1]]))
    starts_array = np.asarray(event_starts, dtype=np.int64)
    ends_array = np.asarray(event_ends, dtype=np.int64)
    queries = np.asarray(event_queries, dtype=np.int64)
    centers = np.arange(2, max(2, len(event_features) - 2), dtype=np.int64)
    eligible = np.ones(len(centers), dtype=bool)
    for shift in range(-2, 2):
        eligible &= queries[centers + shift + 1] == queries[centers + shift] + 1
        eligible &= ends_array[centers + shift] == starts_array[centers + shift + 1]
    centers = centers[eligible]
    features = {
        "features": np.asarray(event_features, dtype=np.float32).reshape(-1, hidden.shape[1]),
        "centers": centers,
        "center_ids": starts_array[centers],
        "currents": np.asarray([
            np.median(signal[starts_array[index]:ends_array[index]].astype(np.float64))
            for index in centers
        ], dtype=np.float32),
        "event_starts": starts_array,
        "event_ends": ends_array,
    }
    labels = np.asarray(event_labels, dtype=np.int64)[centers]
    stats.update(complete_events=len(event_features), eligible_centers=len(centers))
    _validate_features(features)
    return features, labels, stats


def prepare_corpus(
    source_manifest: Path,
    output_dir: Path,
    *,
    allow_unverified_register: bool = False,
    max_reads_per_split: int | None = None,
) -> dict[str, Any]:
    """Pool complete events and create common five-event eligible centers.

    All variants (one, three, and five events) must use the returned identical
    centers. Source labels are never silently shifted. A max-read limit is a
    deterministic smoke option, not a scientifically representative subsample.
    Existing output directories are rejected, including empty ones.
    """
    if not allow_unverified_register:
        raise ValueError("register is unverified; explicitly set allow_unverified_register=True for exploratory use")
    if max_reads_per_split is not None:
        _integer(max_reads_per_split, "max_reads_per_split", 1)
    source_path, output = Path(source_manifest).resolve(), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    source = _json(source_path)
    if source.get("status") != "complete":
        raise ValueError("source dense corpus must have status complete")
    _identities(source.get("samples"), source=True, root=source_path.parent)
    geometry = source.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("source requires explicit dense geometry")
    start_offset = _integer(geometry.get("sequence_offset_bases"), "sequence_offset_bases")
    _integer(geometry.get("model_stride_samples"), "model_stride_samples", 1)
    if _integer(geometry.get("frame_lag", 0), "frame_lag") != 0:
        raise ValueError("nonzero source frame_lag is unsupported; this tool does not transform hidden coordinates")
    central = geometry.get("central_supervised_samples")
    if not isinstance(central, list) or len(central) != 2:
        raise ValueError("source requires central_supervised_samples [start, end]")
    for endpoint in central:
        _integer(endpoint, "central_supervised_samples endpoint", 0)
    for row in source["samples"]:
        if "sequence_offset_bases" in row and row["sequence_offset_bases"] != start_offset:
            raise ValueError("sample and manifest sequence offsets disagree")
    selected = []
    for split in sorted(SPLITS):
        rows = sorted((row for row in source["samples"] if row["split"] == split), key=lambda row: row["read_id"])
        selected.extend(rows if max_reads_per_split is None else rows[:max_reads_per_split])
    output.mkdir(parents=True, exist_ok=False)
    samples, totals, split_counts = [], Counter(), Counter()
    hidden_dim = None
    for row in selected:
        path = _checked_path(source_path.parent, row, "path", "npz_sha256")
        features, labels, stats = _extract_events(path, geometry)
        dim = features["features"].shape[1]
        if hidden_dim is not None and hidden_dim != dim:
            raise ValueError("hidden dimensions differ across source records")
        hidden_dim = dim
        totals.update(stats)
        if len(labels) == 0:
            totals["omitted_reads_without_eligible_centers"] += 1
            continue
        physical_read = row.get("physical_read_id", row["read_id"])
        record = row.get("record_id", row.get("query_name", row["read_id"]))
        key = hashlib.sha256((physical_read + "\0" + record).encode()).hexdigest()[:24]
        relative_features = Path("features") / row["split"] / f"{key}.npz"
        relative_truth = Path("truth") / row["split"] / f"{key}.npz"
        for relative, arrays in ((relative_features, features), (relative_truth, {"labels": labels})):
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
        samples.append({
            "read_id": physical_read,
            "physical_read_id": physical_read,
            "record_id": record,
            "split": row["split"],
            "features_path": relative_features.as_posix(),
            "truth_path": relative_truth.as_posix(),
            "features_sha256": file_sha256(output / relative_features),
            "truth_sha256": file_sha256(output / relative_truth),
            "n_centers": len(labels),
            "source_path": row["path"],
            "source_npz_sha256": row["npz_sha256"],
            "source_hidden_representation": row.get("hidden_representation"),
        })
        split_counts[row["split"]] += 1
    if not samples:
        raise ValueError("no eligible five-event centers; incomplete output has no complete manifest")
    manifest = {
        "schema_version": 1,
        "kind": "kmer_event_context",
        "status": "complete",
        "hidden_dim": hidden_dim,
        "num_classes": 1024,
        "max_window": 5,
        "source_manifest": str(source_path),
        "source_manifest_sha256": file_sha256(source_path),
        "source_geometry": geometry,
        "source_representation_contract": source.get("representation_contract"),
        "source_model_provenance": source.get("model"),
        "register": {
            "status": "unverified",
            "start_offset_bases": start_offset,
            "center_offset_bases": start_offset + 2,
            "warning": REGISTER_WARNING,
        },
        "information_boundary": INFORMATION_BOUNDARY,
        "selection": {
            "event_pooling": "arithmetic mean of raw hidden over complete, pure, owned event frames",
            "center_current": "median of all unclamped Stone samples of central event only",
            "common_centers_for_windows": [1, 3, 5],
            "full_dwell_required": True,
            "missing_states_break_windows": True,
            "one_chunk_per_physical_read": True,
            "truth_dependent_filtering": "existing exact-alignment labels, purity, ownership, full dwell and consecutive query positions",
        },
        "smoke_limit": {
            "max_reads_per_split": max_reads_per_split,
            "enabled": max_reads_per_split is not None,
            "selection": "read_id lexicographic order independently within each split",
            "limitation": "Any read limit is a smoke/debug subset, not a representative full evaluation.",
        },
        "counts": {
            "source_reads": len(source["samples"]),
            "selected_reads": len(selected),
            "written_reads": len(samples),
            "written_reads_by_split": dict(split_counts),
            **dict(totals),
        },
        "samples": samples,
    }
    _write_json(output / "manifest.json", manifest)
    # Validate the serialized contract as well as in-memory arrays.
    load_manifest(output / "manifest.json")
    return manifest
