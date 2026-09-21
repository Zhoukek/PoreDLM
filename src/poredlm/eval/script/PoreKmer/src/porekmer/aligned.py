"""Adapt calibrated NanoSignalAlign records and independently exported hidden.

Only NumPy and the PoreKmer data contract are required. Calibrated spans are
consumed as supplied: neither the waveform nor the labels are shifted again.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    INFORMATION_BOUNDARY, SPLITS, _hash, _integer, _validate_features,
    _write_json, file_sha256,
)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _parse(raw, label):
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _path(value, root, name):
    path = Path(_string(value, name))
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _real(array, name, ndim):
    if not isinstance(array, np.ndarray) or array.ndim != ndim or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric {ndim}-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


class _Files:
    """Cache mmap arrays and file hashes; detect mutations during preparation."""

    def __init__(self):
        self.hashes = {}
        self.signatures = {}
        self.arrays = OrderedDict()

    @staticmethod
    def signature(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def digest(self, path):
        signature = self.signature(path)
        if path in self.hashes:
            if signature != self.signatures[path]:
                raise ValueError(f"source file changed during preparation: {path}")
            return self.hashes[path]
        digest = file_sha256(path)
        if self.signature(path) != signature:
            raise ValueError(f"source file changed while hashing: {path}")
        self.hashes[path], self.signatures[path] = digest, signature
        return digest

    def array(self, path):
        self.digest(path)
        if path not in self.arrays:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if not isinstance(value, np.ndarray):
                value.close()
                raise ValueError(f"expected a single NPY array: {path}")
            self.arrays[path] = value
            if len(self.arrays) > 8:
                self.arrays.popitem(last=False)
        self.arrays.move_to_end(path)
        return self.arrays[path]

    def verify(self):
        for path, signature in self.signatures.items():
            if self.signature(path) != signature:
                raise ValueError(f"source file changed during preparation: {path}")


def _signal_ref(reference, root, files, name):
    if not isinstance(reference, dict):
        raise ValueError(f"{name} must be an object")
    path = _path(reference.get("path"), root, f"{name}.path")
    array = files.array(path)
    if "row" in reference:
        row = _integer(reference["row"], f"{name}.row", 0)
        if array.ndim != 2 or row >= len(array):
            raise ValueError(f"{name}.row requires an in-range row of a two-dimensional array")
        values = array[row]
    else:
        row = None
        values = array
    if values.ndim != 1:
        raise ValueError(f"{name} must resolve to a one-dimensional signal")
    start = _integer(reference.get("start", 0), f"{name}.start", 0)
    end = _integer(reference.get("end", len(values)), f"{name}.end", 0)
    if not 0 <= start < end <= len(values):
        raise ValueError(f"{name} requires a nonempty in-range [start,end) slice")
    if "length" in reference and _integer(reference["length"], f"{name}.length", 1) != end - start:
        raise ValueError(f"{name}.length disagrees with its slice")
    selected = _real(values[start:end], name, 1)
    descriptor = {"path": str(path), "sha256": files.digest(path), "start": start, "end": end,
                  "length": len(selected)}
    if row is not None:
        descriptor["row"] = row
    return selected, descriptor


def _signal(record, root, files):
    inline, referenced = "signal" in record, "signal_ref" in record
    if inline == referenced:
        raise ValueError("record must contain exactly one of signal and signal_ref")
    if inline:
        if not isinstance(record["signal"], list):
            raise ValueError("inline signal must be a numeric list")
        values = _real(np.asarray(record["signal"]), "inline signal", 1)
        if not len(values):
            raise ValueError("inline signal must be nonempty")
        return values, {"storage": "inline", "length": len(values),
                        "sha256": hashlib.sha256(json.dumps(record["signal"], separators=(",", ":"),
                                                             allow_nan=False).encode()).hexdigest()}
    return _signal_ref(record.get("signal_ref"), root, files, "signal_ref")


def _physical_read(record):
    return _string(record.get("physical_read_id"), "record physical_read_id")


def _record_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _selected_records(samples, files):
    grouped = defaultdict(dict)
    for sample in samples:
        grouped[sample["_aligned_path"]][sample["record_index"]] = sample
    for path in sorted(grouped):
        requested = grouped[path]
        files.digest(path)
        opener = gzip.open if path.suffix == ".gz" else open
        found = set()
        with opener(path, "rt", encoding="utf-8") as handle:
            index = 0
            for line in handle:
                if not line.strip():
                    continue
                if index in requested:
                    record = _parse(line, f"{path} record {index}")
                    sample = requested[index]
                    if record.get("read_id") != sample["read_id"]:
                        raise ValueError(f"record_index/read_id mismatch at {path}:{index}")
                    if _physical_read(record) != sample["physical_read_id"]:
                        raise ValueError(f"aligned physical_read_id mismatch at {path}:{index}")
                    found.add(index)
                    yield sample, record
                    if len(found) == len(requested):
                        break
                index += 1
        if found != set(requested):
            raise ValueError(f"aligned JSONL record_index out of range: {path}, {sorted(set(requested) - found)}")


def _calibration(record, root, files, cache):
    alignment = record.get("signal_alignment")
    if not isinstance(alignment, dict) or alignment.get("stage") != "phase_calibrated":
        raise ValueError("record requires phase_calibrated signal_alignment")
    orientation = alignment.get("query_orientation")
    if not isinstance(orientation, str) or orientation not in {"direct", "reverse"}:
        raise ValueError("query_orientation must be direct or reverse")
    offset = _integer(alignment.get("offset_bases"), "signal_alignment.offset_bases")
    role = alignment.get("calibration_role")
    if not isinstance(role, str) or role not in {"analysis", "calibration"}:
        raise ValueError("calibration_role must be analysis or calibration")
    path = _path(alignment.get("calibration_path"), root, "calibration_path")
    expected = _hash(alignment.get("calibration_sha256"), "calibration_sha256")
    if files.digest(path) != expected:
        raise ValueError(f"calibration checksum mismatch: {path}")
    if path not in cache:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"calibration changed while reading: {path}")
        value = _parse(raw, "calibration")
        stats = value.get("stats", {})
        ids = stats.get("calibration_read_ids", []) if isinstance(stats, dict) else None
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("calibration stats.calibration_read_ids must be a string list")
        cache[path] = (value, set(ids))
    value, ids = cache[path]
    if _integer(value.get("selected_shift"), "calibration selected_shift") != offset:
        raise ValueError("calibration selected_shift and signal_alignment offset disagree")
    if value.get("selected_orientation") != orientation:
        raise ValueError("calibration selected_orientation and record orientation disagree")
    for container, label in ((alignment, "signal_alignment"), (value, "calibration")):
        pattern = _string(container.get("kmer_pattern"), f"{label}.kmer_pattern")
        if pattern.count("N") != 1 or set(pattern) - {"X", "N"}:
            raise ValueError(f"{label}.kmer_pattern must contain one N and otherwise only X")
    if alignment["kmer_pattern"] != value["kmer_pattern"]:
        raise ValueError("calibration and record kmer_pattern disagree")
    descriptor = {"path": str(path), "sha256": expected, "query_orientation": orientation,
                  "offset_bases": offset, "kmer_pattern": value.get("kmer_pattern"),
                  "confidence": value.get("confidence"), "calibration_read_count": len(ids)}
    if "phase_inherited_from" in value:
        descriptor["phase_inherited_from"] = value["phase_inherited_from"]
    return descriptor, ids, role


def _reference_matches(record):
    sequence = _string(record.get("seq"), "seq").upper()
    reference = _string(record.get("ref"), "ref").upper()
    alignment = record.get("align")
    if not isinstance(alignment, dict):
        raise ValueError("record requires align")
    query = _string(alignment.get("seq_aligned"), "seq_aligned").upper()
    ref = _string(alignment.get("ref_aligned"), "ref_aligned").upper()
    if len(query) != len(ref) or query.replace("-", "") != sequence or ref.replace("-", "") != reference:
        raise ValueError("aligned strings have inconsistent lengths or ungapped seq/ref")
    matches, q, r = {}, 0, 0
    for qb, rb in zip(query, ref):
        if qb == "-" and rb == "-":
            raise ValueError("alignment cannot contain a double gap")
        if qb != "-" and rb != "-" and qb == rb and qb in "ACGT":
            matches[r] = q
        q += qb != "-"
        r += rb != "-"
    return reference, matches


def _events(record, hidden, signal_length, current, geometry, orientation):
    reference, matches = _reference_matches(record)
    spans = record.get("base_sample_span_ref_calibrated")
    if not isinstance(spans, list) or len(spans) != len(reference):
        raise ValueError("calibrated spans must contain one interval per reference base")
    parsed = []
    for r, span in enumerate(spans):
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("each calibrated span must be a two-element array")
        if span == [None, None]:
            continue
        if None in span:
            raise ValueError("calibrated span must have either two coordinates or two nulls")
        start = _integer(span[0], "event start", 0)
        end = _integer(span[1], "event end", 0)
        if start > end or end > signal_length:
            raise ValueError("calibrated span lies outside the signal or is reversed")
        if start == end:
            continue
        parsed.append((start, end, r, matches.get(r)))
    parsed.sort()
    step = 1 if orientation == "direct" else -1
    for left, right in zip(parsed, parsed[1:]):
        if right[0] < left[1]:
            raise ValueError("calibrated event intervals overlap")
        if (right[2] - left[2]) * step <= 0:
            raise ValueError("calibrated spans contradict query_orientation")
    stride, origin = geometry["model_stride_samples"], geometry["frame_origin_samples"]
    events, pooled = [], []
    stats = Counter(reference_bases=len(reference), nonempty_spans=len(parsed))
    for start, end, r, q in parsed:
        if q is None:
            stats["omitted_nonexact_events"] += 1
            continue
        first = max(0, (start - origin + stride - 1) // stride)
        last = min(len(hidden), (end - origin + stride - 1) // stride)
        if first >= last:
            stats["omitted_events_without_hidden_anchors"] += 1
            continue
        value = np.mean(hidden[first:last], axis=0, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("pooled hidden cannot be represented as finite float32")
        events.append((start, end, r, q))
        pooled.append(value)
    centers = []
    for index in range(2, len(events) - 2):
        window = events[index - 2:index + 3]
        if all(b[0] == a[1] and b[2] - a[2] == step and b[3] - a[3] == step
               for a, b in zip(window, window[1:])):
            centers.append(index)
    centers = np.asarray(centers, dtype=np.int64)
    starts = np.asarray([event[0] for event in events], dtype=np.int64)
    ends = np.asarray([event[1] for event in events], dtype=np.int64)
    currents = np.asarray([np.median(current[events[i][0]:events[i][1]].astype(np.float64))
                           for i in centers], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        currents = currents.astype(np.float32)
    if not np.isfinite(currents).all():
        raise ValueError("center currents cannot be represented as finite float32")
    features = {"features": np.asarray(pooled, dtype=np.float32).reshape(-1, hidden.shape[1]),
                "centers": centers, "center_ids": starts[centers], "currents": currents,
                "event_starts": starts, "event_ends": ends}
    labels = []
    for index in centers:
        center_ref = events[index][2]
        kmer = reference[center_ref - 2:center_ref + 3]
        if len(kmer) != 5 or any(base not in "ACGT" for base in kmer):
            raise ValueError("eligible center has invalid reference-direction 5-mer")
        code = 0
        for base in kmer:
            code = 4 * code + "ACGT".index(base)
        labels.append(code)
    _validate_features(features)
    stats.update(complete_events=len(events), eligible_centers=len(centers))
    coordinates = {"event_reference_positions": [event[2] for event in events],
                   "event_query_positions": [event[3] for event in events],
                   "center_reference_positions": [events[i][2] for i in centers]}
    return features, np.asarray(labels, dtype=np.int64), stats, coordinates


def prepare_aligned_corpus(source_manifest, output_dir, *, max_reads_per_split=None,
                           allow_calibration_reads=False) -> dict[str, Any]:
    """Write a new event corpus from calibrated records and supplied dense hidden.

    The debug limit is deterministic and applies after the complete source
    manifest's identity check, before calibration exclusions and event filters.
    """
    if not isinstance(allow_calibration_reads, bool):
        raise ValueError("allow_calibration_reads must be boolean")
    if max_reads_per_split is not None:
        _integer(max_reads_per_split, "max_reads_per_split", 1)
    source_path = Path(source_manifest).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    files = _Files()
    source_hash = files.digest(source_path)
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_hash:
        raise ValueError("source manifest changed during preparation")
    source = _parse(source_bytes, "source manifest")
    if (type(source.get("schema_version")) is not int or source["schema_version"] != 1
            or source.get("kind") != "nanosignalalign_hidden" or source.get("status") != "complete"):
        raise ValueError("expected complete nanosignalalign_hidden source manifest schema_version=1")
    geometry = source.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("source requires geometry")
    stride = _integer(geometry.get("model_stride_samples"), "model_stride_samples", 1)
    origin = _integer(geometry.get("frame_origin_samples", 0), "frame_origin_samples", 0)
    if origin >= stride:
        raise ValueError("frame_origin_samples must be less than model_stride_samples")
    geometry = {"model_stride_samples": stride, "frame_origin_samples": origin}
    current_spec = source.get("current")
    if (not isinstance(current_spec, dict) or not isinstance(current_spec.get("normalization"), str)
            or current_spec["normalization"] not in {"input_signal", "stone_unclamped"}):
        raise ValueError("current.normalization must be input_signal or stone_unclamped")
    _string(current_spec.get("units"), "current.units")
    samples = source.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("source samples must be a nonempty list")
    physical, read_ids, references = {}, set(), set()
    normalized = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be an object")
        read = _string(sample.get("read_id"), "read_id")
        physical_read = _string(sample.get("physical_read_id"), "physical_read_id")
        split = sample.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"unsupported split: {split}")
        if physical_read in physical:
            message = "physical-read split leakage" if physical[physical_read] != split else "duplicate physical read violates one-chunk-per-read policy"
            raise ValueError(f"{message}: {physical_read}")
        if read in read_ids:
            raise ValueError(f"duplicate read_id: {read}")
        physical[physical_read] = split
        read_ids.add(read)
        aligned_path = _path(sample.get("aligned_jsonl"), source_path.parent, "aligned_jsonl")
        index = _integer(sample.get("record_index"), "record_index", 0)
        if (aligned_path, index) in references:
            raise ValueError("duplicate aligned JSONL record reference")
        references.add((aligned_path, index))
        hidden_path = _path(sample.get("hidden_path"), source_path.parent, "hidden_path")
        _hash(sample.get("hidden_sha256"), "hidden_sha256")
        if current_spec["normalization"] == "stone_unclamped" and "current_ref" not in sample:
            raise ValueError("stone_unclamped requires an explicit current_ref for every sample")
        normalized.append({**sample, "_aligned_path": aligned_path, "_hidden_path": hidden_path})
    selected = []
    for split in sorted(SPLITS):
        rows = sorted((row for row in normalized if row["split"] == split), key=lambda row: row["read_id"])
        selected.extend(rows if max_reads_per_split is None else rows[:max_reads_per_split])

    # First pass retains metadata and hashes only, never all inline waveforms.
    # Gather calibration reads across files before accepting any record.
    record_metadata, calibrations, calibration_cache = {}, {}, {}
    calibration_reads, explicitly_calibration = set(), set()
    for sample, record in _selected_records(selected, files):
        descriptor, ids, role = _calibration(record, sample["_aligned_path"].parent, files, calibration_cache)
        calibrations[descriptor["path"]] = descriptor
        calibration_reads.update(ids)
        if role == "calibration":
            explicitly_calibration.add(sample["physical_read_id"])
        record_metadata[(sample["_aligned_path"], sample["record_index"])] = (
            descriptor, role, _record_digest(record),
        )
    calibration_reads.update(explicitly_calibration)
    output.mkdir(parents=True, exist_ok=False)
    output_rows, totals, split_counts = [], Counter(), Counter()
    hidden_dim = None
    # Second streaming pass processes each selected record independently.
    for sample, record in _selected_records(selected, files):
        calibration, role, record_hash = record_metadata[(sample["_aligned_path"], sample["record_index"])]
        if _record_digest(record) != record_hash:
            raise ValueError("aligned record changed between preparation passes")
        physical_read = sample["physical_read_id"]
        is_calibration = physical_read in calibration_reads
        if is_calibration and not allow_calibration_reads:
            totals["omitted_calibration_reads"] += 1
            continue
        hidden_path = sample["_hidden_path"]
        if files.digest(hidden_path) != sample["hidden_sha256"]:
            raise ValueError(f"hidden checksum mismatch: {hidden_path}")
        hidden = files.array(hidden_path)
        if hidden.ndim != 2 or hidden.shape[1] < 1:
            raise ValueError("hidden must have nonempty feature dimension and shape [frames,hidden_dim]")
        original_shape = list(hidden.shape)
        hidden_slice = sample.get("hidden_slice", [0, len(hidden)])
        if not isinstance(hidden_slice, list) or len(hidden_slice) != 2:
            raise ValueError("hidden_slice must be [start,end]")
        first = _integer(hidden_slice[0], "hidden_slice start", 0)
        last = _integer(hidden_slice[1], "hidden_slice end", 0)
        if not 0 <= first < last <= len(hidden):
            raise ValueError("hidden_slice must select a nonempty in-range interval")
        hidden = _real(hidden[first:last], "hidden", 2)
        signal, signal_descriptor = _signal(record, sample["_aligned_path"].parent, files)
        expected_frames = len(range(origin, len(signal), stride))
        if len(hidden) != expected_frames:
            raise ValueError(f"hidden frame count {len(hidden)} disagrees with signal/stride/origin ({expected_frames})")
        if hidden_dim is not None and hidden_dim != hidden.shape[1]:
            raise ValueError("hidden dimensions differ across source records")
        hidden_dim = hidden.shape[1]
        if "current_ref" in sample:
            current, current_descriptor = _signal_ref(sample["current_ref"], source_path.parent, files, "current_ref")
            if len(current) != len(signal):
                raise ValueError("current_ref and input signal lengths disagree")
        else:
            current, current_descriptor = signal, dict(signal_descriptor)
        features, labels, stats, coordinates = _events(
            record, hidden, len(signal), current, geometry, calibration["query_orientation"],
        )
        totals.update(stats)
        if not len(labels):
            totals["omitted_reads_without_eligible_centers"] += 1
            continue
        key = hashlib.sha256((physical_read + "\0" + sample["read_id"]).encode()).hexdigest()[:24]
        feature_path = Path("features") / sample["split"] / f"{key}.npz"
        truth_path = Path("truth") / sample["split"] / f"{key}.npz"
        for relative, arrays in ((feature_path, features), (truth_path, {"labels": labels})):
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
        provenance = {"aligned_jsonl": str(sample["_aligned_path"]),
                      "aligned_jsonl_sha256": files.digest(sample["_aligned_path"]),
                      "record_index": sample["record_index"], "aligned_record_sha256": record_hash,
                      "read_id": sample["read_id"], "physical_read_id": physical_read,
                      "hidden_path": str(hidden_path), "hidden_sha256": sample["hidden_sha256"],
                      "hidden_original_shape": original_shape, "hidden_slice": [first, last],
                      "signal_ref": signal_descriptor, "current_ref": current_descriptor,
                      "normal_mode": record.get("normal_mode"),
                      "signal_alignment": dict(record["signal_alignment"]),
                      "calibration": calibration, "is_calibration_read": is_calibration,
                      "sample_coordinate_system": "zero_based_half_open_relative_to_effective_signal",
                      **coordinates}
        output_rows.append({"read_id": physical_read, "physical_read_id": physical_read,
                            "record_id": sample["read_id"], "split": sample["split"],
                            "features_path": feature_path.as_posix(), "truth_path": truth_path.as_posix(),
                            "features_sha256": file_sha256(output / feature_path),
                            "truth_sha256": file_sha256(output / truth_path), "n_centers": len(labels),
                            "source": provenance})
        split_counts[sample["split"]] += 1
    if not output_rows:
        raise ValueError("no eligible five-event centers; incomplete output has no complete manifest")
    output_rows.sort(key=lambda row: (row["split"], row["read_id"]))
    manifest = {
        "schema_version": 2, "kind": "kmer_event_context", "status": "complete",
        "hidden_dim": hidden_dim, "num_classes": 1024, "max_window": 5,
        "source_manifest": str(source_path), "source_manifest_sha256": source_hash,
        "source_geometry": geometry, "source_model_provenance": source.get("model"),
        "source_representation_contract": source.get("representation_contract"),
        "register": {"status": "phase_calibrated", "label_coordinate_system": "reference",
                     "kmer_pattern": "XXNXX", "additional_offset_bases": 0,
                     "independently_validated": False,
                     "calibrations": [calibrations[path] for path in sorted(calibrations)]},
        "current": dict(current_spec), "information_boundary": INFORMATION_BOUNDARY,
        "selection": {
            "event_pooling": "arithmetic mean of hidden frames whose origin+j*stride anchors fall in calibrated [start,end)",
            "frame_assignment": "anchor membership, not receptive-field purity or overlap weighting",
            "center_current": "median of supplied current over the central calibrated event only",
            "event_order": "signal time ascending; reference/query increments +1 for direct and -1 for reverse",
            "labels": "ACGT lexicographic encoding of ref[center_ref-2:center_ref+3], always reference direction",
            "common_centers_for_windows": [1, 3, 5], "missing_states_break_windows": True,
            "one_chunk_per_physical_read": True, "additional_offset_bases": 0,
            "truth_dependent_filtering": "exact ACGT alignments, calibrated nonempty spans, hidden-anchor support, consecutive query/reference and signal intervals",
            "allow_calibration_reads": allow_calibration_reads,
            "calibration_read_policy": "union of selected calibrations' stats.calibration_read_ids and calibration annotations",
            "calibration_overlap_allowed": allow_calibration_reads,
            "calibrated_boundaries_are_not_independently_validated": True,
        },
        "smoke_limit": {"max_reads_per_split": max_reads_per_split, "enabled": max_reads_per_split is not None,
                        "selection": "read_id lexicographic order per split before calibration exclusions and event filtering",
                        "limitation": "Debug subset, not a representative scientific evaluation"},
        "counts": {"source_reads": len(samples), "selected_reads": len(selected),
                   "written_reads": len(output_rows), "written_reads_by_split": dict(split_counts),
                   **dict(totals)},
        "samples": output_rows,
    }
    files.verify()
    _write_json(output / "manifest.json", manifest)
    return manifest
