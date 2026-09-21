"""Read frozen representations without changing labels, splits, or feature scale.

Manifest rows are observations; ``sample_index`` addresses rows in each source
NPY.  It is deliberately independent of the current CSV row order.  Probe
partitions must be explicit: the corpus's ``split=evaluation`` is not a
training/validation/test partition.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation", "test")
_KMER = re.compile(r"[ACGT]{5}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"Expected an input file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _nonempty_string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a nonempty string")
    return value


def _resolve(value: Any, base: Path, what: str) -> str:
    path = Path(_nonempty_string(value, what)).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON and resolve all input paths relative to its directory.

    Extra runner settings are preserved.  Paths in the returned configuration
    are strings so the resolved protocol can be archived directly as JSON.
    """
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    _nonempty_string(config.get("model_id"), "model_id")
    _nonempty_string(config.get("scheme"), "scheme")
    config["source_run"] = _resolve(config.get("source_run"), path.parent, "source_run")
    config.setdefault("split_column", "probe_split")
    config.setdefault("group_column", "physical_read_id")
    config.setdefault("label_column", "kmer")
    _validate_columns(config)
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages or any(
        not isinstance(stage, str) or not _NAME.fullmatch(stage) for stage in stages
    ) or len(set(stages)) != len(stages):
        raise ValueError("stages must be a nonempty list of unique stage names")
    domains = config.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("domains must be a nonempty list")
    names = set()
    for domain in domains:
        if not isinstance(domain, dict):
            raise ValueError("Every domain must be an object")
        name = domain.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise ValueError(f"Invalid or duplicate domain name: {name!r}")
        names.add(name)
        domain["manifest"] = _resolve(domain.get("manifest"), path.parent, f"{name}.manifest")
        representations = domain.get("representations")
        if not isinstance(representations, dict) or set(representations) != set(stages):
            raise ValueError(f"{name}: representations must contain exactly configured stages")
        domain["representations"] = {
            stage: _resolve(representations[stage], path.parent, f"{name}.{stage}")
            for stage in stages
        }
    if "seed" in config and (isinstance(config["seed"], bool) or not isinstance(config["seed"], int)):
        raise ValueError("seed must be an integer")
    if "regularization" in config:
        values = config["regularization"]
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError("regularization must contain positive finite numbers")
    if "bootstrap_repeats" in config:
        repeats = config["bootstrap_repeats"]
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
            raise ValueError("bootstrap_repeats must be a nonnegative integer")
    return config


def _validate_columns(config: dict[str, Any]) -> None:
    if config.get("split_column", "probe_split") != "probe_split":
        raise ValueError("split_column must be probe_split; split=evaluation is not a probe partition")
    if config.get("group_column", "physical_read_id") != "physical_read_id":
        raise ValueError("group_column must be physical_read_id to preserve read grouping")
    if config.get("label_column", "kmer") != "kmer":
        raise ValueError("label_column must be kmer for the 5-mer probe protocol")


@dataclass
class Domain:
    """One domain, retaining CSV observation order and NPY row addresses."""

    name: str
    manifest_path: Path
    labels: np.ndarray
    groups: np.ndarray
    splits: np.ndarray
    sample_ids: np.ndarray
    sample_indices: np.ndarray
    feature_paths: dict[str, Path]
    _manifest_signature: tuple[int, int, int, int, int] | None = field(default=None, repr=False)

    def load_features(self, stage: str) -> np.ndarray:
        """Return finite floating N x D features in this manifest's row order.

        Full sequential manifests retain a read-only memory map. Subsets and
        permutations use explicit indexing; no normalization or dtype cast is
        applied. No assumption about feature width or class count is made.
        """
        if stage not in self.feature_paths:
            raise ValueError(f"{self.name}: unknown feature stage {stage!r}")
        path = Path(self.feature_paths[stage])
        before = _signature(path)
        try:
            source = np.load(path, mmap_mode="r", allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ValueError(f"{self.name}/{stage}: cannot load numeric NPY {path}: {exc}") from exc
        if not isinstance(source, np.ndarray):
            if hasattr(source, "close"):
                source.close()
            raise ValueError(f"{self.name}/{stage}: expected one NPY array, not an archive")
        if source.ndim != 2 or source.shape[1] == 0 or not np.issubdtype(source.dtype, np.floating):
            raise ValueError(f"{self.name}/{stage}: expected a floating two-dimensional N x D array")
        indices = np.asarray(self.sample_indices)
        if indices.ndim != 1 or len(indices) != len(self.labels) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{self.name}: sample_indices must be a one-dimensional integer array")
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= source.shape[0]):
            raise ValueError(f"{self.name}/{stage}: sample_index outside NPY row bounds")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{self.name}: duplicate sample_index")
        sequential = np.array_equal(indices, np.arange(len(indices), dtype=indices.dtype))
        values = source[:len(indices)] if sequential else source[indices]
        # Avoid a full N x D temporary for the finite-value check.
        for start in range(0, len(values), 8192):
            if not np.isfinite(values[start:start + 8192]).all():
                raise ValueError(f"{self.name}/{stage}: nonfinite selected representation values")
        if _signature(path) != before:
            raise ValueError(f"Input changed while loading: {path}")
        values.flags.writeable = False
        return values


def _read_domain(spec: dict[str, Any], stages: list[str]) -> Domain:
    name = spec["name"]
    path = Path(spec["manifest"]).resolve()
    before = _signature(path)
    required = ("sample_index", "sample_id", "kmer", "physical_read_id", "probe_split")
    labels, groups, splits, sample_ids, indices = [], [], [], [], []
    seen_ids, seen_indices = set(), set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{name}: missing or duplicate manifest column names")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{name}: manifest missing required columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[key] is None for key in required):
                raise ValueError(f"{name}:{line}: malformed CSV row")
            label, group, split, sample_id = (
                row["kmer"], row["physical_read_id"], row["probe_split"], row["sample_id"]
            )
            if not _KMER.fullmatch(label):
                raise ValueError(f"{name}:{line}: kmer must be exactly five uppercase A/C/G/T bases")
            if not group.strip() or not sample_id.strip():
                raise ValueError(f"{name}:{line}: empty physical_read_id or sample_id")
            if split not in SPLITS:
                raise ValueError(f"{name}:{line}: unknown probe_split {split!r}")
            text_index = row["sample_index"]
            if not re.fullmatch(r"[0-9]+", text_index):
                raise ValueError(f"{name}:{line}: sample_index must be a nonnegative integer")
            index = int(text_index)
            if index > np.iinfo(np.int64).max:
                raise ValueError(f"{name}:{line}: sample_index exceeds int64")
            if sample_id in seen_ids or index in seen_indices:
                raise ValueError(f"{name}:{line}: duplicate sample_id or sample_index")
            if "corpus" in row and row["corpus"] != name:
                raise ValueError(f"{name}:{line}: manifest corpus does not match configured domain")
            seen_ids.add(sample_id)
            seen_indices.add(index)
            labels.append(label)
            groups.append(group)
            splits.append(split)
            sample_ids.append(sample_id)
            indices.append(index)
    if not labels:
        raise ValueError(f"{name}: empty manifest")
    if _signature(path) != before:
        raise ValueError(f"Manifest changed while loading: {path}")
    return Domain(
        name=name, manifest_path=path, labels=np.asarray(labels, dtype=str),
        groups=np.asarray(groups, dtype=str), splits=np.asarray(splits, dtype=str),
        sample_ids=np.asarray(sample_ids, dtype=str), sample_indices=np.asarray(indices, dtype=np.int64),
        feature_paths={stage: Path(spec["representations"][stage]).resolve() for stage in stages},
        _manifest_signature=before,
    )


def _validate_groups(domains: dict[str, Domain]) -> dict[str, set[str]]:
    assignments: dict[str, tuple[str, str]] = {}
    sample_ids: dict[str, str] = {}
    membership: dict[str, set[str]] = defaultdict(set)
    for name, domain in domains.items():
        lengths = {len(domain.labels), len(domain.groups), len(domain.splits), len(domain.sample_ids), len(domain.sample_indices)}
        if lengths != {len(domain.labels)} or not len(domain.labels):
            raise ValueError(f"{name}: domain observation arrays differ in length or are empty")
        for group, split, sample_id in zip(domain.groups, domain.splits, domain.sample_ids):
            if split not in SPLITS:
                raise ValueError(f"{name}: unknown probe_split {split!r}")
            prior = assignments.setdefault(str(group), (str(split), name))
            if prior[0] != split:
                raise ValueError(f"physical_read_id {group!r} crosses probe splits: {prior[1]}/{prior[0]} and {name}/{split}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id across observations: {sample_id!r} in {sample_ids[sample_id]} and {name}")
            sample_ids[str(sample_id)] = name
            membership[str(group)].add(name)
    return membership


def load_domains(config: dict[str, Any]) -> dict[str, Domain]:
    """Read observations and reject within/across-domain physical-read leakage."""
    _validate_columns(config)
    result = {}
    for spec in config["domains"]:
        if spec["name"] in result:
            raise ValueError(f"Duplicate domain name: {spec['name']!r}")
        result[spec["name"]] = _read_domain(spec, config["stages"])
    if not result:
        raise ValueError("No domains configured")
    _validate_groups(result)
    return result


def _fingerprint(path: Path, expected: tuple[int, int, int, int, int] | None = None) -> dict[str, Any]:
    before = _signature(path)
    if expected is not None and before != expected:
        raise ValueError(f"Input changed after it was loaded: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if _signature(path) != before:
        raise ValueError(f"Input changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before[2]}


def audit_domains(domains: dict[str, Domain]) -> dict[str, Any]:
    """Validate feature arrays and report support, grouped splits, and hashes.

    Every unique input path is hashed once per audit. Missing classes in a split
    are reported, never silently removed or reassigned. In particular, training
    coverage is observed from the data rather than assumed to be 1024 classes.
    """
    if not domains:
        raise ValueError("No domains to audit")
    membership = _validate_groups(domains)
    labels = sorted(set().union(*(set(domain.labels) for domain in domains.values())))
    fingerprints: dict[Path, dict[str, Any]] = {}
    hashed_signatures: dict[Path, tuple[int, int, int, int, int]] = {}

    def record_input(path: Path, expected: tuple[int, int, int, int, int] | None = None) -> None:
        before = _signature(path)
        if expected is not None and before != expected:
            raise ValueError(f"Input changed after it was loaded: {path}")
        if path in fingerprints:
            if before != hashed_signatures[path]:
                raise ValueError(f"Input changed during audit: {path}")
        else:
            fingerprints[path] = _fingerprint(path, before)
            hashed_signatures[path] = before

    reports = {}
    widths: dict[str, int] = {}
    stage_sets = {frozenset(domain.feature_paths) for domain in domains.values()}
    if len(stage_sets) != 1 or not next(iter(stage_sets)):
        raise ValueError("Domains must have the same nonempty feature stages")
    train_sets, test_sets = [], []
    for name, domain in domains.items():
        manifest_path = Path(domain.manifest_path).resolve()
        record_input(manifest_path, domain._manifest_signature)
        support = {split: dict(sorted(Counter(domain.labels[domain.splits == split]).items())) for split in SPLITS}
        train_sets.append(set(support["train"]))
        test_sets.append(set(support["test"]))
        features = {}
        for stage, source_path in domain.feature_paths.items():
            source_path = Path(source_path).resolve()
            before = _signature(source_path)
            values = domain.load_features(stage)
            if stage in widths and widths[stage] != values.shape[1]:
                raise ValueError(f"{name}/{stage}: feature width differs across domains")
            widths[stage] = int(values.shape[1])
            features[stage] = {"path": str(source_path), "shape": list(values.shape), "dtype": str(values.dtype)}
            del values
            record_input(source_path, before)
        domain_labels = sorted(set(domain.labels))
        reports[name] = {
            "samples": len(domain.labels), "groups": len(set(domain.groups)),
            "labels": domain_labels, "n_classes": len(domain_labels),
            "split_counts": {split: int(np.count_nonzero(domain.splits == split)) for split in SPLITS},
            "group_counts_by_split": {split: len(set(domain.groups[domain.splits == split])) for split in SPLITS},
            "support_by_split": support,
            "class_counts_by_split": {split: len(support[split]) for split in SPLITS},
            "missing_labels_by_split": {split: sorted(set(labels) - set(support[split])) for split in SPLITS},
            "features": features,
        }
    common_test = sorted(set.intersection(*test_sets))
    common_train = sorted(set.intersection(*train_sets))
    return {
        "domains": reports, "labels": labels, "n_classes": len(labels),
        "common_test_labels": common_test, "n_common_test_labels": len(common_test),
        "common_train_labels": common_train, "n_common_train_labels": len(common_train),
        "group_split_conflicts": [],
        "shared_groups_across_domains": sum(len(names) > 1 for names in membership.values()),
        "group_column": "physical_read_id", "split_column": "probe_split", "label_column": "kmer",
        "inputs": [fingerprints[path] for path in sorted(fingerprints, key=str)],
    }
