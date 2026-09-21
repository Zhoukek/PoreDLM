#!/usr/bin/env python3
"""Prepare, verify and plot a frozen model-independent signal corpus.

The only runnable entry point in this directory. Prep/Align are imported from
project/script; extraction, RNA adaptation and deterministic sample selection
are kept here. Signal arrays and manifests stay in their dataset directories.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
import hashlib
import heapq
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import tempfile
import time
try:
    import tomllib
except ModuleNotFoundError:  # Apple extraction also runs in existing Python 3.10 POD5 environments.
    from pip._vendor import tomli as tomllib
from importlib.metadata import version, PackageNotFoundError

import numpy as np

ARRAY_NAMES = ('chunks', 'moves', 'move_lengths', 'move_strides', 'references', 'reference_lengths')


def dump(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure(project):
    for package in ('NanoSignalPrep', 'NanoSignalAlign', 'NanoRepDist'):
        sys.path.insert(0, str(Path(project) / 'script' / package / 'src'))


def prepare(project, root, dataset, seed, pool_size):
    configure(project)
    import numpy as np
    from nanosignalprep.array_chunks import ChunkArrayOptions, BamNameIndex, _build_record, _validate_shapes
    from nanosignalprep.validation import validate_record, validate_jsonl
    from nanosignalprep.errors import ContractError as PrepContractError
    from nanosignalalign.calibration import CalibrationOptions, calibrate_jsonl
    from nanosignalalign.alignment import align_jsonl
    from nanosignalalign.tables import write_kmer_summary
    from nanosignalalign.validation import validate_jsonl as validate_aligned

    name = dataset['name']
    target = Path(root) / name
    # A completed immutable dataset may be reused to add missing domains later.
    complete = target / 'dataset.complete.json'
    if complete.exists():
        old = json.loads(complete.read_text())
        if old['seed'] != seed or old['pool_size_requested'] != pool_size or old['dataset'] != dataset:
            raise RuntimeError(f'{name}: existing corpus configuration differs; use a new output directory')
        for rel, checksum in old['sha256'].items():
            if digest(target / rel) != checksum:
                raise RuntimeError(f'{name}: checksum changed: {rel}')
        print(f'{name}: verified existing dataset', flush=True)
        return old
    if target.exists():
        raise RuntimeError(f'{target} already exists without completion; inspect it or choose a new output directory')
    prep_dir = target / 'prep'
    aligned_dir = target / 'aligned'
    prep_dir.mkdir(parents=True)
    aligned_dir.mkdir()
    start_time = time.time()
    shards = dataset['shards']
    if not shards:
        raise ValueError(f'{name}: no shards')
    widths = []
    for shard in shards:
        a = np.load(shard['chunks'], mmap_mode='r', allow_pickle=False)
        widths.append(a.shape[1])
    # Signals are frozen locally, retaining the stored dtype, scale and complete chunk context.
    dtype = np.result_type(*[np.load(s['chunks'], mmap_mode='r', allow_pickle=False).dtype for s in shards])
    signal_file = prep_dir / 'signals.npy'
    signals = np.lib.format.open_memmap(signal_file, mode='w+', dtype=dtype, shape=(pool_size, max(widths)))
    seen_reads = set()
    written = 0
    rejected = {}
    sources = []
    jsonl = prep_dir / 'chunks.jsonl'
    with jsonl.open('w') as out, (prep_dir / 'selection.tsv').open('w', newline='') as selection, (prep_dir / 'rejected.jsonl').open('w') as bad:
        writer = csv.DictWriter(selection, fieldnames=['record_index','read_id','physical_read_id','source_shard','source_row','signal_length'], delimiter='\t')
        writer.writeheader()
        for shard_no, shard in enumerate(shards):
            arrays = {key: np.load(shard[key], mmap_mode='r', allow_pickle=False) for key in ARRAY_NAMES}
            _validate_shapes(arrays)
            n = len(arrays['chunks'])
            quota = pool_size // len(shards) + (shard_no < pool_size % len(shards))
            rng_seed = int(hashlib.sha256(f'{seed}:{name}:{shard_no}'.encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(rng_seed)
            # Oversample candidate rows to replace duplicate physical reads or invalid mappings.
            rows = rng.choice(n, size=min(n, quota * 4), replace=False).tolist()
            wanted = set(rows)
            metadata = {}
            with Path(shard['summary']).open(newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row, item in enumerate(reader):
                    if row in wanted:
                        metadata[row] = item
                    if len(metadata) == len(wanted):
                        break
            if len(metadata) != len(wanted):
                raise RuntimeError(f'{name}: summary lacks requested array rows in {shard["summary"]}')
            options = ChunkArrayOptions(**{key: Path(shard[key]) for key in ARRAY_NAMES}, summary=Path(shard['summary']), bam=Path(shard['bam']), signal_storage='reference', label=dataset.get('label','UNSPECIFIED'), negative_query_orientation=shard.get('negative_query_orientation','reverse'))
            print(f'{name}: shard {shard_no+1}/{len(shards)}, indexing BAM, candidate rows={len(rows)}', flush=True)
            index = BamNameIndex(options.bam)
            accepted = 0
            try:
                for row in rows:
                    if accepted >= quota:
                        break
                    meta = metadata[row]
                    try:
                        bam = index.find(meta['read_id'])
                        record = _build_record(row, *[arrays[key] for key in ARRAY_NAMES], meta, bam, options)
                        if dataset.get('molecule') == 'rna':
                            record = restore_rna_biological_reference(record, options.negative_query_orientation)
                        match_fraction = record['align']['match_string'].count('|') / max(1, len(record['ref']))
                        if match_fraction < 0.90:
                            raise ValueError(f'reference/query exact-match fraction {match_fraction:.4f} < 0.90')
                        errors = validate_record(record)
                        if errors:
                            raise ValueError('; '.join(errors))
                    except (PrepContractError, ValueError, RuntimeError, KeyError, OSError) as exc:
                        reason = type(exc).__name__
                        rejected[reason] = rejected.get(reason,0)+1
                        bad.write(json.dumps({'source_shard':str(Path(shard['chunks']).parent),'source_row':row,'read_id':meta.get('read_id'),'reason':str(exc)})+'\n')
                        continue
                    physical = record['physical_read_id']
                    if physical in seen_reads:
                        rejected['duplicate_physical_read'] = rejected.get('duplicate_physical_read',0)+1
                        continue
                    seen_reads.add(physical)
                    length = record['signal_ref']['length']
                    signals[written, :length] = arrays['chunks'][row,:length]
                    signals[written, length:] = 0
                    record['meta']['source_signal_ref'] = record['signal_ref']
                    record['meta']['source_summary'] = shard['summary']
                    record['meta']['source_shard'] = str(Path(shard['chunks']).parent)
                    record['meta']['corpus'] = name
                    record['meta']['label_source'] = 'reference_array_with_bam_cigar'
                    record['signal_ref'] = {'path': str(signal_file.resolve()), 'row': written, 'start': 0, 'end':length, 'length':length}
                    out.write(json.dumps(record, separators=(',',':'))+'\n')
                    writer.writerow({'record_index':written,'read_id':record['read_id'],'physical_read_id':physical,'source_shard':str(Path(shard['chunks']).parent),'source_row':row,'signal_length':length})
                    written += 1
                    accepted += 1
            finally:
                index.close()
            sources.append({'shard':shard,'array_rows':n,'candidate_rows':len(rows),'accepted':accepted,'files':{k:{'bytes':Path(v).stat().st_size,'mtime_ns':Path(v).stat().st_mtime_ns} for k,v in shard.items() if k in ARRAY_NAMES or k in ('bam','summary')}})
            print(f'{name}: accepted {accepted}, total {written}', flush=True)
    if not written:
        raise RuntimeError(f'{name}: no valid records')
    signals.flush()
    del signals
    validated = validate_jsonl(jsonl).as_dict()
    dump(prep_dir / 'validation.json', validated)
    if validated['status'] != 'pass':
        raise RuntimeError(f'{name}: Prep validation failed')
    dump(prep_dir / 'export_summary.json', {'seed':seed,'pool_size_requested':pool_size,'records':written,'signal_array_allocated_rows':pool_size,'physical_reads':len(seen_reads),'rejected':rejected,'sources':sources,'normalization':'precomputed; unchanged from input arrays','sampling':'fixed-seed uniform shard selection, equal shard quota, random array rows, at most one chunk per physical read'})
    print(f'{name}: calibrating {written} chunks', flush=True)
    calibration = aligned_dir / 'calibration.json'
    options = CalibrationOptions(seed=seed, events_per_read=16, calibration_fraction=0.2, min_test_predictions=200, fail_fast=True)
    payload = calibrate_jsonl(jsonl, calibration, options)
    write_kmer_summary(payload, aligned_dir / 'kmers.tsv')
    warnings = []
    if payload.get('confidence') != 'high':
        warnings.append('Phase confidence is ' + str(payload.get('confidence')))
    if payload.get('selected_shift') in (-6, 6):
        warnings.append('Selected phase reaches search boundary; inspect before interpreting representations')
    if written < pool_size:
        warnings.append(f'Unique valid physical reads available: {written}/{pool_size} requested')
    dump(aligned_dir / 'qc.json', {'warnings':warnings,'selected_metrics':payload.get('selected_metrics'),'confidence':payload.get('confidence'),'scope':'phase calibration on this dataset, not an independent alignment accuracy estimate'})
    aligned = aligned_dir / 'chunks.jsonl'
    result = align_jsonl(jsonl, calibration, aligned, fail_fast=True, summary_json=aligned_dir/'summary.json')
    validation = validate_aligned(aligned).as_dict()
    dump(aligned_dir / 'validation.json', validation)
    if validation['status'] != 'pass':
        raise RuntimeError(f'{name}: Align validation failed')
    output = {'dataset':dataset,'seed':seed,'pool_size_requested':pool_size,'records':written,'rejected':rejected,'calibration':{k:payload.get(k) for k in ('selected_orientation','selected_shift','confidence','selected_metrics')},'seconds':round(time.time()-start_time,2),'sha256':{str(p.relative_to(target)):digest(p) for p in sorted(target.rglob('*')) if p.is_file()}}
    dump(complete, output)
    print(f'{name}: finished {output["seconds"]} s; calibration={output["calibration"]}', flush=True)
    return output


def restore_rna_biological_reference(record, negative_query_orientation='reverse'):
    """Adapt official NanoSignalPrep record from reversed RNA target arrays."""
    from nanosignalprep.array_chunks import _bam_alignment

    result = dict(record)
    result['ref'] = str(record['ref'])[::-1]
    query_spans = [list(span) for span in reversed(record['base_sample_span_seq'])]
    alignment, ref_spans = _bam_alignment(
        str(record['meta']['cigar']), result['ref'], str(record['seq']),
        query_spans, str(record['strand']), negative_query_orientation,
    )
    result['align'] = alignment
    result['base_sample_span_seq'] = query_spans
    result['base_sample_span_ref'] = ref_spans
    result['meta'] = {
        **record['meta'],
        'reference_array_orientation': 'reverse_biological',
        'sequence_orientation': 'biological_5_to_3',
        'signal_query_orientation_baseline': 'reverse',
        'corpus_adapter': 'rna_reference_reverse_v1',
        'signal_and_moves_unchanged': True,
    }
    return result

KMERS = tuple(map("".join, itertools.product("ACGT", repeat=5)))
FIELDS = ("sample_index sample_id kmer corpus read_id physical_read_id "
          "ref_start ref_end center_ref signal_start signal_end "
          "center_signal_start center_signal_end signal_query_orientation "
          "sequence_offset_bases source_jsonl record_index signal_path signal_row "
          "signal_ref_start signal_ref_end signal_length split probe_split reference_name "
          "target_ref_start target_ref_end strand").split()


def _digest(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def _split(physical, seed):
    value = int(_digest(seed, "split", physical)[:16], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def _physical(record):
    value = record.get("physical_read_id")
    if value:
        return str(value)
    read_id = str(record.get("read_id", ""))
    parts = read_id.rsplit(":", 2)
    return parts[0] if len(parts) == 3 else read_id


def _portable(path, root):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _resolve(path, base):
    value = Path(path)
    return value if value.is_absolute() else base / value


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _atomic_table(path, rows, fields, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


class Signals:
    def __init__(self, source):
        self.source = source
        self.arrays = {}

    def get(self, record):
        if isinstance(record.get("signal"), list):
            values = np.asarray(record["signal"], dtype=np.float32)
            info = ("", "", 0, len(values))
        else:
            ref = record.get("signal_ref", {})
            path = _resolve(ref["path"], self.source.parent).resolve()
            if path not in self.arrays:
                self.arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            array = self.arrays[path]
            row = int(ref["row"])
            if array.ndim != 2 or not 0 <= row < array.shape[0]:
                raise ValueError("signal_ref must index a row of a 2D array")
            start = int(ref.get("start", 0))
            end = int(ref.get("end", array.shape[1]))
            if not 0 <= start < end <= array.shape[1]:
                raise ValueError("signal_ref slice is outside its row")
            if "length" in ref and int(ref["length"]) != end - start:
                raise ValueError("signal_ref.length differs from slice length")
            values = array[row, start:end]
            info = (path, row, start, end)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("signal is empty, non-vector, or non-finite")
        return values, info


def _exact_windows(record, signal_length, stats=None):
    """Yield ref-start, kmer and five spans; reject internal indels/mismatches."""
    stats = stats if stats is not None else Counter()
    ref = str(record["ref"]).upper()
    seq = str(record["seq"]).upper()
    alignment = record["align"]
    ra, qa = str(alignment["ref_aligned"]).upper(), str(alignment["seq_aligned"]).upper()
    if not ra or len(ra) != len(qa) or ra.replace("-", "") != ref or qa.replace("-", "") != seq:
        raise ValueError("gapped alignment does not agree with ref/seq")
    spans = record["base_sample_span_ref_calibrated"]
    stats["reference_windows_total"] += max(0, len(ref) - 4)
    if len(spans) != len(ref):
        raise ValueError("calibrated spans must have one entry per reference base")
    annotation = record["signal_alignment"]
    orientation = annotation["query_orientation"]
    if annotation.get("stage") != "phase_calibrated" or orientation not in {"direct", "reverse"}:
        raise ValueError("record is missing valid phase calibration")
    run = 0
    ref_pos = -1
    for r, q in zip(ra, qa):
        if r != "-":
            ref_pos += 1
        run = run + 1 if r == q and r in "ACGT" else 0
        if run < 5:
            continue
        start = ref_pos - 4
        stats["exact_match_windows"] += 1
        window = spans[start:start + 5]
        if any(not isinstance(s, (list, tuple)) or len(s) != 2
               or any(type(x) is not int for x in s)
               or not 0 <= s[0] < s[1] <= signal_length for s in window):
            stats["invalid_span_windows"] += 1
            continue
        contiguous = all(a[1] == b[0] for a, b in zip(window, window[1:])) if orientation == "direct" else all(a[0] == b[1] for a, b in zip(window, window[1:]))
        if not contiguous:
            stats["nonadjacent_or_nonmonotonic_windows"] += 1
            continue
        stats["valid_windows"] += 1
        yield start, ref[start:start + 5], window


class Calibration:
    def __init__(self, source):
        self.source = source
        self.cache = {}

    def excluded(self, record):
        annotation = record.get("signal_alignment", {})
        role = annotation.get("calibration_role")
        if role == "calibration":
            return True
        raw_path = annotation.get("calibration_path")
        # Align writes roles and the exact physical-read list. Check both when
        # the artifact is present; keep working after a portable bundle moves.
        if raw_path:
            path = _resolve(raw_path, self.source.parent)
            if path not in self.cache:
                if path.exists():
                    document = json.loads(path.read_text())
                    self.cache[path] = set(map(str, document.get("stats", {}).get("calibration_read_ids", [])))
                else:
                    self.cache[path] = None
            ids = self.cache[path]
            if ids is not None and _physical(record) in ids:
                return True
        if role != "analysis":
            raise ValueError("calibration_role must be analysis or calibration")
        return False


def _collect(root, name, seed, cap):
    source = root / name / "aligned" / "chunks.jsonl"
    signals, calibration = Signals(source), Calibration(source)
    seen = defaultdict(set)
    banks = defaultdict(dict)
    heaps = defaultdict(list)
    counts = Counter()
    stats = Counter()
    with source.open() as handle:
        for record_index, line in enumerate(handle):
            if not line.strip():
                continue
            stats["records_seen"] += 1
            record = json.loads(line)
            if calibration.excluded(record):
                stats["calibration_records_excluded"] += 1
                continue
            physical = _physical(record)
            if not physical or not record.get("read_id"):
                raise ValueError(f"{source}:{record_index + 1}: missing read identity")
            try:
                values, info = signals.get(record)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                stats["invalid_signal_records"] += 1
                continue
            stats["analysis_records"] += 1
            annotation = record["signal_alignment"]
            rank_cache = {}
            for start, kmer, spans in _exact_windows(record, len(values), stats):
                counts[kmer] += 1
                seen[kmer].add(physical)
                if kmer not in rank_cache:
                    rank_cache[kmer] = int(_digest(seed, "read", name, kmer, physical), 16)
                rank = rank_cache[kmer]
                bank, heap = banks[kmer], heaps[kmer]
                if physical not in bank and len(bank) >= cap and rank >= -heap[0][0]:
                    continue
                signal_start = min(s[0] for s in spans)
                signal_end = max(s[1] for s in spans)
                sample_id = _digest(name, physical, str(record["read_id"]), kmer,
                                    start, start + 5, signal_start, signal_end)
                window_rank = int(_digest(seed, "window", sample_id), 16)
                if physical in bank and window_rank >= bank[physical][1]:
                    continue
                if physical not in bank:
                    if len(bank) == cap:
                        _, evicted = heapq.heappop(heap)
                        del bank[evicted]
                    heapq.heappush(heap, (-rank, physical))
                path, row, ref_start, ref_end = info
                metadata = dict(sample_index=-1, sample_id=sample_id, kmer=kmer,
                    corpus=name, read_id=str(record["read_id"]), physical_read_id=physical,
                    ref_start=start, ref_end=start + 5, center_ref=start + 2,
                    signal_start=signal_start, signal_end=signal_end,
                    center_signal_start=spans[2][0], center_signal_end=spans[2][1],
                    signal_query_orientation=annotation["query_orientation"],
                    sequence_offset_bases=int(annotation["offset_bases"]),
                    source_jsonl=_portable(source, root), record_index=record_index,
                    signal_path=_portable(path, root) if path else "", signal_row=row,
                    signal_ref_start=ref_start, signal_ref_end=ref_end,
                    signal_length=len(values), split="evaluation", probe_split=_split(physical, seed),
                    reference_name=record.get("ref_name", ""),
                    target_ref_start=record.get("target_ref_start", ""),
                    target_ref_end=record.get("target_ref_end", ""), strand=record.get("strand", ""))
                bank[physical] = (rank, window_rank, metadata)
    availability = {kmer: len(seen[kmer]) for kmer in KMERS}
    stats["alignment_rejected_windows"] = stats["reference_windows_total"] - stats["exact_match_windows"]
    stats["distinct_valid_physical_reads"] = len(set().union(*seen.values())) if seen else 0
    stats["calibration_artifacts_found"] = sum(v is not None for v in calibration.cache.values())
    stats["calibration_artifacts_missing"] = sum(v is None for v in calibration.cache.values())
    return banks, availability, counts, dict(stats)


def freeze(root: Path, names: list[str], seed=20260916, cap=100, minimum=1):
    """Stream each dataset, freeze shared 5-mers, and return a JSON-safe summary."""
    root = Path(root).resolve()
    if not names or len(set(names)) != len(names) or cap < 1 or not 1 <= minimum <= cap:
        raise ValueError("names must be unique/nonempty; 1 <= minimum <= cap")
    availability, windows, statistics = {}, {}, {}
    for name in names:
        banks, availability[name], windows[name], statistics[name] = _collect(root, name, seed, cap)
        candidate_path = root / name / "freeze_candidates.jsonl.partial"
        with candidate_path.open("w") as handle:
            for kmer in KMERS:
                for _, _, row in sorted(banks[kmer].values(), key=lambda item: (item[0], item[1])):
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        del banks
    balanced = {kmer: min(cap, *(availability[name][kmer] for name in names)) for kmer in KMERS}
    balanced = {kmer: n if n >= minimum else 0 for kmer, n in balanced.items()}
    coverage = []
    for name in names:
        for kmer in KMERS:
            count = availability[name][kmer]
            coverage.append(dict(corpus=name, kmer=kmer, valid_windows=windows[name][kmer],
                available_physical_reads=count, capped_available=min(cap, count),
                selected=balanced[kmer], missing=int(count == 0), below_cap=int(count < cap),
                below_minimum=int(count < minimum), common=int(balanced[kmer] > 0)))
    _atomic_table(root / "reports" / "coverage.tsv", coverage, list(coverage[0]), "\t")
    for name in names:
        path = root / name / "freeze_candidates.jsonl.partial"
        accepted, counts = [], Counter()
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if counts[row["kmer"]] >= balanced[row["kmer"]]:
                    continue
                counts[row["kmer"]] += 1
                row["sample_index"] = len(accepted)
                accepted.append(row)
        _atomic_table(root / name / "manifest.csv", accepted, FIELDS)
        statistics[name]["selected_samples"] = len(accepted)
        statistics[name]["selected_kmers"] = len(counts)
        _atomic_json(root / name / "filter_statistics.json", statistics[name])
        path.unlink()
    summary = dict(schema_version="1.0", names=list(names), seed=seed, k=5, cap=cap,
        minimum=minimum, target_kmers=1024, common_kmers=sum(v > 0 for v in balanced.values()),
        samples_per_dataset=sum(balanced.values()), selected_counts=balanced,
        selection="SHA256 read priority, then minimum SHA256 window priority per read/kmer",
        orientation="ref-string 5-mers; signal remains in acquisition order",
        coordinates="0-based record-local half-open; record_index is 0-based JSONL line",
        split="evaluation (model training overlap unknown)",
        probe_split="SHA256(seed, split, physical_read_id), train/validation/test 80/10/10",
        filter_statistics=statistics)
    _atomic_json(root / "reports" / "freeze_summary.json", summary)
    if not summary["common_kmers"]:
        raise ValueError("No common eligible 5-mers; see coverage.tsv and filter_statistics.json")
    return summary


def verify(root: Path, names: list[str] | None = None, *, write_report=False):
    """Check manifest balance, identity, leakage, exact alignment and source signals."""
    root = Path(root).resolve()
    summary = json.loads((root / "reports" / "freeze_summary.json").read_text())
    names = names or summary["names"]
    expected = Counter({k: v for k, v in summary["selected_counts"].items() if v})
    global_splits, results = {}, {}
    for name in names:
        with (root / name / "manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"{name}: sample_index must be sequential from zero")
        if Counter(row["kmer"] for row in rows) != expected:
            raise ValueError(f"{name}: selected k-mer counts differ from frozen balance")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate sample_id")
        if len({(row["physical_read_id"], row["kmer"]) for row in rows}) != len(rows):
            raise ValueError(f"{name}: repeated physical read within a k-mer")
        by_source = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["corpus"] != name:
                raise ValueError("corpus name mismatch")
            physical = row["physical_read_id"]
            if row["split"] != "evaluation" or row["probe_split"] != _split(physical, summary["seed"]):
                raise ValueError("split/probe_split is not the frozen evaluation/physical-read split")
            if global_splits.setdefault(physical, row["probe_split"]) != row["probe_split"]:
                raise ValueError("physical read crosses splits")
            by_source[_resolve(row["source_jsonl"], root)][int(row["record_index"])].append(row)
        checked = 0
        for source, indices in by_source.items():
            signals, calibration = Signals(source), Calibration(source)
            with source.open() as handle:
                for record_index, line in enumerate(handle):
                    if record_index not in indices:
                        continue
                    record = json.loads(line)
                    if calibration.excluded(record):
                        raise ValueError("calibration physical read leaked into manifest")
                    signal, info = signals.get(record)
                    valid = {start: (kmer, spans) for start, kmer, spans in _exact_windows(record, len(signal))}
                    for row in indices[record_index]:
                        start = int(row["ref_start"])
                        if start not in valid or valid[start][0] != row["kmer"]:
                            raise ValueError("manifest kmer is not a trusted exact alignment")
                        kmer, spans = valid[start]
                        expected_fields = {"read_id": str(record["read_id"]), "physical_read_id": _physical(record),
                            "ref_end": start + 5, "center_ref": start + 2,
                            "signal_start": min(s[0] for s in spans), "signal_end": max(s[1] for s in spans),
                            "center_signal_start": spans[2][0], "center_signal_end": spans[2][1],
                            "signal_length": len(signal), "signal_query_orientation": record["signal_alignment"]["query_orientation"],
                            "sequence_offset_bases": record["signal_alignment"]["offset_bases"],
                            "signal_path": _portable(info[0], root) if info[0] else "", "signal_row": info[1],
                            "signal_ref_start": info[2], "signal_ref_end": info[3]}
                        expected_fields["sample_id"] = _digest(name, _physical(record), str(record["read_id"]),
                            kmer, start, start + 5, expected_fields["signal_start"], expected_fields["signal_end"])
                        for field, value in expected_fields.items():
                            if row[field] != str(value):
                                raise ValueError(f"{name}/{row['sample_index']}: {field} disagrees with source")
                        checked += 1
            if checked > len(rows):
                raise ValueError("duplicate source samples")
        if checked != len(rows):
            raise ValueError(f"{name}: unresolved source record indices")
        results[name] = {"samples": checked, "kmers": len(expected), "physical_reads": len({r["physical_read_id"] for r in rows})}
    result = {"status": "passed", "datasets": results}
    if write_report:
        _atomic_json(root / "reports" / "freeze_verification.json", result)
    return result


TITLES = {'cyclone-dna': 'Cyclone DNA', 'ont-r10-hg002': 'ONT R10 HG002',
          'dna-amplicon': 'DNA amplicon', 'cyclone-rna': 'Cyclone RNA'}


def plot_corpus(root, project, overwrite=False):
    """Regenerate derived figures; never change signal/table/manifest hashes."""
    root, project = Path(root).resolve(), Path(project).resolve()
    (root / 'reports').mkdir(parents=True, exist_ok=True)
    names = json.loads((root/'reports/freeze_summary.json').read_text())['names']
    jobs = [(name, order, root/name/'aligned'/suffix)
            for name in names for order, suffix in
            [('current', 'kmers.line'), ('kmer', 'kmers.lexicographic.line')]]
    targets = [Path(str(prefix)+ext) for _,_,prefix in jobs for ext in ('.png','.svg')]
    targets += [root/'reports/plots.manifest.json', root/'reports/PLOTS.md']
    if not overwrite and any(path.exists() for path in targets):
        raise RuntimeError('Plot outputs already exist; use --overwrite to regenerate figures')
    sys.path.insert(0,str(project/'script/NanoSignalAlign/src'))
    with tempfile.TemporaryDirectory(prefix='nanocorpus-matplotlib-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib
        import nanosignalalign
        from nanosignalalign.plotting import plot_kmer_table
        reports = []
        for name, order, prefix in jobs:
            source = root/name/'aligned/kmers.tsv'
            report = plot_kmer_table(source, prefix, order=order, dpi=450,
                                     title=f'{TITLES.get(name,name)}: calibration 5-mer current',
                                     overwrite=overwrite)
            report.update(corpus=name, input_sha256=digest(source))
            for output in report['outputs'].values():
                output['sha256'] = digest(output['path'])
            reports.append(report)
            print(f'{name}: {order}, {report["rows"]} k-mers -> {prefix.name}.png/.svg',flush=True)
        provenance = dict(python=platform.python_version(), matplotlib=matplotlib.__version__,
                          nanosignalalign=nanosignalalign.__version__, dpi=450,
                          plotting_source_sha256=digest(project/'script/NanoSignalAlign/src/nanosignalalign/plotting.py'),
                          plots=reports)
    (root/'reports/plots.manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    lines=['# k-mer 电流折线图', '',
           '图表由 NanoSignalAlign 从各域现有 `aligned/kmers.tsv` 生成，不需要模型推理。提供 450 DPI PNG 和可编辑 SVG。', '',
           '| 数据集 | 校准表类别数 | 按电流升序 | 按 k-mer 字典序 |',
           '|---|---:|---|---|']
    for name in names:
        count=next(r['rows'] for r in reports if r['corpus']==name)
        parent=f'../{name}/aligned'
        lines.append(f'| {name} | {count} | [PNG]({parent}/kmers.line.png) · [SVG]({parent}/kmers.line.svg) | [PNG]({parent}/kmers.lexicographic.line.png) · [SVG]({parent}/kmers.lexicographic.line.svg) |')
    lines += ['', '字典序图按首位碱基分成 `ANNNN`、`CNNNN`、`GNNNN`、`TNNNN` 四段，组间加入竖向虚线并在段顶标注段名。分界位置按表中实际收录的 k-mer 计算；校准表不完整时，各段宽度不一定相等。', '', '按电流升序的上升趋势来自排序；不同图的同一 rank 不代表同一个 k-mer。纵轴是已有预处理信号的归一化电流中位数，不是 pA。', '',
              '这些表来自校准子集及其最少计数过滤，类别数见上表；固定评测集合的覆盖情况见 [summary.md](summary.md)。未补造缺失的校准统计值，因此未使用要求完整 1024 类的 Gray 排序。', '',
              'RNA 校准表按信号方向标记 k-mer，与评测 manifest 的参考方向标签互为逆序（不互补）；跨域按标签比较前需统一方向。各域相位校准置信度详见 [summary.md](summary.md)。', '',
              '重画全部图（在项目根目录运行）：', '', '```bash',
              f'python {Path(__file__).resolve()} plot --root {root} --overwrite', '```', '',
              '图片和来源信息见 [plots.manifest.json](plots.manifest.json)。在语料根目录运行 `sha256sum -c SHA256SUMS --quiet` 校验文件；重画仅更新图表相关校验和。']
    (root/'reports/PLOTS.md').write_text('\n'.join(lines)+'\n')
    if (root / 'SHA256SUMS').exists():
        write_checksums(root, changed=targets)
    return reports


def finalize_corpus(root, project):
    """Write reports, the model-input template and the build-time tool snapshot."""
    for directory in ('config', 'reports', 'provenance', 'logs'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/'reports/freeze_summary.json').read_text())
    verified = json.loads((root/'reports/freeze_verification.json').read_text())
    if verified.get('status') != 'passed':
        raise RuntimeError('Manifest verification has not passed')
    config = json.loads((root/'config/run_config.json').read_text())
    names = summary['names']
    datasets = {d['name']:d for d in config['datasets']}
    counts = list(summary['selected_counts'].values())
    lines = [
        '# 固定语料制备结果', '',
        f"随机种子 `{summary['seed']}`；k=5；每域每类最多 {summary['cap']} 个独立物理 read 样本。",
        f"已完成 {len(names)} 个数据域，共同覆盖 {summary['common_kmers']}/1024 种 5-mer，每域固定 {summary['samples_per_dataset']:,} 个样本，总计 {summary['samples_per_dataset']*len(names):,} 个。",
        f"每类实际平衡数量 {min(counts)}–{max(counts)}；达到上限的类别 {sum(n==summary['cap'] for n in counts)} 种，不足上限但非零 {sum(0<n<summary['cap'] for n in counts)} 种，未纳入共同集合 {sum(n==0 for n in counts)} 种。逐域覆盖见 [coverage.tsv](coverage.tsv)。", '',
        '| 数据域 | 候选物理 reads | 排除校准 reads | 最终样本 | 方向 | shift | phase confidence | held-out Pearson r |',
        '|---|---:|---:|---:|---|---:|---|---:|'
    ]
    for name in names:
        complete=json.loads((root/name/'dataset.complete.json').read_text())
        for stage in ('prep','aligned'):
            validation=json.loads((root/name/stage/'validation.json').read_text())
            if validation['status'] != 'pass' or validation['valid_records'] != validation['records']:
                raise RuntimeError(f'{name}/{stage} validation did not pass')
        alignment_summary=json.loads((root/name/'aligned/summary.json').read_text())
        if alignment_summary['counts'].get('skipped_records',0):
            raise RuntimeError(f'{name} alignment skipped records')
        calibration=json.loads((root/name/'aligned/calibration.json').read_text())
        stats=summary['filter_statistics'][name]
        m=calibration['selected_metrics']
        corr=m.get('pearson_r')
        corr=f'{corr:.4f}' if isinstance(corr,(int,float)) else str(corr)
        lines.append(f"| {name} | {complete['records']:,} | {stats.get('calibration_records_excluded',0):,} | {stats['selected_samples']:,} | {calibration['selected_orientation']} | {calibration['selected_shift']} | {calibration['confidence']} | {corr} |")
    lines += ['', 'Prep 与 Align 的逐域契约校验、最终 manifest 的信号/索引/匹配/分组/数量平衡检查均已执行，详见各域 validation.json 及 freeze_verification.json。', '', '校准置信度是 phase 候选之间的支持程度，不等于逐碱基对齐正确率。校准用物理 reads（含其内部 held-out reads）已从固定评测集排除。尚未获得待测模型的训练清单，因此训练重叠未知。', '', '未纳入的数据域及原因见 [config/run_config.json](../config/run_config.json) 中的 unavailable_datasets；新增数据域请建立新版本。', '', '所有信号来自现有预处理数组，保持原 dtype、有效数据和时间顺序。RNA 的参考数组方向适配、来源与复用方法见 [README.md](../README.md)。', '', '本阶段没有模型推理结果或表征距离；后续模型按 manifest.csv 的 sample_index 写各层 [N,D] 数组，再交给 NanoRepDist。']
    (root/'reports/summary.md').write_text('\n'.join(lines)+'\n')
    presets=['AAAAA','CCCCC','GGGGG','TTTTT','ACGTA','TGCAG']
    targets=[k for k in presets if summary['selected_counts'].get(k,0)>0]
    model_directory = Path(os.path.relpath(project / '01.capability/model_runs/model_A', root / 'config'))
    template=['# Template only: encoder.npy is to be produced by the future model adapter.', '# Paths are relative to this file. Keep model outputs outside the frozen corpus.', 'project:', '  name: fixed_corpus_model_example', f'  output_dir: {model_directory}/distance', f"  seed: {summary['seed']}", '', 'datasets:']
    for name in names:
        template += [f'  - name: {name}', f"    molecule: {datasets[name]['molecule']}", f'    manifest: ../{name}/manifest.csv', '    representations:', f'      encoder: {model_directory}/representations/{name}/encoder.npy']
    if len(names) < 2:
        template.insert(2, '# NanoRepDist requires at least two datasets; add another domain before use.')
    template += ['', 'analysis:', '  targets: ['+', '.join(targets)+']', '  stages: [encoder]', '  normalization: l2', '  distance: cosine', '  projection: global_pca', '  bootstrap_iterations: 200', '  confidence_level: 0.95', '', 'plots:', '  enabled: true', '  stage: encoder', '  formats: [png, svg]']
    (root/'config/nanorepdist.example.yaml').write_text('\n'.join(template)+'\n')
    environment = {'python':platform.python_version(),'executable':sys.executable,'packages':{}}
    for package in ('numpy','scipy','h5py','pysam','PyYAML','pandas','pod5'):
        try:
            environment['packages'][package] = version(package)
        except PackageNotFoundError:
            environment['packages'][package] = None
    (root/'provenance/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    with tarfile.open(root/'provenance/tool_sources.tar.gz', 'w:gz') as archive:
        for relative, expected_hash in config['tool_source_sha256'].items():
            source = project / relative
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
                raise RuntimeError(f'Tool source changed during preparation: {relative}')
            archive.add(source, arcname=relative)
        for package in config['tool_versions']:
            for name in ('pyproject.toml','LICENSE','README.md'):
                source = project/'script'/package/name
                if source.exists():
                    archive.add(source, arcname=str(source.relative_to(project)))


def read_checksums(root):
    entries = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        checksum, relative = line.split('  ', 1)
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or len(checksum) != 64:
            raise ValueError(f'Invalid checksum entry: {relative}')
        if relative in entries:
            raise ValueError(f'Duplicate checksum entry: {relative}')
        entries[relative] = checksum
    if not entries:
        raise ValueError('Empty SHA256SUMS')
    return entries


def write_checksums(root, changed=None):
    """Snapshot a new corpus, or refresh only the supplied derived outputs."""
    root = Path(root).resolve()
    if changed is None:
        entries = {}
        paths = [p for p in root.rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix != '.log'
                 and p.name != 'SHA256SUMS' and not p.name.endswith('.partial')]
    else:
        entries = read_checksums(root)
        paths = changed
    for path in paths:
        path = Path(path).resolve()
        entries[str(path.relative_to(root))] = digest(path)
    temporary = root / 'SHA256SUMS.partial'
    temporary.write_text(''.join(f'{entries[rel]}  {rel}\n' for rel in sorted(entries)))
    os.replace(temporary, root / 'SHA256SUMS')
    return len(entries)


def verify_checksums(root):
    entries = read_checksums(root)
    failures = [relative for relative, expected in entries.items()
                if not (root / relative).is_file() or digest(root / relative) != expected]
    if failures:
        raise RuntimeError('Checksum mismatch or missing file: ' + ', '.join(failures))
    return len(entries)


def build_corpus(args):
    """Prepare a new version, then freeze, verify, report and plot it."""
    root, project = args.root.resolve(), args.project.resolve()
    resume_frozen = (root / 'reports/freeze_summary.json').exists()
    if (root / 'freeze_summary.json').exists() or (resume_frozen and (
            not args.freeze_only or (root / 'SHA256SUMS').exists())):
        raise RuntimeError('This corpus is frozen; use verify or choose a new --root')
    if min(args.pool_size, args.cap, args.workers) <= 0:
        raise ValueError('pool-size, cap and workers must be positive')
    inventory = json.loads(args.inventory.read_text())
    datasets = inventory['datasets']
    if args.dataset:
        unknown = set(args.dataset) - {d['name'] for d in datasets}
        if unknown:
            raise ValueError(f'Unknown datasets: {sorted(unknown)}')
        datasets = [d for d in datasets if d['name'] in args.dataset]
    names = [d['name'] for d in datasets]
    reserved = {'config', 'reports', 'provenance', 'logs'}
    if not names or len(set(names)) != len(names) or any(
            n in reserved or not n or n in {'.', '..'} or Path(n).name != n for n in names):
        raise ValueError('Dataset names must be unique directory names, excluding reserved folders')
    config = {
        'schema_version': 2, 'seed': args.seed, 'k': 5, 'cap_per_kmer': args.cap,
        'pool_size_per_dataset': args.pool_size, 'datasets': datasets,
        'unavailable_datasets': inventory.get('unavailable_datasets', []),
        'calibration': {'kmer_pattern': 'XXNXX', 'shift_min': -6, 'shift_max': 6,
                        'orientation': 'auto', 'events_per_read': 16, 'max_events': 50000,
                        'calibration_fraction': 0.2, 'holdout_fraction': 0.25,
                        'min_kmer_count': 3, 'min_test_predictions': 200},
        'minimum_reference_match_fraction': 0.90,
        'tool_versions': {}, 'tool_source_sha256': {},
        'corpus_code_sha256': {'corpus.py': digest(Path(__file__))},
    }
    for package, module in (('NanoSignalPrep', 'nanosignalprep'),
                            ('NanoSignalAlign', 'nanosignalalign'), ('NanoRepDist', 'nanorepdist')):
        directory = project / 'script' / package
        config['tool_versions'][package] = tomllib.loads((directory / 'pyproject.toml').read_text())['project']['version']
        for source in sorted((directory / 'src' / module).rglob('*.py')):
            config['tool_source_sha256'][str(source.relative_to(project))] = digest(source)
    config_path = root / 'config/run_config.json'
    if resume_frozen and not config_path.exists():
        raise RuntimeError('Incomplete frozen build has no run_config; inspect before resuming')
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing run_config differs; choose a new version directory')
    if args.freeze_only:
        pending = [name for name in names if not (root / name / 'dataset.complete.json').exists()]
        if pending:
            raise RuntimeError(f'Preparation is incomplete: {pending}')
    for directory in ('config', 'reports', 'provenance', 'logs'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    dump(config_path, config)
    inventory_target = root / 'config/sources.inventory.json'
    if args.inventory.resolve() != inventory_target:
        if inventory_target.exists() and inventory_target.read_bytes() != args.inventory.read_bytes():
            raise RuntimeError('Existing source inventory differs; choose a new version directory')
        shutil.copy2(args.inventory, inventory_target)
    # Keep the single live runner at the corpus collection root.
    with tarfile.open(root / 'provenance/corpus_runner.tar.gz', 'w:gz') as archive:
        archive.add(Path(__file__), arcname='corpus.py')
    relative_runner = os.path.relpath(Path(__file__).resolve(), root)
    if not (root / 'README.md').exists():
        (root / 'README.md').write_text(
            '# 固定电信号语料\n\n'
            '结果：[reports/summary.md](reports/summary.md)；'
            '图表：[reports/PLOTS.md](reports/PLOTS.md)。\n\n'
            f'在本目录运行 `python {relative_runner} verify --root .` 校验；'
            f'`python {relative_runner} plot --root . --overwrite` 重画。'
            f'制备新版本使用 `python {relative_runner} build --root ../stone_v2`。\n\n'
            'config/ 保存源清单、运行参数和 NanoRepDist 模板；reports/ 保存报告；'
            'provenance/ 保存环境与工具源码；logs/ 可保存运行日志。'
            '每个数据域包含 prep/、aligned/ 和 manifest.csv。\n\n'
            '模型按 sample_index 生成表征数组；信号保持采集时间顺序。'
            '新增数据域或改变参数请使用新的 --root。'
            'JSONL 含绝对信号路径，不能直接移动整个语料目录。\n')
    configure(project)
    if args.freeze_only:
        # Reuse goes through the same completion/configuration/hash checks.
        for dataset in datasets:
            prepare(project, root, dataset, args.seed, args.pool_size)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(prepare, str(project), str(root), dataset,
                                    args.seed, args.pool_size) for dataset in datasets]
            for job in concurrent.futures.as_completed(jobs):
                job.result()
    if args.prepare_only:
        print('Preparation complete; resume with build --freeze-only and the same options.', flush=True)
        return
    if resume_frozen:
        frozen = json.loads((root / 'reports/freeze_summary.json').read_text())
        if (frozen['names'], frozen['seed'], frozen['cap']) != (names, args.seed, args.cap):
            raise RuntimeError('Frozen selection differs from requested configuration')
    else:
        freeze(root, names, seed=args.seed, cap=args.cap)
    result = verify(root, names, write_report=True)
    finalize_corpus(root, project)
    plot_corpus(root, project, overwrite=resume_frozen)
    result['checksum_files'] = write_checksums(root)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


# Apple variants preserve the parent's sample identities and signal coordinates.
_APPLE_HANDLES = {}
_APPLE_ARRAYS = {}


def _apple_worker_init(project):
    configure(project)


def _apple_raw_signal(path, read_id):
    """Read calibrated pA, requiring channel calibration rather than guessing units."""
    path = Path(path)
    key = str(path)
    if key not in _APPLE_HANDLES:
        if len(_APPLE_HANDLES) >= 12:
            oldest = next(iter(_APPLE_HANDLES))
            _APPLE_HANDLES.pop(oldest).close()
        if path.suffix.lower() == '.pod5':
            import pod5
            _APPLE_HANDLES[key] = pod5.Reader(path)
        elif path.suffix.lower() == '.fast5':
            import h5py
            _APPLE_HANDLES[key] = h5py.File(path, 'r')
        else:
            raise ValueError(f'Unsupported raw signal format: {path}')
    handle = _APPLE_HANDLES[key]
    if path.suffix.lower() == '.pod5':
        read = next(handle.reads(selection=[read_id], missing_ok=False))
        signal = np.asarray(read.signal_pa, dtype=np.float32)
    else:
        from nanosignalprep.fast5 import _find_signal, _calibrate
        values, attrs = _find_signal(handle, read_id)
        if any(k not in attrs for k in ('digitisation', 'offset', 'range')):
            raise ValueError(f'{path}/{read_id}: missing raw-to-pA calibration')
        if not np.isfinite([float(attrs[k]) for k in ('digitisation', 'offset', 'range')]).all() or float(attrs['digitisation']) <= 0 or float(attrs['range']) <= 0:
            raise ValueError(f'{path}/{read_id}: invalid raw-to-pA calibration')
        signal = np.asarray(_calibrate(np, np.asarray(values, dtype=np.float32), attrs), dtype=np.float32)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError(f'{path}/{read_id}: invalid calibrated pA signal')
    return signal


def _apple_median_filter(signal, size, mode='reflect'):
    """Exact rank median with SciPy reflect edges, using an O(N log W) window."""
    if mode != 'reflect' or not 1 <= size <= len(signal):
        raise ValueError('Apple baseline filter requires a positive reflect window')
    import pandas as pd
    # SciPy's even-size median is the upper middle value, not their average.
    padded = np.pad(signal, (size // 2, (size - 1) // 2), mode='symmetric')
    filtered = pd.Series(padded).rolling(size).quantile(0.5, interpolation='higher')
    return filtered.to_numpy()[size - 1:].astype(signal.dtype, copy=False)


def apple_normalize_pa(signal, implementation='prep'):
    """Apple on a complete pA read; never apply this to pre-normalized chunks."""
    from scipy.signal import medfilt
    from nanosignalprep import normalization as normal
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError('Apple requires a nonempty finite one-dimensional pA signal')
    if implementation != 'prep':
        raise ValueError('This entry implements NanoSignalPrep Apple (prep)')
    cleaned = normal._repair(np, signal)
    cleaned = normal._remove_spikes(np, _apple_median_filter, cleaned)
    result = medfilt(normal._robust_residual(np, cleaned), kernel_size=5).astype(np.float32)
    result = normal._soft_bounds(np, result)
    return np.asarray(result, dtype=np.float32)


def _apple_affine_check(raw_chunk, parent_chunk):
    """Confirm raw chunk identity against an affine-normalized, float16 parent."""
    x = np.asarray(raw_chunk, dtype=np.float64)
    y = np.asarray(parent_chunk, dtype=np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Raw/parent chunks differ in length or have invalid values')
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(xc @ xc)
    if denominator <= 0 or float(yc @ yc) <= 0:
        raise ValueError('Cannot confirm chunk identity from constant signals')
    slope = float(xc @ yc) / denominator
    intercept = float(y.mean() - slope * x.mean())
    corr = float(np.corrcoef(x, y)[0, 1])
    error = y - (slope * x + intercept)
    maximum = float(np.max(np.abs(error)))
    tolerance = 0.002 * max(1.0, float(np.max(np.abs(y))))
    if slope <= 0 or corr < 0.99999 or maximum > tolerance:
        raise ValueError(f'Raw signal does not reproduce parent chunk: r={corr:.8f}, max_error={maximum:.6g}')
    return {'correlation': corr, 'affine_slope': slope, 'affine_intercept': intercept,
            'max_abs_error': maximum, 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'meaning': 'identity check only; fitted values are not claimed upstream normalization parameters'}


def _apple_one_chunk(job):
    row, read_id, physical, path, parent_path, parent_row, parent_start, length, settings, implementation = job
    raw = _apple_raw_signal(path, physical)
    trim = int(settings.get('trim_samples', 0))
    size, overlap = int(settings['chunk_size']), int(settings['overlap'])
    if not 0 <= overlap < size or not 0 <= trim < len(raw):
        raise ValueError(f'{read_id}: invalid chunk/trim settings')
    signal = raw[trim:]
    parts = read_id.rsplit(':', 2)
    if len(parts) != 3 or parts[0] != physical:
        raise ValueError(f'{read_id}: cannot recover indexed chunk identity')
    index, total = map(int, parts[1:])
    expected_total = 1 + (len(signal) - size) // (size - overlap)
    start = (index - 1) * (size - overlap)
    if total != expected_total or not 1 <= index <= total or length != size:
        raise ValueError(f'{read_id}: raw length/chunk count does not match source export ({expected_total} expected)')
    if parent_path not in _APPLE_ARRAYS:
        _APPLE_ARRAYS[parent_path] = np.load(parent_path, mmap_mode='r', allow_pickle=False)
    parent = _APPLE_ARRAYS[parent_path][parent_row, parent_start:parent_start + length]
    check = _apple_affine_check(signal[start:start + length], parent)
    apple = apple_normalize_pa(signal, implementation)
    chunk = np.array(apple[start:start + length], dtype=np.float32, copy=True)
    if len(chunk) != length or not np.isfinite(chunk).all():
        raise ValueError(f'{read_id}: Apple changed signal length or introduced invalid values')
    provenance = {
        'record_index': row, 'read_id': read_id, 'physical_read_id': physical,
        'path': str(path), 'units': 'pA', 'raw_signal_length': len(raw),
        'normalization_start': trim, 'normalization_end': len(raw),
        'chunk_start': trim + start, 'chunk_end': trim + start + length,
        'chunk_size': size, 'overlap': overlap,
        'raw_pa_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
        'apple_chunk_sha256': hashlib.sha256(chunk.tobytes()).hexdigest(),
        'parent_match': check,
    }
    return row, chunk, provenance


def prepare_apple_dataset(project, source_root, root, name, settings, implementation, workers):
    """Create an Apple signal pool in the same row order, with resumable checkpoints."""
    configure(project)
    source, target = source_root / name, root / name
    complete = target / 'dataset.complete.json'
    if complete.exists():
        metadata = json.loads(complete.read_text())
        for rel, expected in metadata['sha256'].items():
            if digest(target / rel) != expected:
                raise RuntimeError(f'{name}: completed Apple artifact changed: {rel}')
        return metadata
    prep = target / 'prep'
    prep.mkdir(parents=True, exist_ok=True)
    source_jsonl = source / 'prep/chunks.jsonl'
    original = np.load(source / 'prep/signals.npy', mmap_mode='r', allow_pickle=False)
    records = []
    with source_jsonl.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    provenance_path = prep / 'raw_sources.jsonl'
    partial_path = prep / 'raw_sources.jsonl.partial'
    signal_path = prep / 'signals.npy'
    partial_signal = prep / 'signals.npy.partial'
    finalized = signal_path.exists() and provenance_path.exists()
    if finalized:
        store_path, journal_path = signal_path, provenance_path
    else:
        # Recover an interruption between either of the final atomic renames.
        store_path = signal_path if signal_path.exists() else partial_signal
        journal_path = provenance_path if provenance_path.exists() else partial_path
    array = np.lib.format.open_memmap(store_path, mode='r+' if store_path.exists() else 'w+',
                                    dtype=np.float32, shape=original.shape)
    if array.shape != original.shape or array.dtype != np.dtype('float32'):
        raise RuntimeError(f'{name}: incompatible partial Apple pool')
    done = {}
    if journal_path.exists():
        # An interrupted final append is uncommitted; recompute that row.
        with journal_path.open('r+b') as journal:
            while True:
                boundary = journal.tell()
                line = journal.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    journal.truncate(boundary)
                    break
        with journal_path.open() as handle:
            for line in handle:
                value = json.loads(line)
                row = int(value['record_index'])
                if row in done or not 0 <= row < len(records) or value['read_id'] != records[row]['read_id']:
                    raise RuntimeError(f'{name}: invalid or duplicate resume record')
                length = int(records[row]['signal_ref']['length'])
                if hashlib.sha256(np.asarray(array[row, :length]).tobytes()).hexdigest() != value['apple_chunk_sha256']:
                    raise RuntimeError(f'{name}/{row}: partial signal checksum differs')
                done[row] = value
    jobs = []
    for row, record in enumerate(records):
        if row in done:
            continue
        ref = record['signal_ref']
        if int(ref.get('start', 0)) != 0 or int(ref['row']) != row:
            raise ValueError(f'{name}/{row}: parent signal pool must have row-aligned zero-based slices')
        path = settings['raw_files'][record['meta']['filename']]
        jobs.append((row, record['read_id'], _physical(record), path,
                     str(_resolve(ref['path'], source_jsonl.parent)), int(ref['row']),
                     int(ref.get('start', 0)), int(ref['length']),
                     {key: settings[key] for key in ('chunk_size', 'overlap', 'trim_samples')}, implementation))
    started = time.monotonic()
    print(f'{name}: Apple {len(done)}/{len(records)} resumed; workers={workers}', flush=True)
    if jobs:
        with journal_path.open('a') as journal, concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_apple_worker_init, initargs=(str(project),)) as executor:
            iterator = iter(jobs)
            pending = {}
            for job in itertools.islice(iterator, workers * 2):
                pending[executor.submit(_apple_one_chunk, job)] = job[1]
            batch = []
            while pending:
                ready, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    read_id = pending.pop(future)
                    try:
                        row, chunk, info = future.result()
                    except Exception as exc:
                        for outstanding in pending:
                            outstanding.cancel()
                        raise RuntimeError(f'{name}/{read_id}: Apple extraction failed: {exc}') from exc
                    array[row, :len(chunk)] = chunk
                    array[row, len(chunk):] = 0
                    done[row] = info
                    batch.append(info)
                    job = next(iterator, None)
                    if job is not None:
                        pending[executor.submit(_apple_one_chunk, job)] = job[1]
                if len(batch) >= 100 or not pending:
                    array.flush()
                    for info in batch:
                        journal.write(json.dumps(info, separators=(',', ':')) + '\n')
                    journal.flush()
                    os.fsync(journal.fileno())
                    batch.clear()
                    print(f'{name}: Apple {len(done)}/{len(records)} ({time.monotonic()-started:.1f}s)', flush=True)
    array.flush()
    del array
    if len(done) != len(records):
        raise RuntimeError(f'{name}: not all parent chunks were generated')
    if store_path != signal_path:
        os.replace(store_path, signal_path)
    if journal_path != provenance_path:
        os.replace(journal_path, provenance_path)
    output_jsonl = prep / 'chunks.jsonl'
    with output_jsonl.with_suffix('.jsonl.partial').open('w') as handle:
        for row, record in enumerate(records):
            previous_ref = dict(record['signal_ref'])
            record['signal_ref'] = {**previous_ref, 'path': str(signal_path.resolve()), 'row': row}
            record['normal_mode'] = 'apple'
            record['meta'] = {**record['meta'], 'parent_signal_ref': previous_ref,
                              'normalization_implementation': 'NanoSignalPrep.apple',
                              'normalization_support': 'complete source read before chunking',
                              'raw_signal_source': done[row]}
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
    os.replace(output_jsonl.with_suffix('.jsonl.partial'), output_jsonl)
    for file in ('selection.tsv', 'rejected.jsonl'):
        shutil.copy2(source / 'prep' / file, prep / file)
    from nanosignalprep.validation import validate_jsonl
    validated = validate_jsonl(output_jsonl).as_dict()
    dump(prep / 'validation.json', validated)
    if validated['status'] != 'pass':
        raise RuntimeError(f'{name}: Apple Prep validation failed')
    matches = [value['parent_match'] for value in done.values()]
    summary = {
        'records': len(records), 'normalization': 'apple', 'implementation': implementation,
        'signal_dtype': 'float32', 'normalization_support': 'complete source read before chunking',
        'parent': str(source), 'parent_prep_sha256': digest(source_jsonl),
        'pool_size_requested': len(original), 'signal_array_allocated_rows': len(original),
        'physical_reads': len({_physical(r) for r in records}),
        'raw_chunk_identity': {'checked': len(matches), 'min_correlation': min(r['correlation'] for r in matches),
                               'max_abs_error': max(r['max_abs_error'] for r in matches)},
    }
    dump(prep / 'export_summary.json', summary)
    del records, done, original
    return finalize_apple_dataset(project, source_root, root, name)

def finalize_apple_dataset(project, source_root, variant_root, name):
    """Fit Apple current values at the inherited phase; keep every evaluation sample."""
    configure(project)
    import itertools
    import math
    import shutil
    from nanosignalalign.alignment import align_jsonl
    from nanosignalalign.calibration import CalibrationOptions, _collect_observations, _score_candidate
    from nanosignalalign.tables import write_kmer_summary
    from nanosignalalign.validation import validate_jsonl as validate_aligned

    started = time.time()
    source_root, variant_root = Path(source_root).resolve(), Path(variant_root).resolve()
    source, target = source_root / name, variant_root / name
    original_complete_path = source / 'dataset.complete.json'
    original_complete = json.loads(original_complete_path.read_text())
    original_complete_sha256 = digest(original_complete_path)
    complete = target / 'dataset.complete.json'
    if complete.exists():
        result = json.loads(complete.read_text())
        if result.get('normal_mode') != 'apple' or result.get('source_complete_sha256') != original_complete_sha256:
            raise RuntimeError(f'{name}: existing Apple output belongs to a different source')
        for rel, expected in result['sha256'].items():
            if not (target / rel).is_file() or digest(target / rel) != expected:
                raise RuntimeError(f'{name}: Apple artifact checksum mismatch: {rel}')
        for filename in ('manifest.csv', 'filter_statistics.json'):
            if digest(target / filename) != digest(source / filename):
                raise RuntimeError(f'{name}: source sample selection changed: {filename}')
        return result

    prep = target / 'prep'
    for filename in ('signals.npy', 'chunks.jsonl', 'selection.tsv', 'rejected.jsonl',
                     'validation.json', 'export_summary.json', 'raw_sources.jsonl'):
        if not (prep / filename).is_file():
            raise RuntimeError(f'{name}: Apple preparation is incomplete: {filename}')
    validation = json.loads((prep / 'validation.json').read_text())
    expected_records = original_complete['records']
    if validation.get('status') != 'pass' or validation.get('valid_records') != expected_records or validation.get('records') != expected_records:
        raise RuntimeError(f'{name}: Apple Prep validation failed or record count changed')
    aligned = target / 'aligned'
    aligned.mkdir(exist_ok=True)
    original_calibration_path = source / 'aligned/calibration.json'
    original = json.loads(original_calibration_path.read_text())
    # Keep the complete candidate set while selecting events, so eligibility and
    # stable ranking remain identical; score only the inherited selected phase.
    settings = CalibrationOptions(**original['settings'], kmer_pattern=original['kmer_pattern'], fail_fast=True)
    observations, stats = _collect_observations(prep / 'chunks.jsonl', settings)
    if stats != original['stats']:
        raise RuntimeError(f'{name}: Apple calibration membership/event counts changed')
    key = (original['selected_orientation'], original['selected_shift'])
    metrics, model = _score_candidate(observations, key, settings)
    canonical = {k: {'median_current': v[0], 'train_count': v[1]} for k, v in sorted(model.items())}
    original_counts = {k: v['train_count'] for k, v in original['canonical_current_model'].items()}
    if {k: v['train_count'] for k, v in canonical.items()} != original_counts:
        raise RuntimeError(f'{name}: Apple calibration k-mers or training counts changed')
    if metrics['test_predictions'] != original['selected_metrics']['test_predictions']:
        raise RuntimeError(f'{name}: Apple held-out prediction membership changed')
    if metrics['mae'] is None or not math.isfinite(metrics['mae']) or metrics['test_predictions'] < settings.min_test_predictions:
        raise RuntimeError(f'{name}: Apple fixed-phase calibration has insufficient finite predictions')
    input_path = (prep / 'chunks.jsonl').resolve()
    inherited = {'path': str(original_calibration_path), 'sha256': digest(original_calibration_path),
                 'confidence': original['confidence']}
    calibration = {**original, 'operation': 'paired_normalization_fixed_phase',
                   'input': {'path': str(input_path), 'bytes': input_path.stat().st_size,
                             'mtime_ns': input_path.stat().st_mtime_ns},
                   'confidence': 'unknown', 'margin_ratio': None,
                   'selected_metrics': metrics, 'candidate_metrics': [metrics],
                   'canonical_current_model': canonical, 'stats': stats,
                   'phase_inherited_from': inherited, 'normal_mode': 'apple',
                   'elapsed_seconds': time.time() - started}
    calibration_path = aligned / 'calibration.json'
    dump(calibration_path, calibration)
    write_kmer_summary(calibration, aligned / 'kmers.tsv', overwrite=True)
    result = align_jsonl(input_path, calibration_path, aligned / 'chunks.jsonl',
                         overwrite=True, fail_fast=True, summary_json=aligned / 'summary.json')
    if result['counts'].get('skipped_records', 0) or result['counts'].get('output_records') != expected_records:
        raise RuntimeError(f'{name}: Apple alignment skipped or changed records')
    validation = validate_aligned(aligned / 'chunks.jsonl').as_dict()
    dump(aligned / 'validation.json', validation)
    if validation['status'] != 'pass' or validation['valid_records'] != expected_records:
        raise RuntimeError(f'{name}: Apple Align validation failed')
    fields = ('read_id', 'chunk_id', 'chunk_index', 'physical_read_id', 'ref', 'seq',
              'ref_name', 'strand', 'target_ref_start', 'target_ref_end', 'align',
              'moves', 'mv_stride', 'base_sample_span_seq', 'base_sample_span_ref',
              'base_sample_span_ref_baseline', 'base_sample_span_ref_calibrated')
    phase_fields = ('stage', 'kmer_pattern', 'query_orientation', 'offset_bases',
                    'calibration_role', 'baseline_preserved', 'boundary_refined')
    compared = 0
    with (source / 'aligned/chunks.jsonl').open() as old_handle, (aligned / 'chunks.jsonl').open() as new_handle:
        for row, pair in enumerate(itertools.zip_longest(old_handle, new_handle)):
            if None in pair:
                raise RuntimeError(f'{name}: paired alignment record count mismatch at row {row}')
            old, new = map(json.loads, pair)
            for field in fields:
                if old.get(field) != new.get(field):
                    raise RuntimeError(f'{name}: paired alignment mismatch at row {row}: {field}')
            for field in phase_fields:
                if old['signal_alignment'].get(field) != new['signal_alignment'].get(field):
                    raise RuntimeError(f'{name}: paired phase mismatch at row {row}: {field}')
            if {k:v for k,v in old['signal_ref'].items() if k != 'path'} != {k:v for k,v in new['signal_ref'].items() if k != 'path'}:
                raise RuntimeError(f'{name}: paired signal indices changed at row {row}')
            compared += 1
    for filename in ('manifest.csv', 'filter_statistics.json'):
        shutil.copyfile(source / filename, target / filename)
        if digest(source / filename) != digest(target / filename):
            raise RuntimeError(f'{name}: paired sample file copy failed: {filename}')
    dump(aligned / 'paired.validation.json', {'status': 'passed', 'records': compared,
         'record_fields_compared': list(fields), 'phase_fields_compared': list(phase_fields),
         'source_calibration_sha256': inherited['sha256'], 'manifest_byte_identical': True,
         'calibration_membership_and_counts_identical': True, 'resampled': False})
    dump(aligned / 'qc.json', {'warnings': ['Phase inherited from source; Apple phase search was not performed.',
         'Inherited phase confidence: ' + original['confidence']], 'confidence': 'unknown',
         'selected_metrics': metrics, 'phase_inherited_from': inherited,
         'scope': 'Apple held-out current fit at fixed source phase; not independent alignment accuracy'})
    output = {k: original_complete[k] for k in ('dataset', 'seed', 'pool_size_requested', 'records')}
    output.update({'normal_mode': 'apple', 'rejected': 0, 'source_complete_sha256': original_complete_sha256,
                   'calibration': {k: calibration[k] for k in ('selected_orientation', 'selected_shift', 'confidence', 'selected_metrics')},
                   'seconds': round(time.time() - started, 2),
                   'sha256': {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob('*')) if p.is_file() and p != complete}})
    dump(complete, output)
    return output


def verify_frozen_apple(root, source_root, requested_names=None):
    """Read-only reuse of an immutable Apple corpus after runner/path changes."""
    count = verify_checksums(root)
    checksums = read_checksums(root)
    required = {'config/apple.json', 'reports/freeze_summary.json',
                'reports/freeze_verification.json', 'reports/paired.validation.json'}
    if not required.issubset(checksums):
        raise RuntimeError('Frozen Apple checksum list omits its configuration or verification reports')
    config = json.loads((root / 'config/apple.json').read_text())
    names = config.get('names', [])
    if (config.get('normal_mode') != 'apple' or not names or len(set(names)) != len(names)
            or any(not name or Path(name).name != name or name in {'.', '..'} for name in names)):
        raise RuntimeError('Invalid frozen Apple configuration')
    if Path(config['source_root']).resolve() != source_root:
        raise RuntimeError('Frozen Apple parent differs from --source-root')
    if requested_names is not None and requested_names != names:
        raise RuntimeError('Frozen Apple datasets differ from --dataset; choose a new output directory')
    parent_checksums = read_checksums(source_root)
    parent_artifacts = config.get('parent_artifacts_sha256', {})
    required_parent = {'reports/freeze_summary.json', 'config/run_config.json'}
    required_parent.update(f'{name}/manifest.csv' for name in names)
    if not required_parent.issubset(parent_artifacts):
        raise RuntimeError('Frozen Apple configuration omits parent fingerprints')
    for relative, expected in parent_artifacts.items():
        if parent_checksums.get(relative) != expected or digest(source_root / relative) != expected:
            raise RuntimeError(f'Parent corpus changed: {relative}')
    summary = json.loads((root / 'reports/freeze_summary.json').read_text())
    parent_summary = json.loads((source_root / 'reports/freeze_summary.json').read_text())
    paired = json.loads((root / 'reports/paired.validation.json').read_text())
    verified = json.loads((root / 'reports/freeze_verification.json').read_text())
    if (summary['names'] != names or set(names) - set(parent_summary['names'])
            or any(summary[key] != parent_summary[key] for key in
                   ('seed', 'cap', 'selected_counts', 'samples_per_dataset'))):
        raise RuntimeError('Frozen Apple selection differs from its parent')
    if (paired.get('status') != 'passed' or verified.get('status') != 'passed'
            or Path(paired['source_root']).resolve() != source_root
            or set(paired['datasets']) != set(names) or set(verified['datasets']) != set(names)):
        raise RuntimeError('Frozen Apple verification reports are incomplete or inconsistent')
    for name in names:
        manifest = f'{name}/manifest.csv'
        report_path = f'{name}/aligned/paired.validation.json'
        if not {manifest, report_path, f'{name}/dataset.complete.json'}.issubset(checksums):
            raise RuntimeError(f'{name}: frozen checksum list omits paired artifacts')
        if digest(root / manifest) != digest(source_root / manifest):
            raise RuntimeError(f'{name}: paired manifest differs from its parent')
        report = json.loads((root / report_path).read_text())
        complete = json.loads((root / name / 'dataset.complete.json').read_text())
        parent_complete = json.loads((source_root / name / 'dataset.complete.json').read_text())
        if (report.get('status') != 'passed' or not report.get('manifest_byte_identical')
                or not report.get('calibration_membership_and_counts_identical')
                or report.get('resampled') is not False
                or report['source_calibration_sha256'] != digest(source_root / name / 'aligned/calibration.json')
                or complete['source_complete_sha256'] != digest(source_root / name / 'dataset.complete.json')
                or complete['records'] != parent_complete['records'] or report['records'] != complete['records']
                or paired['datasets'][name]['chunks'] != complete['records']
                or paired['datasets'][name]['manifest_sha256'] != digest(root / manifest)
                or paired['datasets'][name]['samples'] != summary['samples_per_dataset']
                or verified['datasets'][name]['samples'] != summary['samples_per_dataset']):
            raise RuntimeError(f'{name}: frozen paired validation differs from its parent')
    return count


def build_apple_corpus(args):
    """Generate a paired Apple variant from calibrated raw reads, without resampling."""
    source_root, root, project = args.source_root.resolve(), args.out_dir.resolve(), args.project.resolve()
    if root == source_root or source_root.is_relative_to(root):
        raise ValueError('Apple output must be a separate child or sibling directory')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if (root / 'SHA256SUMS').exists():
        count = verify_frozen_apple(root, source_root, args.dataset)
        print(f'Existing Apple corpus: {count} files verified; parent and manifests paired', flush=True)
        return
    configure(project)
    inventory = json.loads(args.raw_config.read_text())
    summary = json.loads((source_root / 'reports/freeze_summary.json').read_text())
    names = args.dataset or summary['names']
    if not names or len(set(names)) != len(names) or set(names) - set(summary['names']):
        raise ValueError('dataset must select unique names from the frozen parent')
    raw_settings = inventory['datasets']
    missing = []
    files = {}
    for name in names:
        if name not in raw_settings:
            raise ValueError(f'{name}: missing raw source configuration')
        settings = raw_settings[name]
        if any(key not in settings for key in ('raw_files', 'chunk_size', 'overlap', 'trim_samples')):
            raise ValueError(f'{name}: incomplete raw chunk configuration')
        with (source_root / name / 'prep/chunks.jsonl').open() as handle:
            for line in handle:
                record = json.loads(line)
                filename = record['meta']['filename']
                path = settings['raw_files'].get(filename)
                if path is None:
                    missing.append(f'{name}/{filename}')
                else:
                    files[str(Path(path).resolve())] = None
    for path in files:
        file = Path(path)
        if not file.is_file():
            missing.append(path)
        else:
            stat = file.stat()
            files[path] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if missing:
        raise FileNotFoundError('Missing original raw sources: ' + ', '.join(sorted(set(missing))))
    parent_checksums = read_checksums(source_root)
    parent_artifacts = {path: sha for path, sha in parent_checksums.items()
                        if Path(path).parts[0] in names or path in {'reports/freeze_summary.json', 'config/run_config.json'}}
    for relative, expected in parent_artifacts.items():
        if digest(source_root / relative) != expected:
            raise RuntimeError(f'Parent corpus changed: {relative}')
    from nanosignalprep import normalization
    normal_source = Path(normalization.__file__)
    config = {
        'schema_version': 1, 'normal_mode': 'apple', 'implementation': 'NanoSignalPrep.apple',
        'normalization_support': 'complete source read before chunking',
        'normalization_source_sha256': digest(normal_source),
        'median_filter': 'exact pandas rolling upper-median; SciPy reflect-equivalent',
        'source_root': str(source_root), 'names': names,
        'phase_policy': 'inherit frozen parent phase, recompute currents using the same calibration events',
        'sample_policy': 'copy identical manifests; never resample or drop reads',
        'raw_sources': {name: raw_settings[name] for name in names},
        'raw_files': files, 'parent_artifacts_sha256': parent_artifacts,
        'corpus_code_sha256': digest(Path(__file__)),
    }
    config_path = root / 'config/apple.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Existing Apple configuration differs; choose a new output directory')
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError('Apple output is not empty and lacks its configuration')
    for directory in ('config', 'reports', 'provenance', 'logs'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    dump(config_path, config)
    # Preserve the precise normalizer and runner used by this variant.
    with tarfile.open(root / 'provenance/apple_sources.tar.gz', 'w:gz') as archive:
        archive.add(Path(__file__), arcname='corpus.py')
        archive.add(normal_source, arcname='nanosignalprep/normalization.py')
    for name in names:
        prepare_apple_dataset(project, source_root, root, name, raw_settings[name], 'prep', args.workers)
    variant_summary = {**summary, 'names': list(names),
                       'filter_statistics': {name: summary['filter_statistics'][name] for name in names},
                       'normal_mode': 'apple', 'paired_source': str(source_root),
                       'phase_policy': config['phase_policy']}
    dump(root / 'reports/freeze_summary.json', variant_summary)
    with (source_root / 'reports/coverage.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = list(reader.fieldnames)
        coverage = [row for row in reader if row['corpus'] in names]
    _atomic_table(root / 'reports/coverage.tsv', coverage, fields, '\t')
    verified = verify(root, write_report=True)
    parent_run = json.loads((source_root / 'config/run_config.json').read_text())
    run_config = {**parent_run, 'schema_version': 2, 'normal_mode': 'apple',
                  'datasets': [d for d in parent_run['datasets'] if d['name'] in names],
                  'paired_source': str(source_root), 'phase_policy': config['phase_policy'],
                  'corpus_code_sha256': {'corpus.py': digest(Path(__file__))},
                  'tool_source_sha256': {}}
    for package, module in (('NanoSignalPrep', 'nanosignalprep'), ('NanoSignalAlign', 'nanosignalalign'),
                            ('NanoRepDist', 'nanorepdist')):
        for file in sorted((project / 'script' / package / 'src' / module).rglob('*.py')):
            run_config['tool_source_sha256'][str(file.relative_to(project))] = digest(file)
    dump(root / 'config/run_config.json', run_config)
    dump(root / 'config/sources.inventory.json', {'datasets': run_config['datasets'], 'raw_sources': config['raw_sources']})
    finalize_corpus(root, project)
    template_path = root / 'config/nanorepdist.example.yaml'
    template_path.write_text(template_path.read_text().replace('model_A/', 'model_A/apple/'))
    summary_path = root / 'reports/summary.md'
    text = summary_path.read_text()
    text = text.replace('所有信号来自现有预处理数组，保持原 dtype、有效数据和时间顺序。',
                        '所有信号从原始 pA read 重新执行 NanoSignalPrep Apple，再按原 chunk 边界裁剪，以 float32 保存。')
    text += ('\n本版本与父语料保持相同 sample_id、sample_index、k-mer、read 分组及全部窗口坐标。'
             '方向和偏移沿用父语料；Apple 电流模型使用相同校准 reads 和事件重新统计。'
             '此处 phase confidence=unknown 表示未重新搜索偏移，父语料的原始置信度仍保存在各域 calibration.json 的 phase_inherited_from 中。\n')
    summary_path.write_text(text)
    paired = {}
    for name in names:
        if (root / name / 'manifest.csv').read_bytes() != (source_root / name / 'manifest.csv').read_bytes():
            raise RuntimeError(f'{name}: paired manifest changed')
        complete = json.loads((root / name / 'dataset.complete.json').read_text())
        paired[name] = {'manifest_sha256': digest(root / name / 'manifest.csv'),
                        'samples': verified['datasets'][name]['samples'],
                        'chunks': complete['records'],
                        'raw_identity': json.loads((root / name / 'prep/export_summary.json').read_text())['raw_chunk_identity']}
    dump(root / 'reports/paired.validation.json', {'status': 'passed', 'source_root': str(source_root), 'datasets': paired})
    relative_runner = os.path.relpath(Path(__file__).resolve(), root)
    (root / 'README.md').write_text(
        '# Apple 配对语料\n\n'
        f'父语料：`{source_root}`。使用 NanoSignalPrep Apple，包含末尾软限幅。'
        '对原始、已换算成 pA 的完整有效 read 处理，再按同一采样点边界切 chunk；不把预标准化 NPY 再次输入 Apple。'
        'RNA 使用上游已去 polyA 的 basecall 原始信号，与父语料保持相同支持范围。\n\n'
        '[结果](reports/summary.md) · [图表](reports/PLOTS.md) · [配对核验](reports/paired.validation.json) · '
        '[NanoRepDist 模板](config/nanorepdist.example.yaml)\n\n'
        '所有 manifest 与父语料逐字节一致；加载时以本目录为语料根目录。'
        '信号保存为 float32，原 read 身份、有效长度、chunk 边界和样本坐标不变。'
        '校准成员和 phase 沿用父语料，Apple k-mer 电流统计重新计算；这不构成新的 phase 搜索。\n\n'
        f'仍使用唯一入口：在本目录运行 `python {relative_runner} verify --root .`；'
        f'重画使用 `python {relative_runner} plot --root . --overwrite`。\n\n'
        '`config/apple.json` 保存原始文件映射、指纹、实现和父语料哈希；'
        '各域 `prep/raw_sources.jsonl` 记录每条原始信号哈希、裁剪坐标和与父信号的仿射一致性检查。'
        'R10 父输入呈固定均值/尺度的仿射标准化，不能称为已确认的逐 read Stone MAD；'
        '本版本四域统一从 pA 按 Apple 重算。\n')
    plot_corpus(root, project, overwrite=True)
    count = write_checksums(root)
    print(json.dumps({'status': 'passed', 'normal_mode': 'apple', 'root': str(root),
                      'datasets': paired, 'checksum_files': count}, ensure_ascii=False, indent=2), flush=True)


def main(argv=None):
    runner_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Prepare and freeze a new corpus version')
    build.add_argument('--root', type=Path, required=True, help='Output directory; frozen versions cannot be overwritten')
    build.add_argument('--project', type=Path, default=runner_dir.parents[1])
    build.add_argument('--inventory', type=Path, default=runner_dir / 'stone/config/sources.inventory.json')
    build.add_argument('--seed', type=int, default=20260916)
    build.add_argument('--pool-size', type=int, default=12000)
    build.add_argument('--cap', type=int, default=100)
    build.add_argument('--workers', type=int, default=2)
    build.add_argument('--dataset', action='append', help='Select dataset name; repeat to select multiple')
    stages = build.add_mutually_exclusive_group()
    stages.add_argument('--prepare-only', action='store_true')
    stages.add_argument('--freeze-only', action='store_true')
    check = commands.add_parser('verify', help='Read-only checksum and full manifest verification')
    check_root = check.add_mutually_exclusive_group()
    check_root.add_argument('--root', type=Path, help='Explicit corpus directory')
    check_root.add_argument('--strategy', choices=('stone', 'apple'), default='stone',
                            help='Corpus strategy (default: stone)')
    check.add_argument('--checksums-only', action='store_true', help='Skip the slower sample-to-source comparison')
    plot = commands.add_parser('plot', help='Plot existing k-mer tables with prefix dividers')
    plot_root = plot.add_mutually_exclusive_group()
    plot_root.add_argument('--root', type=Path, help='Explicit corpus directory')
    plot_root.add_argument('--strategy', choices=('stone', 'apple'), default='stone',
                           help='Corpus strategy (default: stone)')
    plot.add_argument('--project', type=Path, default=runner_dir.parents[1])
    plot.add_argument('--overwrite', action='store_true', help='Replace existing derived figures')
    apple = commands.add_parser('apple', help='Create a paired Apple variant from original FAST5/POD5 reads')
    apple.add_argument('--source-root', type=Path, default=runner_dir / 'stone')
    apple.add_argument('--out-dir', type=Path, default=runner_dir / 'apple')
    apple.add_argument('--project', type=Path, default=runner_dir.parents[1])
    apple.add_argument('--raw-config', type=Path, default=runner_dir / 'apple/config/apple.sources.json')
    apple.add_argument('--workers', type=int, default=8)
    apple.add_argument('--dataset', action='append')
    args = parser.parse_args(argv)
    if args.command in {'verify', 'plot'} and args.root is None:
        args.root = runner_dir / args.strategy
    try:
        if args.command == 'build':
            build_corpus(args)
        elif args.command == 'apple':
            build_apple_corpus(args)
        elif args.command == 'plot':
            plot_corpus(args.root, args.project, args.overwrite)
        else:
            root = args.root.resolve()
            count = verify_checksums(root)
            print(f'SHA256 verified: {count} files', flush=True)
            if not args.checksums_only:
                print(json.dumps(verify(root), indent=2, ensure_ascii=False), flush=True)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'{args.command}: {exc}\n')


if __name__ == '__main__':
    main()
