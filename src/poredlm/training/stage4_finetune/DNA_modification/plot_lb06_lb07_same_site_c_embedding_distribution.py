#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_c_modification_embedding_distribution import (  # noqa: E402
    BasecallModel,
    ReadRecord,
    build_model,
    forward_sequence_hidden,
    iter_records,
    make_batch,
    normalize_embedding_source,
    pca_2d,
)


DEFAULT_LB07_JSONL = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/"
    "all_data/split_LB07_only/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
)
DEFAULT_LB06_JSONL = (
    "/mnt/zzbnew/rnamodel/zhoukexuan/PoreDLM/data/DNA_modifiction/LB07_AND_LB06/"
    "stage4_modification/validation_seq1_to_seq17_ref_target_cropped_token_c_modlabel.jsonl.gz"
)


@dataclass
class EmbeddingRow:
    dataset: str
    sequence_key: str
    read_id: str
    row_index: int
    token_position: int
    label: int
    embedding: np.ndarray
    site_key: str | None = None
    base_position_1based: int | None = None
    base: str | None = None
    kmer: str | None = None
    token_positions: list[int] | None = None


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))[:180]


def sequence_key_for_record(record: ReadRecord, mode: str) -> str:
    meta = record.meta or {}
    if mode == "label":
        value = meta.get("label")
    elif mode == "ref":
        value = meta.get("ref")
    elif mode == "seq":
        value = meta.get("seq")
    elif mode == "auto":
        value = meta.get("label") or meta.get("ref") or meta.get("seq")
    else:
        raise ValueError(f"Unsupported sequence key mode: {mode}")
    if value is None or str(value) == "":
        return "unknown_sequence"
    return str(value)


def parse_maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_base_types(value: str) -> set[str]:
    bases = {part.strip().upper() for part in value.split(",") if part.strip()}
    allowed = {"A", "C", "G", "T"}
    unknown = bases - allowed
    if unknown:
        raise ValueError(f"Unsupported base types: {sorted(unknown)}; allowed bases are A,C,G,T.")
    if not bases:
        raise ValueError("--base-types must contain at least one of A,C,G,T.")
    return bases


def sample_span_to_token_span(
    start_sample: int,
    end_sample: int,
    *,
    samples_per_token: int,
    token_count: int,
) -> tuple[int, int]:
    import math

    token_start = max(0, int(start_sample) // int(samples_per_token))
    token_end = min(token_count, int(math.ceil(float(end_sample) / float(samples_per_token))))
    return token_start, max(token_start, token_end)


def sample_span_to_center_token(
    start_sample: int,
    end_sample: int,
    *,
    samples_per_token: int,
    token_count: int,
) -> int | None:
    token_start, token_end = sample_span_to_token_span(
        start_sample,
        end_sample,
        samples_per_token=samples_per_token,
        token_count=token_count,
    )
    if token_end <= token_start:
        return None
    return int((token_start + token_end - 1) // 2)


def collect_dataset_embeddings(
    args: argparse.Namespace,
    *,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    dataset_name: str,
    embedding_source: str,
    wanted_positions_by_sequence: dict[str, set[int]] | None,
    site_by_sequence_token: dict[str, dict[int, str]] | None,
    base_position_by_site: dict[str, int] | None,
    wanted_label_values: set[int],
) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    batch: list[ReadRecord] = []
    pbar = tqdm(total=len(records), desc=f"extracting {dataset_name} {args.embedding_source}", unit="read")

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        input_ids, attention_mask, effective_lengths = make_batch(
            batch,
            pad_token_id=args.pad_token_id,
            max_length=args.max_length,
            device=device,
        )
        with torch.inference_mode():
            hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=embedding_source,
            ).float().detach().cpu().numpy()

        for idx, record in enumerate(batch):
            sequence_key = sequence_key_for_record(record, args.sequence_key)
            wanted_positions = None
            if wanted_positions_by_sequence is not None:
                wanted_positions = wanted_positions_by_sequence.get(sequence_key, set())
                if not wanted_positions:
                    continue
            token_count = min(effective_lengths[idx], hidden.shape[1], len(record.c_modification_label))
            for token_position, label in enumerate(record.c_modification_label[:token_count]):
                label = int(label)
                if label not in wanted_label_values:
                    continue
                if wanted_positions is not None and token_position not in wanted_positions:
                    continue
                site_key = None
                base_position = None
                if site_by_sequence_token is not None:
                    site_key = site_by_sequence_token.get(sequence_key, {}).get(int(token_position))
                    if site_key is not None and base_position_by_site is not None:
                        base_position = base_position_by_site.get(site_key)
                rows.append(
                    EmbeddingRow(
                        dataset=dataset_name,
                        sequence_key=sequence_key,
                        read_id=record.record_id,
                        row_index=record.row_index,
                        token_position=int(token_position),
                        label=label,
                        embedding=hidden[idx, token_position].astype(np.float32),
                        site_key=site_key,
                        base_position_1based=base_position,
                    )
                )
        pbar.update(len(batch))
        batch = []

    for record in records:
        batch.append(record)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()
    return rows


def collect_label_positions(
    records: list[ReadRecord],
    *,
    sequence_key_mode: str,
    label_values: set[int],
) -> dict[str, set[int]]:
    positions: dict[str, set[int]] = defaultdict(set)
    for record in records:
        sequence_key = sequence_key_for_record(record, sequence_key_mode)
        for token_position, label in enumerate(record.c_modification_label):
            if int(label) in label_values:
                positions[sequence_key].add(int(token_position))
    return positions


def interleave_dataset_items(
    datasets: list[tuple[str, list[ReadRecord], dict[str, set[int]] | None, set[int]]]
) -> list[tuple[str, ReadRecord, dict[str, set[int]] | None, set[int]]]:
    items: list[tuple[str, ReadRecord, dict[str, set[int]] | None, set[int]]] = []
    max_len = max((len(records) for _name, records, _wanted_positions, _labels in datasets), default=0)
    for record_index in range(max_len):
        for dataset_name, records, wanted_positions_by_sequence, wanted_label_values in datasets:
            if record_index < len(records):
                items.append((dataset_name, records[record_index], wanted_positions_by_sequence, wanted_label_values))
    return items


def collect_mixed_dataset_embeddings(
    args: argparse.Namespace,
    *,
    model: BasecallModel,
    device: torch.device,
    datasets: list[tuple[str, list[ReadRecord], dict[str, set[int]] | None, set[int]]],
    embedding_source: str,
    site_by_sequence_token: dict[str, dict[int, str]] | None,
    base_position_by_site: dict[str, int] | None,
) -> dict[str, list[EmbeddingRow]]:
    rows_by_dataset: dict[str, list[EmbeddingRow]] = defaultdict(list)
    items = interleave_dataset_items(datasets)
    batch: list[tuple[str, ReadRecord, dict[str, set[int]] | None, set[int]]] = []
    pbar = tqdm(total=len(items), desc=f"extracting mixed c-mod-sites {args.embedding_source}", unit="read")

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        batch_records = [record for _dataset_name, record, _wanted_positions, _labels in batch]
        input_ids, attention_mask, effective_lengths = make_batch(
            batch_records,
            pad_token_id=args.pad_token_id,
            max_length=args.max_length,
            device=device,
        )
        with torch.inference_mode():
            hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=embedding_source,
            ).float().detach().cpu().numpy()

        for idx, (dataset_name, record, wanted_positions_by_sequence, wanted_label_values) in enumerate(batch):
            sequence_key = sequence_key_for_record(record, args.sequence_key)
            wanted_positions = None
            if wanted_positions_by_sequence is not None:
                wanted_positions = wanted_positions_by_sequence.get(sequence_key, set())
                if not wanted_positions:
                    continue
            token_count = min(effective_lengths[idx], hidden.shape[1], len(record.c_modification_label))
            for token_position, label in enumerate(record.c_modification_label[:token_count]):
                label = int(label)
                if label not in wanted_label_values:
                    continue
                if wanted_positions is not None and token_position not in wanted_positions:
                    continue
                site_key = None
                base_position = None
                if site_by_sequence_token is not None:
                    site_key = site_by_sequence_token.get(sequence_key, {}).get(int(token_position))
                    if site_key is not None and base_position_by_site is not None:
                        base_position = base_position_by_site.get(site_key)
                rows_by_dataset[dataset_name].append(
                    EmbeddingRow(
                        dataset=dataset_name,
                        sequence_key=sequence_key,
                        read_id=record.record_id,
                        row_index=record.row_index,
                        token_position=int(token_position),
                        label=label,
                        embedding=hidden[idx, token_position].astype(np.float32),
                        site_key=site_key,
                        base_position_1based=base_position,
                    )
                )
        pbar.update(len(batch))
        batch = []

    for item in items:
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()
    return rows_by_dataset


def collect_base_type_embeddings(
    args: argparse.Namespace,
    *,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    dataset_name: str,
    embedding_source: str,
    base_types: set[str],
) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    batch: list[ReadRecord] = []
    pbar = tqdm(total=len(records), desc=f"extracting {dataset_name} bases {''.join(sorted(base_types))}", unit="read")

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        input_ids, attention_mask, effective_lengths = make_batch(
            batch,
            pad_token_id=args.pad_token_id,
            max_length=args.max_length,
            device=device,
        )
        with torch.inference_mode():
            hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=embedding_source,
            ).float().detach().cpu().numpy()

        for idx, record in enumerate(batch):
            meta = record.meta or {}
            ref = meta.get("ref")
            spans = parse_maybe_json(meta.get("base_sample_span_ref"))
            if not isinstance(ref, str) or not isinstance(spans, list):
                continue

            sequence_key = sequence_key_for_record(record, args.sequence_key)
            token_count = min(effective_lengths[idx], hidden.shape[1])
            max_base_count = min(len(ref), len(spans))
            center_candidates: dict[int, list[tuple[int, str]]] = defaultdict(list)
            for base_index in range(max_base_count):
                base = ref[base_index].upper()
                if base not in base_types:
                    continue
                span = spans[base_index]
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    continue
                start_sample, end_sample = span
                if start_sample is None or end_sample is None:
                    continue
                try:
                    start_sample = int(start_sample)
                    end_sample = int(end_sample)
                except (TypeError, ValueError):
                    continue
                if args.base_token_mode == "all-overlap":
                    token_start, token_end = sample_span_to_token_span(
                        start_sample,
                        end_sample,
                        samples_per_token=args.samples_per_token,
                        token_count=token_count,
                    )
                    for token_position in range(token_start, token_end):
                        rows.append(
                            EmbeddingRow(
                                dataset=dataset_name,
                                sequence_key=sequence_key,
                                read_id=record.record_id,
                                row_index=record.row_index,
                                token_position=int(token_position),
                                label=0,
                                embedding=hidden[idx, token_position].astype(np.float32),
                                base_position_1based=base_index + 1,
                                base=base,
                            )
                        )
                    continue

                center_token = sample_span_to_center_token(
                    start_sample,
                    end_sample,
                    samples_per_token=args.samples_per_token,
                    token_count=token_count,
                )
                if center_token is None:
                    continue
                center_candidates[center_token].append((base_index, base))

            if args.base_token_mode == "center-unique":
                for token_position, candidates in sorted(center_candidates.items()):
                    if len(candidates) != 1:
                        continue
                    base_index, base = candidates[0]
                    rows.append(
                        EmbeddingRow(
                            dataset=dataset_name,
                            sequence_key=sequence_key,
                            read_id=record.record_id,
                            row_index=record.row_index,
                            token_position=int(token_position),
                            label=0,
                            embedding=hidden[idx, token_position].astype(np.float32),
                            base_position_1based=base_index + 1,
                            base=base,
                        )
                    )
        pbar.update(len(batch))
        batch = []

    for record in records:
        batch.append(record)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()
    return rows


def load_limited_records(path: str, *, limit_reads: int | None) -> list[ReadRecord]:
    records = []
    for record in iter_records(Path(path)):
        if limit_reads is not None and len(records) >= limit_reads:
            break
        records.append(record)
    return records


def modified_positions_from_rows(lb06_rows: list[EmbeddingRow]) -> dict[str, set[int]]:
    positions: dict[str, set[int]] = defaultdict(set)
    for row in lb06_rows:
        if row.label == 2:
            positions[row.sequence_key].add(row.token_position)
    return positions


def collect_modified_base_positions_from_records(
    records: list[ReadRecord],
    *,
    sequence_key_mode: str,
) -> dict[str, set[int]]:
    positions: dict[str, set[int]] = defaultdict(set)
    for record in records:
        sequence_key = sequence_key_for_record(record, sequence_key_mode)
        for _site_key, _token_start, _token_end, base_position in parse_modified_c_token_spans(record, sequence_key):
            if base_position is not None:
                positions[sequence_key].add(int(base_position))
    return positions


def collect_kmer_embeddings(
    args: argparse.Namespace,
    *,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    dataset_name: str,
    embedding_source: str,
    modified_base_positions_by_sequence: dict[str, set[int]],
) -> list[EmbeddingRow]:
    rows: list[EmbeddingRow] = []
    batch: list[ReadRecord] = []
    pbar = tqdm(total=len(records), desc=f"extracting {dataset_name} 5-mer {args.embedding_source}", unit="read")

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        input_ids, attention_mask, effective_lengths = make_batch(
            batch,
            pad_token_id=args.pad_token_id,
            max_length=args.max_length,
            device=device,
        )
        with torch.inference_mode():
            hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=embedding_source,
            ).float().detach().cpu().numpy()

        for idx, record in enumerate(batch):
            sequence_key = sequence_key_for_record(record, args.sequence_key)
            modified_base_positions = modified_base_positions_by_sequence.get(sequence_key, set())
            if not modified_base_positions:
                continue

            meta = record.meta or {}
            ref = meta.get("ref")
            spans = parse_maybe_json(meta.get("base_sample_span_ref"))
            if not isinstance(ref, str) or not isinstance(spans, list):
                continue

            token_count = min(effective_lengths[idx], hidden.shape[1])
            for base_position in sorted(modified_base_positions):
                center_index = int(base_position) - 1
                start_index = center_index - 2
                end_index = center_index + 3
                if start_index < 0 or end_index > len(ref) or end_index > len(spans):
                    continue

                token_positions: list[int] = []
                skip_site = False
                for base_index in range(start_index, end_index):
                    span = spans[base_index]
                    if not isinstance(span, (list, tuple)) or len(span) != 2:
                        skip_site = True
                        break
                    start_sample, end_sample = span
                    if start_sample is None or end_sample is None:
                        skip_site = True
                        break
                    try:
                        token_start, token_end = sample_span_to_token_span(
                            int(start_sample),
                            int(end_sample),
                            samples_per_token=args.samples_per_token,
                            token_count=token_count,
                        )
                    except (TypeError, ValueError):
                        skip_site = True
                        break
                    if token_end <= token_start:
                        skip_site = True
                        break
                    token_positions.extend(range(token_start, token_end))

                if skip_site:
                    continue
                token_positions = sorted(set(token_positions))
                if len(token_positions) < args.min_kmer_tokens:
                    continue

                pooled_embedding = hidden[idx, token_positions].mean(axis=0).astype(np.float32)
                kmer = ref[start_index:end_index].upper()
                site_key = f"{sequence_key}:base_{int(base_position)}:5mer"
                rows.append(
                    EmbeddingRow(
                        dataset=dataset_name,
                        sequence_key=sequence_key,
                        read_id=record.record_id,
                        row_index=record.row_index,
                        token_position=int(token_positions[len(token_positions) // 2]),
                        label=2 if dataset_name.startswith("LB06") else 1,
                        embedding=pooled_embedding,
                        site_key=site_key,
                        base_position_1based=int(base_position),
                        base=ref[center_index].upper(),
                        kmer=kmer,
                        token_positions=token_positions,
                    )
                )
        pbar.update(len(batch))
        batch = []

    for record in records:
        batch.append(record)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()
    return rows


def parse_modified_c_token_spans(record: ReadRecord, sequence_key: str) -> list[tuple[str, int, int, int | None]]:
    spans = (record.meta or {}).get("modified_c_token_spans")
    if not isinstance(spans, list):
        return []

    parsed: list[tuple[str, int, int, int | None]] = []
    for span in spans:
        if not isinstance(span, (list, tuple)):
            continue
        try:
            if len(span) >= 5:
                base_position = int(span[0])
                token_start = int(span[-2])
                token_end = int(span[-1])
            elif len(span) >= 3:
                base_position = int(span[0])
                token_start = int(span[1])
                token_end = int(span[2])
            else:
                continue
        except (TypeError, ValueError):
            continue
        if token_end <= token_start:
            continue
        site_key = f"{sequence_key}:base_{base_position}"
        parsed.append((site_key, token_start, token_end, base_position))
    return parsed


def contiguous_runs(values: set[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    sorted_values = sorted(values)
    runs = []
    start = sorted_values[0]
    previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous + 1))
        start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def build_modified_site_maps(
    records: list[ReadRecord],
    *,
    sequence_key_mode: str,
    fallback_positions_by_sequence: dict[str, set[int]],
) -> tuple[dict[str, dict[str, set[int]]], dict[str, dict[int, str]], dict[str, int]]:
    positions_by_site: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    site_by_sequence_token: dict[str, dict[int, str]] = defaultdict(dict)
    base_position_by_site: dict[str, int] = {}

    for record in records:
        sequence_key = sequence_key_for_record(record, sequence_key_mode)
        for site_key, token_start, token_end, base_position in parse_modified_c_token_spans(record, sequence_key):
            if base_position is not None:
                base_position_by_site[site_key] = base_position
            for token_position in range(token_start, token_end):
                positions_by_site[sequence_key][site_key].add(token_position)
                site_by_sequence_token[sequence_key][token_position] = site_key

    if positions_by_site:
        return positions_by_site, site_by_sequence_token, base_position_by_site

    for sequence_key, positions in fallback_positions_by_sequence.items():
        for site_index, (token_start, token_end) in enumerate(contiguous_runs(positions), start=1):
            site_key = f"{sequence_key}:token_run_{site_index}_{token_start}_{token_end}"
            for token_position in range(token_start, token_end):
                positions_by_site[sequence_key][site_key].add(token_position)
                site_by_sequence_token[sequence_key][token_position] = site_key
    return positions_by_site, site_by_sequence_token, base_position_by_site


def sample_rows(rows: list[EmbeddingRow], max_rows: int, rng: random.Random) -> list[EmbeddingRow]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), max_rows))]


def stack_embeddings(rows: list[EmbeddingRow]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    x = np.stack([row.embedding for row in rows], axis=0).astype(np.float32)
    y = np.asarray([1 if row.dataset.startswith("LB06") else 0 for row in rows], dtype=np.int64)
    return x, y


def stack_embeddings_by_c_label(rows: list[EmbeddingRow]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    x = np.stack([row.embedding for row in rows], axis=0).astype(np.float32)
    y = np.asarray([1 if int(row.label) == 2 else 0 for row in rows], dtype=np.int64)
    return x, y


def site_label_for_row(row: EmbeddingRow) -> str:
    if row.base_position_1based is not None:
        return f"base_{int(row.base_position_1based)}"
    if row.site_key:
        return str(row.site_key).split(":", 1)[-1]
    return f"token_{int(row.token_position)}"


def plot_rows(
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
    class0_label: str = "LB07 unmodified same-site C",
    class1_label: str = "LB06 modified C",
    pca_title: str = "same-site C PCA",
    annotate_site_labels: str = "none",
    modified_color_by: str = "class",
    max_read_legend_items: int = 30,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    x, y = stack_embeddings(rows)
    coords = pca_2d(x)
    norms = np.linalg.norm(x, axis=1) if x.size else np.empty((0,), dtype=np.float32)
    lb07_mask = y == 0
    lb06_mask = y == 1

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    if lb07_mask.any():
        ax.scatter(
            coords[lb07_mask, 0],
            coords[lb07_mask, 1],
            s=10,
            alpha=0.32,
            color="#2563eb",
            label=f"{class0_label} ({int(lb07_mask.sum())})",
        )
    read_color_counts: dict[str, int] = {}
    if lb06_mask.any() and modified_color_by == "read":
        lb06_indices = np.flatnonzero(lb06_mask)
        read_ids = sorted({rows[int(row_index)].read_id for row_index in lb06_indices})
        read_color_counts = {
            read_id: int(sum(1 for row_index in lb06_indices if rows[int(row_index)].read_id == read_id))
            for read_id in read_ids
        }
        cmap = plt.get_cmap("tab20" if len(read_ids) <= 20 else "turbo")
        color_denominator = max(len(read_ids) - 1, 1)
        show_read_legend = max_read_legend_items > 0
        for read_index, read_id in enumerate(read_ids):
            row_indices = [int(row_index) for row_index in lb06_indices if rows[int(row_index)].read_id == read_id]
            if not row_indices:
                continue
            color = cmap(read_index / color_denominator)
            label = f"{read_id} ({len(row_indices)})" if show_read_legend and read_index < max_read_legend_items else None
            ax.scatter(
                coords[row_indices, 0],
                coords[row_indices, 1],
                s=28,
                alpha=0.85,
                color=color,
                edgecolors="black",
                linewidths=0.2,
                marker="*",
                label=label,
            )
        if show_read_legend and len(read_ids) > max_read_legend_items:
            ax.scatter([], [], s=28, color="#737373", marker="*", label=f"other reads ({len(read_ids) - max_read_legend_items})")
    elif lb06_mask.any():
        ax.scatter(
            coords[lb06_mask, 0],
            coords[lb06_mask, 1],
            s=28,
            alpha=0.85,
            color="#dc2626",
            edgecolors="black",
            linewidths=0.2,
            marker="*",
            label=f"{class1_label} ({int(lb06_mask.sum())})",
        )
    if lb06_mask.any() and annotate_site_labels == "modified-points":
        for row_index in np.flatnonzero(lb06_mask):
            ax.annotate(
                site_label_for_row(rows[int(row_index)]),
                (coords[row_index, 0], coords[row_index, 1]),
                fontsize=5,
                alpha=0.72,
                xytext=(2, 2),
                textcoords="offset points",
            )
    elif lb06_mask.any() and annotate_site_labels == "modified-centroids":
        site_to_indices: dict[str, list[int]] = defaultdict(list)
        for row_index in np.flatnonzero(lb06_mask):
            site_to_indices[site_label_for_row(rows[int(row_index)])].append(int(row_index))
        for site_label, row_indices in sorted(site_to_indices.items()):
            site_coords = coords[row_indices]
            center = site_coords.mean(axis=0)
            ax.annotate(
                site_label,
                (float(center[0]), float(center[1])),
                fontsize=7,
                fontweight="bold",
                color="#7f1d1d",
                xytext=(4, 4),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#dc2626", "alpha": 0.75, "lw": 0.5},
                arrowprops={"arrowstyle": "-", "color": "#dc2626", "alpha": 0.45, "lw": 0.5},
            )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_label} {pca_title}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    if lb07_mask.any():
        ax2.hist(norms[lb07_mask], bins=70, alpha=0.62, color="#2563eb", density=True, label=class0_label)
    if lb06_mask.any():
        ax2.hist(norms[lb06_mask], bins=45, alpha=0.72, color="#dc2626", density=True, label=class1_label)
        ax2.scatter(norms[lb06_mask], np.zeros(int(lb06_mask.sum())), s=22, color="#dc2626", marker="|")
    ax2.set_xlabel(f"{embedding_label} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "num_points": int(len(rows)),
        "class0_label": class0_label,
        "class1_label": class1_label,
        "class0_points": int(lb07_mask.sum()),
        "class1_points": int(lb06_mask.sum()),
        "lb07_unmodified_same_site_points": int(lb07_mask.sum()),
        "lb06_modified_points": int(lb06_mask.sum()),
        "lb07_norm_mean": float(norms[lb07_mask].mean()) if lb07_mask.any() else None,
        "lb06_norm_mean": float(norms[lb06_mask].mean()) if lb06_mask.any() else None,
        "lb07_norm_p95": float(np.quantile(norms[lb07_mask], 0.95)) if lb07_mask.any() else None,
        "lb06_norm_p95": float(np.quantile(norms[lb06_mask], 0.95)) if lb06_mask.any() else None,
        "annotate_site_labels": annotate_site_labels,
        "modified_color_by": modified_color_by,
        "read_color_counts": read_color_counts,
        "max_read_legend_items": int(max_read_legend_items),
    }


def plot_c_label_rows(
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
    class0_label: str = "LB06 normal C",
    class1_label: str = "LB06 modified C",
    pca_title: str = "LB06 C-label PCA",
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    x, y = stack_embeddings_by_c_label(rows)
    coords = pca_2d(x)
    norms = np.linalg.norm(x, axis=1) if x.size else np.empty((0,), dtype=np.float32)
    normal_mask = y == 0
    modified_mask = y == 1

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    if normal_mask.any():
        ax.scatter(
            coords[normal_mask, 0],
            coords[normal_mask, 1],
            s=10,
            alpha=0.30,
            color="#2563eb",
            label=f"{class0_label} ({int(normal_mask.sum())})",
        )
    if modified_mask.any():
        ax.scatter(
            coords[modified_mask, 0],
            coords[modified_mask, 1],
            s=28,
            alpha=0.85,
            color="#dc2626",
            edgecolors="black",
            linewidths=0.2,
            marker="*",
            label=f"{class1_label} ({int(modified_mask.sum())})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_label} {pca_title}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    if normal_mask.any():
        ax2.hist(norms[normal_mask], bins=70, alpha=0.62, color="#2563eb", density=True, label=class0_label)
    if modified_mask.any():
        ax2.hist(norms[modified_mask], bins=45, alpha=0.72, color="#dc2626", density=True, label=class1_label)
        ax2.scatter(norms[modified_mask], np.zeros(int(modified_mask.sum())), s=22, color="#dc2626", marker="|")
    ax2.set_xlabel(f"{embedding_label} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "num_points": int(len(rows)),
        "class0_label": class0_label,
        "class1_label": class1_label,
        "class0_points": int(normal_mask.sum()),
        "class1_points": int(modified_mask.sum()),
        "lb06_normal_c_points": int(normal_mask.sum()),
        "lb06_modified_c_points": int(modified_mask.sum()),
        "normal_c_norm_mean": float(norms[normal_mask].mean()) if normal_mask.any() else None,
        "modified_c_norm_mean": float(norms[modified_mask].mean()) if modified_mask.any() else None,
        "normal_c_norm_p95": float(np.quantile(norms[normal_mask], 0.95)) if normal_mask.any() else None,
        "modified_c_norm_p95": float(np.quantile(norms[modified_mask], 0.95)) if modified_mask.any() else None,
    }


def plot_lb06_read_modified_site_rows(
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
    pca_title: str = "LB06 read modified-site PCA",
    summary_read_id: bool = True,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("plot_lb06_read_modified_site_rows got no rows.")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.stack([row.embedding for row in rows], axis=0).astype(np.float32)
    coords = pca_2d(x)
    norms = np.linalg.norm(x, axis=1)
    groups = np.asarray([site_label_for_row(row) for row in rows], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    group_counts = {group: int((groups == group).sum()) for group in unique_groups}
    group_norm_mean = {
        group: float(norms[groups == group].mean()) if (groups == group).any() else None
        for group in unique_groups
    }
    group_norm_p95 = {
        group: float(np.quantile(norms[groups == group], 0.95)) if (groups == group).any() else None
        for group in unique_groups
    }

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    for group_index, group in enumerate(unique_groups):
        mask = groups == group
        color = cmap(group_index % 10)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=48,
            alpha=0.85,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            label=f"{group} ({group_counts[group]})",
        )
        center = coords[mask].mean(axis=0)
        ax.annotate(
            group,
            (float(center[0]), float(center[1])),
            fontsize=7,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.72, "lw": 0.5},
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_label} {pca_title}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    for group_index, group in enumerate(unique_groups):
        mask = groups == group
        ax2.hist(
            norms[mask],
            bins=min(20, max(3, int(mask.sum()))),
            alpha=0.45,
            color=cmap(group_index % 10),
            density=True,
            label=group,
        )
        ax2.scatter(norms[mask], np.zeros(int(mask.sum())), s=18, color=cmap(group_index % 10), marker="|")
    ax2.set_xlabel(f"{embedding_label} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    summary = {
        "output_png": str(output_png),
        "num_points": int(len(rows)),
        "sequence_key": rows[0].sequence_key,
        "site_counts": group_counts,
        "site_norm_mean": group_norm_mean,
        "site_norm_p95": group_norm_p95,
    }
    if summary_read_id:
        summary["read_id"] = rows[0].read_id
    return summary


def dataset_base_group(row: EmbeddingRow) -> str:
    dataset = "LB06" if row.dataset.startswith("LB06") else "LB07"
    base = row.base or "N"
    return f"{dataset}-{base}"


def base_group(row: EmbeddingRow) -> str:
    return row.base or "N"


def plot_rows_by_dataset_base(
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
    pca_title: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("plot_rows_by_dataset_base got no rows.")

    x = np.stack([row.embedding for row in rows], axis=0).astype(np.float32)
    coords = pca_2d(x)
    norms = np.linalg.norm(x, axis=1)
    groups = np.asarray([dataset_base_group(row) for row in rows])
    unique_groups = sorted(set(groups.tolist()))

    palette = {
        "LB07-A": "#2563eb",
        "LB07-C": "#0891b2",
        "LB07-G": "#16a34a",
        "LB07-T": "#7c3aed",
        "LB06-A": "#dc2626",
        "LB06-C": "#ea580c",
        "LB06-G": "#ca8a04",
        "LB06-T": "#be123c",
    }
    markers = {
        "LB07-A": "o",
        "LB07-C": "s",
        "LB07-G": "^",
        "LB07-T": "D",
        "LB06-A": "*",
        "LB06-C": "P",
        "LB06-G": "X",
        "LB06-T": "v",
    }

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    group_counts: dict[str, int] = {}
    group_norm_mean: dict[str, float | None] = {}
    group_norm_p95: dict[str, float | None] = {}
    for group in unique_groups:
        mask = groups == group
        group_counts[group] = int(mask.sum())
        group_norm_mean[group] = float(norms[mask].mean()) if mask.any() else None
        group_norm_p95[group] = float(np.quantile(norms[mask], 0.95)) if mask.any() else None
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=30 if group.startswith("LB06") else 13,
            alpha=0.82 if group.startswith("LB06") else 0.35,
            color=palette.get(group, "#525252"),
            marker=markers.get(group, "o"),
            edgecolors="black" if group.startswith("LB06") else "none",
            linewidths=0.2,
            label=f"{group} ({group_counts[group]})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_label} {pca_title}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    for group in unique_groups:
        mask = groups == group
        ax2.hist(
            norms[mask],
            bins=45,
            alpha=0.38 if group.startswith("LB07") else 0.50,
            color=palette.get(group, "#525252"),
            density=True,
            label=group,
        )
    ax2.set_xlabel(f"{embedding_label} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "num_points": int(len(rows)),
        "color_by": "dataset_base",
        "group_counts": group_counts,
        "group_norm_mean": group_norm_mean,
        "group_norm_p95": group_norm_p95,
    }


def plot_rows_by_base(
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
    pca_title: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("plot_rows_by_base got no rows.")

    x = np.stack([row.embedding for row in rows], axis=0).astype(np.float32)
    coords = pca_2d(x)
    norms = np.linalg.norm(x, axis=1)
    groups = np.asarray([base_group(row) for row in rows])
    unique_groups = [base for base in ("A", "C", "G", "T", "N") if base in set(groups.tolist())]

    palette = {
        "A": "#2563eb",
        "C": "#dc2626",
        "G": "#16a34a",
        "T": "#7c3aed",
        "N": "#525252",
    }
    markers = {
        "A": "o",
        "C": "*",
        "G": "^",
        "T": "D",
        "N": "x",
    }

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    ax = axes[0]
    group_counts: dict[str, int] = {}
    group_norm_mean: dict[str, float | None] = {}
    group_norm_p95: dict[str, float | None] = {}
    for group in unique_groups:
        mask = groups == group
        group_counts[group] = int(mask.sum())
        group_norm_mean[group] = float(norms[mask].mean()) if mask.any() else None
        group_norm_p95[group] = float(np.quantile(norms[mask], 0.95)) if mask.any() else None
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.45,
            color=palette.get(group, "#525252"),
            marker=markers.get(group, "o"),
            edgecolors="black" if group == "C" else "none",
            linewidths=0.2,
            label=f"{group} ({group_counts[group]})",
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"{embedding_label} {pca_title}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    for group in unique_groups:
        mask = groups == group
        ax2.hist(
            norms[mask],
            bins=45,
            alpha=0.42,
            color=palette.get(group, "#525252"),
            density=True,
            label=group,
        )
    ax2.set_xlabel(f"{embedding_label} L2 norm")
    ax2.set_ylabel("density")
    ax2.set_title("Embedding norm distribution")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "num_points": int(len(rows)),
        "color_by": "base",
        "group_counts": group_counts,
        "group_norm_mean": group_norm_mean,
        "group_norm_p95": group_norm_p95,
    }


def plot_base_rows(
    args: argparse.Namespace,
    rows: list[EmbeddingRow],
    output_png: Path,
    *,
    title: str,
    embedding_label: str,
    class0_label: str,
    class1_label: str,
    pca_title: str,
) -> dict[str, Any]:
    if args.color_by == "dataset_base":
        return plot_rows_by_dataset_base(
            rows,
            output_png,
            title=title,
            embedding_label=embedding_label,
            pca_title=pca_title,
        )
    return plot_rows(
        rows,
        output_png,
        title=title,
        embedding_label=embedding_label,
        class0_label=class0_label,
        class1_label=class1_label,
        pca_title=pca_title,
    )


def save_points(rows: list[EmbeddingRow], points_path: Path) -> None:
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "dataset": row.dataset,
                        "sequence_key": row.sequence_key,
                        "read_id": row.read_id,
                        "row_index": row.row_index,
                        "token_position": row.token_position,
                        "c_modification_label": row.label,
                        "site_key": row.site_key,
                        "base_position_1based": row.base_position_1based,
                        "base": row.base,
                        "kmer": row.kmer,
                        "token_positions": row.token_positions,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def summarize_positions(positions_by_sequence: dict[str, set[int]]) -> dict[str, Any]:
    return {
        sequence_key: {
            "num_modified_token_positions": len(positions),
            "modified_token_positions": sorted(positions),
        }
        for sequence_key, positions in sorted(positions_by_sequence.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LB06 modified C embeddings against LB07 unmodified C embeddings at the same "
            "sequence key and token positions. Uses c_modification_label: 1=unmodified C, 2=modified C."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument("--lb07-jsonl", default=DEFAULT_LB07_JSONL)
    parser.add_argument("--lb06-jsonl", default=DEFAULT_LB06_JSONL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--compare-mode",
        choices=("c-mod-sites", "c-mod-5mer", "lb06-c-labels", "base", "single-dataset-base"),
        default="c-mod-sites",
        help=(
            "c-mod-sites compares LB06 modified C with LB07 same-site C; "
            "c-mod-5mer compares mean-pooled 5-mer embeddings centered on modified C sites; "
            "lb06-c-labels compares LB06 normal C(label=1) with modified C(label=2); "
            "base compares ordinary A/C/G/T token embeddings between LB06 and LB07; "
            "single-dataset-base plots A/C/G/T distributions inside LB07 and/or LB06 separately."
        ),
    )
    parser.add_argument("--base-types", default="A,T,G", help="Comma-separated bases used by --compare-mode base, for example A,T or A,C,G,T.")
    parser.add_argument("--single-dataset", choices=("LB07", "LB06", "both"), default="both")
    parser.add_argument("--samples-per-token", type=int, default=5)
    parser.add_argument(
        "--base-token-mode",
        choices=("center-unique", "all-overlap"),
        default="center-unique",
        help="Only affects compare-mode base/single-dataset-base. center-unique takes one center token per base and drops tokens shared by multiple bases.",
    )
    parser.add_argument(
        "--c-mod-site-batch-mode",
        choices=("separate", "mixed"),
        default="separate",
        help="Only affects compare-mode c-mod-sites. mixed interleaves LB06/LB07 reads in the same forward batches before selecting site embeddings.",
    )
    parser.add_argument(
        "--annotate-site-labels",
        choices=("none", "modified-centroids", "modified-points"),
        default="none",
        help="Annotate modified-site labels on PCA panels. modified-centroids labels one centroid per site; modified-points labels every modified point.",
    )
    parser.add_argument(
        "--modified-color-by",
        choices=("class", "read"),
        default="class",
        help="For c-mod-5mer PCA panels, color LB06 modified points either as one class or by read_id.",
    )
    parser.add_argument("--max-read-legend-items", type=int, default=30, help="When coloring modified points by read_id, show at most this many read labels in the legend.")
    parser.add_argument("--color-by", choices=("dataset", "dataset_base"), default="dataset")
    parser.add_argument("--embedding-source", choices=("bert", "dlm", "context_hidden", "ode_hidden"), default="bert")
    parser.add_argument("--sequence-key", choices=("auto", "label", "ref", "seq"), default="label")
    parser.add_argument(
        "--plot-mode",
        choices=("all", "per-sequence", "per-site", "per-read", "lb06-per-sequence-sites", "both", "all-modes"),
        default="both",
    )
    parser.add_argument("--max-lb06-read-plots", type=int, default=0, help="For c-mod-sites per-read plots, cap the number of LB06 reads to plot; 0 means all reads.")
    parser.add_argument("--limit-lb07-reads", type=int, default=None)
    parser.add_argument("--limit-lb06-reads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=2000)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.95)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-lb07-points", type=int, default=100000, help="0 means keep all LB07 same-site points.")
    parser.add_argument("--max-lb06-points", type=int, default=0, help="0 means keep all LB06 modified points.")
    parser.add_argument("--min-kmer-tokens", type=int, default=3, help="For compare-mode c-mod-5mer, skip sites with fewer unique covered tokens than this after de-duplication.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_source = normalize_embedding_source(args.embedding_source)
    embedding_label = "bert" if embedding_source == "context_hidden" else "dlm"
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    lb06_records = load_limited_records(args.lb06_jsonl, limit_reads=args.limit_lb06_reads)
    lb07_records = load_limited_records(args.lb07_jsonl, limit_reads=args.limit_lb07_reads)
    if not lb06_records:
        raise ValueError(f"No LB06 records loaded from {args.lb06_jsonl}")
    if not lb07_records:
        raise ValueError(f"No LB07 records loaded from {args.lb07_jsonl}")

    model = build_model(args, device, embedding_source)

    if args.compare_mode == "c-mod-5mer":
        modified_base_positions_by_sequence = collect_modified_base_positions_from_records(
            lb06_records,
            sequence_key_mode=args.sequence_key,
        )
        if not modified_base_positions_by_sequence:
            raise ValueError("No LB06 modified C base positions were found from meta.modified_c_token_spans.")

        lb06_rows = collect_kmer_embeddings(
            args,
            model=model,
            device=device,
            records=lb06_records,
            dataset_name="LB06_modified_5mer",
            embedding_source=embedding_source,
            modified_base_positions_by_sequence=modified_base_positions_by_sequence,
        )
        lb07_rows = collect_kmer_embeddings(
            args,
            model=model,
            device=device,
            records=lb07_records,
            dataset_name="LB07_same_site_5mer",
            embedding_source=embedding_source,
            modified_base_positions_by_sequence=modified_base_positions_by_sequence,
        )
        if not lb06_rows:
            raise ValueError("No LB06 5-mer embeddings were collected.")
        if not lb07_rows:
            raise ValueError("No LB07 same-site 5-mer embeddings were collected.")

        summaries: list[dict[str, Any]] = []
        all_rows = sample_rows(lb07_rows, args.max_lb07_points, rng) + sample_rows(lb06_rows, args.max_lb06_points, rng)
        class0_label = "LB07 same-site 5-mer"
        class1_label = "LB06 modified-centered 5-mer"

        if args.plot_mode in {"all", "both", "all-modes"}:
            output_png = output_dir / f"LB06_vs_LB07_modified_centered_5mer_{embedding_label}_all.png"
            summary = plot_rows(
                all_rows,
                output_png,
                title=f"LB06 modified-centered 5-mer vs LB07 same-site 5-mer | {embedding_label}",
                embedding_label=embedding_label,
                class0_label=class0_label,
                class1_label=class1_label,
                pca_title="modified-centered 5-mer PCA",
                annotate_site_labels=args.annotate_site_labels,
                modified_color_by=args.modified_color_by,
                max_read_legend_items=args.max_read_legend_items,
            )
            summary.update(
                {
                    "compare_mode": "c-mod-5mer",
                    "plot_mode": "all",
                    "embedding_source": embedding_source,
                    "min_kmer_tokens": args.min_kmer_tokens,
                    "annotate_site_labels": args.annotate_site_labels,
                    "modified_color_by": args.modified_color_by,
                    "max_read_legend_items": args.max_read_legend_items,
                }
            )
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(all_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

        if args.plot_mode in {"per-sequence", "both", "all-modes"}:
            lb06_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
            lb07_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
            for row in lb06_rows:
                lb06_by_seq[row.sequence_key].append(row)
            for row in lb07_rows:
                lb07_by_seq[row.sequence_key].append(row)

            for sequence_key in sorted(modified_base_positions_by_sequence):
                seq_rows = sample_rows(lb07_by_seq.get(sequence_key, []), args.max_lb07_points, rng)
                seq_rows += sample_rows(lb06_by_seq.get(sequence_key, []), args.max_lb06_points, rng)
                if not seq_rows:
                    continue
                output_png = output_dir / "per_sequence_5mer" / f"{safe_name(sequence_key)}_{embedding_label}_modified_centered_5mer.png"
                summary = plot_rows(
                    seq_rows,
                    output_png,
                    title=f"{sequence_key} | LB06 modified-centered 5-mer vs LB07 same-site 5-mer | {embedding_label}",
                    embedding_label=embedding_label,
                    class0_label=class0_label,
                    class1_label=class1_label,
                    pca_title="modified-centered 5-mer PCA",
                    annotate_site_labels=args.annotate_site_labels,
                    modified_color_by=args.modified_color_by,
                    max_read_legend_items=args.max_read_legend_items,
                )
                summary.update(
                    {
                        "compare_mode": "c-mod-5mer",
                        "plot_mode": "per-sequence",
                        "sequence_key": sequence_key,
                        "embedding_source": embedding_source,
                        "modified_base_positions_1based": sorted(modified_base_positions_by_sequence[sequence_key]),
                        "min_kmer_tokens": args.min_kmer_tokens,
                        "annotate_site_labels": args.annotate_site_labels,
                        "modified_color_by": args.modified_color_by,
                        "max_read_legend_items": args.max_read_legend_items,
                    }
                )
                points_path = output_png.with_suffix(".points.jsonl")
                save_points(seq_rows, points_path)
                summary["points_jsonl"] = str(points_path)
                summaries.append(summary)

        if args.plot_mode in {"per-site", "all-modes"}:
            lb06_by_site: dict[str, list[EmbeddingRow]] = defaultdict(list)
            lb07_by_site: dict[str, list[EmbeddingRow]] = defaultdict(list)
            for row in lb06_rows:
                if row.site_key is not None:
                    lb06_by_site[row.site_key].append(row)
            for row in lb07_rows:
                if row.site_key is not None:
                    lb07_by_site[row.site_key].append(row)

            for sequence_key in sorted(modified_base_positions_by_sequence):
                for base_position in sorted(modified_base_positions_by_sequence[sequence_key]):
                    site_key = f"{sequence_key}:base_{int(base_position)}:5mer"
                    site_rows = sample_rows(lb07_by_site.get(site_key, []), args.max_lb07_points, rng)
                    site_rows += sample_rows(lb06_by_site.get(site_key, []), args.max_lb06_points, rng)
                    if not site_rows:
                        continue
                    output_png = (
                        output_dir
                        / "per_site_5mer"
                        / safe_name(sequence_key)
                        / f"{safe_name(sequence_key)}_base_{int(base_position)}_{embedding_label}_5mer.png"
                    )
                    summary = plot_rows(
                        site_rows,
                        output_png,
                        title=f"{sequence_key} | base_{int(base_position)} centered 5-mer | {embedding_label}",
                        embedding_label=embedding_label,
                        class0_label=class0_label,
                        class1_label=class1_label,
                        pca_title="modified-centered 5-mer PCA",
                        annotate_site_labels=args.annotate_site_labels,
                        modified_color_by=args.modified_color_by,
                        max_read_legend_items=args.max_read_legend_items,
                    )
                    summary.update(
                        {
                            "compare_mode": "c-mod-5mer",
                            "plot_mode": "per-site",
                            "sequence_key": sequence_key,
                            "site_key": site_key,
                            "base_position_1based": int(base_position),
                            "embedding_source": embedding_source,
                            "min_kmer_tokens": args.min_kmer_tokens,
                            "annotate_site_labels": args.annotate_site_labels,
                            "modified_color_by": args.modified_color_by,
                            "max_read_legend_items": args.max_read_legend_items,
                        }
                    )
                    points_path = output_png.with_suffix(".points.jsonl")
                    save_points(site_rows, points_path)
                    summary["points_jsonl"] = str(points_path)
                    summaries.append(summary)

        if not summaries:
            raise ValueError("No c-mod-5mer plots were created.")

        summary = {
            "lb07_jsonl": args.lb07_jsonl,
            "lb06_jsonl": args.lb06_jsonl,
            "compare_mode": "c-mod-5mer",
            "kmer_size": 5,
            "kmer_pooling": "mean",
            "kmer_token_selection": "all tokens covered by the 5-mer base spans, de-duplicated",
            "min_kmer_tokens": args.min_kmer_tokens,
            "samples_per_token": args.samples_per_token,
            "embedding_source_requested": args.embedding_source,
            "embedding_source_normalized": embedding_source,
            "embedding_label": embedding_label,
            "annotate_site_labels": args.annotate_site_labels,
            "modified_color_by": args.modified_color_by,
            "max_read_legend_items": args.max_read_legend_items,
            "sequence_key": args.sequence_key,
            "lb06_records": len(lb06_records),
            "lb07_records": len(lb07_records),
            "lb06_5mer_points_total": len(lb06_rows),
            "lb07_5mer_points_total": len(lb07_rows),
            "lb06_5mer_points_by_sequence": dict(Counter(row.sequence_key for row in lb06_rows)),
            "lb07_5mer_points_by_sequence": dict(Counter(row.sequence_key for row in lb07_rows)),
            "modified_base_positions_by_sequence": {
                sequence_key: sorted(positions)
                for sequence_key, positions in sorted(modified_base_positions_by_sequence.items())
            },
            "plots": summaries,
        }
        summary_path = output_dir / f"LB06_vs_LB07_modified_centered_5mer_{embedding_label}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        print("Compare mode: c-mod-5mer")
        print(f"LB06 5-mer points: {len(lb06_rows)}")
        print(f"LB07 5-mer points: {len(lb07_rows)}")
        print(f"Output dir: {output_dir}")
        print(f"Summary: {summary_path}")
        return

    if args.compare_mode == "lb06-c-labels":
        modified_positions_by_sequence = collect_label_positions(
            lb06_records,
            sequence_key_mode=args.sequence_key,
            label_values={2},
        )
        positions_by_site, site_by_sequence_token, base_position_by_site = build_modified_site_maps(
            lb06_records,
            sequence_key_mode=args.sequence_key,
            fallback_positions_by_sequence=modified_positions_by_sequence,
        )
        lb06_rows = collect_dataset_embeddings(
            args,
            model=model,
            device=device,
            records=lb06_records,
            dataset_name="LB06_c_labels",
            embedding_source=embedding_source,
            wanted_positions_by_sequence=None,
            site_by_sequence_token=site_by_sequence_token,
            base_position_by_site=base_position_by_site,
            wanted_label_values={1, 2},
        )
        if not lb06_rows:
            raise ValueError("No LB06 C embeddings were collected for labels 1/2.")

        normal_rows = [row for row in lb06_rows if int(row.label) == 1]
        modified_rows = [row for row in lb06_rows if int(row.label) == 2]
        if not normal_rows:
            raise ValueError("No LB06 normal C embeddings with c_modification_label=1 were collected.")
        if not modified_rows:
            raise ValueError("No LB06 modified C embeddings with c_modification_label=2 were collected.")

        summaries: list[dict[str, Any]] = []
        sampled_rows = sample_rows(normal_rows, args.max_lb07_points, rng) + sample_rows(modified_rows, args.max_lb06_points, rng)
        class0_label = "LB06 normal C"
        class1_label = "LB06 modified C"

        if args.plot_mode in {"all", "both", "all-modes"}:
            output_png = output_dir / f"LB06_normal_vs_modified_C_{embedding_label}_all.png"
            summary = plot_c_label_rows(
                sampled_rows,
                output_png,
                title=f"LB06 normal C vs modified C | {embedding_label}",
                embedding_label=embedding_label,
                class0_label=class0_label,
                class1_label=class1_label,
                pca_title="LB06 normal-vs-modified C PCA",
            )
            summary.update(
                {
                    "compare_mode": "lb06-c-labels",
                    "plot_mode": "all",
                    "embedding_source": embedding_source,
                    "normal_sample_cap": args.max_lb07_points,
                    "modified_sample_cap": args.max_lb06_points,
                }
            )
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(sampled_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

        if args.plot_mode in {"per-sequence", "both", "all-modes"}:
            normal_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
            modified_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
            for row in normal_rows:
                normal_by_seq[row.sequence_key].append(row)
            for row in modified_rows:
                modified_by_seq[row.sequence_key].append(row)

            for sequence_key in sorted(set(normal_by_seq) | set(modified_by_seq)):
                seq_rows = sample_rows(normal_by_seq.get(sequence_key, []), args.max_lb07_points, rng)
                seq_rows += sample_rows(modified_by_seq.get(sequence_key, []), args.max_lb06_points, rng)
                if not seq_rows:
                    continue
                output_png = output_dir / "lb06_c_labels_per_sequence" / f"{safe_name(sequence_key)}_{embedding_label}_normal_vs_modified_C.png"
                summary = plot_c_label_rows(
                    seq_rows,
                    output_png,
                    title=f"{sequence_key} | LB06 normal C vs modified C | {embedding_label}",
                    embedding_label=embedding_label,
                    class0_label=class0_label,
                    class1_label=class1_label,
                    pca_title="LB06 normal-vs-modified C PCA",
                )
                summary.update(
                    {
                        "compare_mode": "lb06-c-labels",
                        "plot_mode": "per-sequence",
                        "sequence_key": sequence_key,
                        "embedding_source": embedding_source,
                        "normal_sample_cap": args.max_lb07_points,
                        "modified_sample_cap": args.max_lb06_points,
                        "modified_token_positions": sorted(modified_positions_by_sequence.get(sequence_key, set())),
                    }
                )
                points_path = output_png.with_suffix(".points.jsonl")
                save_points(seq_rows, points_path)
                summary["points_jsonl"] = str(points_path)
                summaries.append(summary)

        if not summaries:
            raise ValueError("No lb06-c-labels plots were created. Use --plot-mode all, both, per-sequence, or all-modes.")

        summary = {
            "lb06_jsonl": args.lb06_jsonl,
            "compare_mode": "lb06-c-labels",
            "label_semantics": {
                "1": "LB06 normal/unmodified C",
                "2": "LB06 modified C",
            },
            "embedding_source_requested": args.embedding_source,
            "embedding_source_normalized": embedding_source,
            "embedding_label": embedding_label,
            "sequence_key": args.sequence_key,
            "lb06_records": len(lb06_records),
            "lb06_c_points_total": len(lb06_rows),
            "lb06_normal_c_points_total": len(normal_rows),
            "lb06_modified_c_points_total": len(modified_rows),
            "lb06_normal_c_points_by_sequence": dict(Counter(row.sequence_key for row in normal_rows)),
            "lb06_modified_c_points_by_sequence": dict(Counter(row.sequence_key for row in modified_rows)),
            "modified_positions_by_sequence": summarize_positions(modified_positions_by_sequence),
            "plots": summaries,
        }
        summary_path = output_dir / f"LB06_normal_vs_modified_C_{embedding_label}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        print("Compare mode: lb06-c-labels")
        print(f"LB06 normal C points: {len(normal_rows)}")
        print(f"LB06 modified C points: {len(modified_rows)}")
        print(f"Output dir: {output_dir}")
        print(f"Summary: {summary_path}")
        return

    if args.compare_mode == "single-dataset-base":
        base_types = parse_base_types(args.base_types)
        dataset_inputs: list[tuple[str, list[ReadRecord], int]] = []
        if args.single_dataset in {"LB07", "both"}:
            dataset_inputs.append(("LB07", lb07_records, args.max_lb07_points))
        if args.single_dataset in {"LB06", "both"}:
            dataset_inputs.append(("LB06", lb06_records, args.max_lb06_points))

        summaries: list[dict[str, Any]] = []
        base_label = "".join(sorted(base_types))
        total_points_by_dataset: dict[str, int] = {}
        points_by_dataset_base: dict[str, dict[str, int]] = {}
        points_by_dataset_sequence: dict[str, dict[str, int]] = {}

        for dataset_name, records, max_points in dataset_inputs:
            rows = collect_base_type_embeddings(
                args,
                model=model,
                device=device,
                records=records,
                dataset_name=f"{dataset_name}_base",
                embedding_source=embedding_source,
                base_types=base_types,
            )
            if not rows:
                print(f"Skip {dataset_name}: no base embeddings found for base_types={sorted(base_types)}.")
                continue

            total_points_by_dataset[dataset_name] = len(rows)
            points_by_dataset_base[dataset_name] = dict(Counter(row.base for row in rows))
            points_by_dataset_sequence[dataset_name] = dict(Counter(row.sequence_key for row in rows))
            sampled_rows = sample_rows(rows, max_points, rng)

            if args.plot_mode in {"all", "both", "all-modes"}:
                output_png = output_dir / "single_dataset_base" / dataset_name / f"{dataset_name}_base_{base_label}_{embedding_label}_all.png"
                summary = plot_rows_by_base(
                    sampled_rows,
                    output_png,
                    title=f"{dataset_name} {base_label} token embeddings | {embedding_label}",
                    embedding_label=embedding_label,
                    pca_title=f"{dataset_name} {base_label} token PCA",
                )
                summary.update(
                    {
                        "compare_mode": "single-dataset-base",
                        "plot_mode": "all",
                        "dataset": dataset_name,
                        "embedding_source": embedding_source,
                        "base_types": sorted(base_types),
                        "base_token_mode": args.base_token_mode,
                    }
                )
                points_path = output_png.with_suffix(".points.jsonl")
                save_points(sampled_rows, points_path)
                summary["points_jsonl"] = str(points_path)
                summaries.append(summary)

            if args.plot_mode in {"per-sequence", "both", "all-modes"}:
                rows_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
                for row in rows:
                    rows_by_seq[row.sequence_key].append(row)
                for sequence_key in sorted(rows_by_seq):
                    seq_rows = sample_rows(rows_by_seq[sequence_key], max_points, rng)
                    if not seq_rows:
                        continue
                    output_png = (
                        output_dir
                        / "single_dataset_base"
                        / dataset_name
                        / "per_sequence"
                        / f"{safe_name(sequence_key)}_{base_label}_{embedding_label}.png"
                    )
                    summary = plot_rows_by_base(
                        seq_rows,
                        output_png,
                        title=f"{dataset_name} {sequence_key} {base_label} token embeddings | {embedding_label}",
                        embedding_label=embedding_label,
                        pca_title=f"{base_label} token PCA",
                    )
                    summary.update(
                        {
                            "compare_mode": "single-dataset-base",
                            "plot_mode": "per-sequence",
                            "dataset": dataset_name,
                            "sequence_key": sequence_key,
                            "embedding_source": embedding_source,
                            "base_types": sorted(base_types),
                            "base_token_mode": args.base_token_mode,
                        }
                    )
                    points_path = output_png.with_suffix(".points.jsonl")
                    save_points(seq_rows, points_path)
                    summary["points_jsonl"] = str(points_path)
                    summaries.append(summary)

            if args.plot_mode in {"per-site", "all-modes"}:
                rows_by_base: dict[str, list[EmbeddingRow]] = defaultdict(list)
                for row in rows:
                    if row.base is not None:
                        rows_by_base[row.base].append(row)
                for base in sorted(base_types):
                    base_rows = sample_rows(rows_by_base.get(base, []), max_points, rng)
                    if not base_rows:
                        continue
                    output_png = (
                        output_dir
                        / "single_dataset_base"
                        / dataset_name
                        / "per_base"
                        / f"{dataset_name}_base_{base}_{embedding_label}.png"
                    )
                    summary = plot_rows_by_base(
                        base_rows,
                        output_png,
                        title=f"{dataset_name} base {base} token embeddings | {embedding_label}",
                        embedding_label=embedding_label,
                        pca_title=f"base {base} token PCA",
                    )
                    summary.update(
                        {
                            "compare_mode": "single-dataset-base",
                            "plot_mode": "per-base",
                            "dataset": dataset_name,
                            "base": base,
                            "embedding_source": embedding_source,
                            "base_types": sorted(base_types),
                            "base_token_mode": args.base_token_mode,
                        }
                    )
                    points_path = output_png.with_suffix(".points.jsonl")
                    save_points(base_rows, points_path)
                    summary["points_jsonl"] = str(points_path)
                    summaries.append(summary)

        if not summaries:
            raise ValueError("No single-dataset base plots were created.")

        summary = {
            "lb07_jsonl": args.lb07_jsonl,
            "lb06_jsonl": args.lb06_jsonl,
            "compare_mode": "single-dataset-base",
            "single_dataset": args.single_dataset,
            "base_types": sorted(base_types),
            "samples_per_token": args.samples_per_token,
            "base_token_mode": args.base_token_mode,
            "embedding_source_requested": args.embedding_source,
            "embedding_source_normalized": embedding_source,
            "embedding_label": embedding_label,
            "sequence_key": args.sequence_key,
            "lb06_records": len(lb06_records),
            "lb07_records": len(lb07_records),
            "points_by_dataset": total_points_by_dataset,
            "points_by_dataset_base": points_by_dataset_base,
            "points_by_dataset_sequence": points_by_dataset_sequence,
            "plots": summaries,
        }
        dataset_label = args.single_dataset if args.single_dataset != "both" else "LB06_LB07"
        summary_path = output_dir / f"{dataset_label}_single_dataset_base_{base_label}_{embedding_label}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        print("Compare mode: single-dataset-base")
        print(f"Dataset: {args.single_dataset}")
        print(f"Base types: {','.join(sorted(base_types))}")
        print(f"Output dir: {output_dir}")
        print(f"Summary: {summary_path}")
        return

    if args.compare_mode == "base":
        base_types = parse_base_types(args.base_types)
        lb06_rows = collect_base_type_embeddings(
            args,
            model=model,
            device=device,
            records=lb06_records,
            dataset_name="LB06_base",
            embedding_source=embedding_source,
            base_types=base_types,
        )
        lb07_rows = collect_base_type_embeddings(
            args,
            model=model,
            device=device,
            records=lb07_records,
            dataset_name="LB07_base",
            embedding_source=embedding_source,
            base_types=base_types,
        )
        if not lb06_rows:
            raise ValueError(f"No LB06 base embeddings found for base_types={sorted(base_types)}.")
        if not lb07_rows:
            raise ValueError(f"No LB07 base embeddings found for base_types={sorted(base_types)}.")

        base_label = "".join(sorted(base_types))
        class0_label = f"LB07 {base_label} tokens"
        class1_label = f"LB06 {base_label} tokens"
        summaries: list[dict[str, Any]] = []
        all_rows = sample_rows(lb07_rows, args.max_lb07_points, rng) + sample_rows(lb06_rows, args.max_lb06_points, rng)

        if args.plot_mode in {"all", "both", "all-modes"}:
            output_png = output_dir / f"LB06_vs_LB07_base_{base_label}_{embedding_label}_all.png"
            summary = plot_base_rows(
                args,
                all_rows,
                output_png,
                title=f"LB06 vs LB07 {base_label} token embeddings | {embedding_label}",
                embedding_label=embedding_label,
                class0_label=class0_label,
                class1_label=class1_label,
                pca_title=f"{base_label} token PCA",
            )
            summary.update({
                "compare_mode": "base",
                "plot_mode": "all",
                "embedding_source": embedding_source,
                "base_types": sorted(base_types),
                "base_token_mode": args.base_token_mode,
                "color_by": args.color_by,
            })
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(all_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

        if args.plot_mode in {"per-sequence", "both", "all-modes"}:
            lb06_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
            lb07_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
            for row in lb06_rows:
                lb06_by_seq[row.sequence_key].append(row)
            for row in lb07_rows:
                lb07_by_seq[row.sequence_key].append(row)
            for sequence_key in sorted(set(lb06_by_seq) | set(lb07_by_seq)):
                seq_rows = sample_rows(lb07_by_seq.get(sequence_key, []), args.max_lb07_points, rng)
                seq_rows += sample_rows(lb06_by_seq.get(sequence_key, []), args.max_lb06_points, rng)
                if not seq_rows:
                    continue
                output_png = output_dir / "per_sequence_base" / f"{safe_name(sequence_key)}_{base_label}_{embedding_label}.png"
                summary = plot_base_rows(
                    args,
                    seq_rows,
                    output_png,
                    title=f"{sequence_key} | LB06 vs LB07 {base_label} token embeddings | {embedding_label}",
                    embedding_label=embedding_label,
                    class0_label=class0_label,
                    class1_label=class1_label,
                    pca_title=f"{base_label} token PCA",
                )
                summary.update(
                    {
                        "compare_mode": "base",
                        "plot_mode": "per-sequence",
                        "sequence_key": sequence_key,
                        "embedding_source": embedding_source,
                        "base_types": sorted(base_types),
                        "base_token_mode": args.base_token_mode,
                        "color_by": args.color_by,
                    }
                )
                points_path = output_png.with_suffix(".points.jsonl")
                save_points(seq_rows, points_path)
                summary["points_jsonl"] = str(points_path)
                summaries.append(summary)

        if args.plot_mode in {"per-site", "all-modes"}:
            lb06_by_base: dict[str, list[EmbeddingRow]] = defaultdict(list)
            lb07_by_base: dict[str, list[EmbeddingRow]] = defaultdict(list)
            for row in lb06_rows:
                if row.base is not None:
                    lb06_by_base[row.base].append(row)
            for row in lb07_rows:
                if row.base is not None:
                    lb07_by_base[row.base].append(row)
            for base in sorted(base_types):
                base_rows = sample_rows(lb07_by_base.get(base, []), args.max_lb07_points, rng)
                base_rows += sample_rows(lb06_by_base.get(base, []), args.max_lb06_points, rng)
                if not base_rows:
                    continue
                output_png = output_dir / "per_base" / f"LB06_vs_LB07_base_{base}_{embedding_label}.png"
                summary = plot_base_rows(
                    args,
                    base_rows,
                    output_png,
                    title=f"LB06 vs LB07 base {base} token embeddings | {embedding_label}",
                    embedding_label=embedding_label,
                    class0_label=f"LB07 {base} tokens",
                    class1_label=f"LB06 {base} tokens",
                    pca_title=f"base {base} token PCA",
                )
                summary.update(
                    {
                        "compare_mode": "base",
                        "plot_mode": "per-base",
                        "base": base,
                        "embedding_source": embedding_source,
                        "base_types": sorted(base_types),
                        "base_token_mode": args.base_token_mode,
                        "color_by": args.color_by,
                    }
                )
                points_path = output_png.with_suffix(".points.jsonl")
                save_points(base_rows, points_path)
                summary["points_jsonl"] = str(points_path)
                summaries.append(summary)

        summary = {
            "lb07_jsonl": args.lb07_jsonl,
            "lb06_jsonl": args.lb06_jsonl,
            "compare_mode": "base",
            "base_types": sorted(base_types),
            "samples_per_token": args.samples_per_token,
            "base_token_mode": args.base_token_mode,
            "color_by": args.color_by,
            "embedding_source_requested": args.embedding_source,
            "embedding_source_normalized": embedding_source,
            "embedding_label": embedding_label,
            "sequence_key": args.sequence_key,
            "lb06_records": len(lb06_records),
            "lb07_records": len(lb07_records),
            "lb06_base_points_total": len(lb06_rows),
            "lb07_base_points_total": len(lb07_rows),
            "lb06_base_points_by_base": dict(Counter(row.base for row in lb06_rows)),
            "lb07_base_points_by_base": dict(Counter(row.base for row in lb07_rows)),
            "lb06_base_points_by_sequence": dict(Counter(row.sequence_key for row in lb06_rows)),
            "lb07_base_points_by_sequence": dict(Counter(row.sequence_key for row in lb07_rows)),
            "plots": summaries,
        }
        summary_path = output_dir / f"LB06_vs_LB07_base_{base_label}_{embedding_label}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        print(f"Compare mode: base")
        print(f"Base types: {','.join(sorted(base_types))}")
        print(f"LB06 records: {len(lb06_records)}")
        print(f"LB07 records: {len(lb07_records)}")
        print(f"LB06 base points: {len(lb06_rows)}")
        print(f"LB07 base points: {len(lb07_rows)}")
        print(f"Output dir: {output_dir}")
        print(f"Summary: {summary_path}")
        return

    if args.c_mod_site_batch_mode == "mixed":
        positions_by_sequence = collect_label_positions(
            lb06_records,
            sequence_key_mode=args.sequence_key,
            label_values={2},
        )
        lb06_rows: list[EmbeddingRow] = []
    else:
        lb06_rows = collect_dataset_embeddings(
            args,
            model=model,
            device=device,
            records=lb06_records,
            dataset_name="LB06_modified",
            embedding_source=embedding_source,
            wanted_positions_by_sequence=None,
            site_by_sequence_token=None,
            base_position_by_site=None,
            wanted_label_values={2},
        )
        positions_by_sequence = modified_positions_from_rows(lb06_rows)
    if not positions_by_sequence:
        raise ValueError("No LB06 label=2 modified C token positions were found.")
    positions_by_site, site_by_sequence_token, base_position_by_site = build_modified_site_maps(
        lb06_records,
        sequence_key_mode=args.sequence_key,
        fallback_positions_by_sequence=positions_by_sequence,
    )
    if args.c_mod_site_batch_mode == "mixed":
        mixed_rows = collect_mixed_dataset_embeddings(
            args,
            model=model,
            device=device,
            datasets=[
                ("LB06_modified", lb06_records, None, {2}),
                ("LB07_unmodified_same_site", lb07_records, positions_by_sequence, {1}),
            ],
            embedding_source=embedding_source,
            site_by_sequence_token=site_by_sequence_token,
            base_position_by_site=base_position_by_site,
        )
        lb06_rows = mixed_rows.get("LB06_modified", [])
        lb07_rows = mixed_rows.get("LB07_unmodified_same_site", [])
    else:
        for row in lb06_rows:
            row.site_key = site_by_sequence_token.get(row.sequence_key, {}).get(row.token_position)
            if row.site_key is not None:
                row.base_position_1based = base_position_by_site.get(row.site_key)

        lb07_rows = collect_dataset_embeddings(
            args,
            model=model,
            device=device,
            records=lb07_records,
            dataset_name="LB07_unmodified_same_site",
            embedding_source=embedding_source,
            wanted_positions_by_sequence=positions_by_sequence,
            site_by_sequence_token=site_by_sequence_token,
            base_position_by_site=base_position_by_site,
            wanted_label_values={1},
        )
    if not lb06_rows:
        raise ValueError("No LB06 label=2 modified C embeddings were collected.")
    if not lb07_rows:
        raise ValueError("No LB07 label=1 C token embeddings were found at LB06 modified token positions.")

    summaries: list[dict[str, Any]] = []
    all_rows = sample_rows(lb07_rows, args.max_lb07_points, rng) + sample_rows(lb06_rows, args.max_lb06_points, rng)
    if args.plot_mode in {"all", "both", "all-modes"}:
        output_png = output_dir / f"LB06_vs_LB07_same_site_C_{embedding_label}_all.png"
        summary = plot_rows(
            all_rows,
            output_png,
            title=(
                f"LB06 modified C vs LB07 same-site unmodified C | {embedding_label} | "
                f"sequences={len(positions_by_sequence)}"
            ),
            embedding_label=embedding_label,
            annotate_site_labels=args.annotate_site_labels,
        )
        summary.update(
            {
                "plot_mode": "all",
                "embedding_source": embedding_source,
                "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
                "annotate_site_labels": args.annotate_site_labels,
            }
        )
        points_path = output_png.with_suffix(".points.jsonl")
        save_points(all_rows, points_path)
        summary["points_jsonl"] = str(points_path)
        summaries.append(summary)

    if args.plot_mode in {"per-sequence", "both", "all-modes"}:
        lb06_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
        lb07_by_seq: dict[str, list[EmbeddingRow]] = defaultdict(list)
        for row in lb06_rows:
            lb06_by_seq[row.sequence_key].append(row)
        for row in lb07_rows:
            lb07_by_seq[row.sequence_key].append(row)

        for sequence_key in sorted(positions_by_sequence):
            seq_rows = sample_rows(lb07_by_seq.get(sequence_key, []), args.max_lb07_points, rng)
            seq_rows += sample_rows(lb06_by_seq.get(sequence_key, []), args.max_lb06_points, rng)
            if not seq_rows:
                continue
            output_png = output_dir / "per_sequence" / f"{safe_name(sequence_key)}_{embedding_label}_same_site_C.png"
            summary = plot_rows(
                seq_rows,
                output_png,
                title=f"{sequence_key} | LB06 modified C vs LB07 same-site unmodified C | {embedding_label}",
                embedding_label=embedding_label,
                annotate_site_labels=args.annotate_site_labels,
            )
            summary.update(
                {
                    "plot_mode": "per-sequence",
                    "sequence_key": sequence_key,
                    "embedding_source": embedding_source,
                    "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
                    "annotate_site_labels": args.annotate_site_labels,
                    "modified_token_positions": sorted(positions_by_sequence[sequence_key]),
                }
            )
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(seq_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

    if args.plot_mode in {"per-site", "all-modes"}:
        lb06_by_site: dict[str, list[EmbeddingRow]] = defaultdict(list)
        lb07_by_site: dict[str, list[EmbeddingRow]] = defaultdict(list)
        for row in lb06_rows:
            if row.site_key is not None:
                lb06_by_site[row.site_key].append(row)
        for row in lb07_rows:
            if row.site_key is not None:
                lb07_by_site[row.site_key].append(row)

        for sequence_key in sorted(positions_by_site):
            for site_key in sorted(positions_by_site[sequence_key]):
                site_rows = sample_rows(lb07_by_site.get(site_key, []), args.max_lb07_points, rng)
                site_rows += sample_rows(lb06_by_site.get(site_key, []), args.max_lb06_points, rng)
                if not site_rows:
                    continue
                base_position = base_position_by_site.get(site_key)
                site_label = f"base_{base_position}" if base_position is not None else safe_name(site_key.split(":", 1)[-1])
                output_png = (
                    output_dir
                    / "per_site"
                    / safe_name(sequence_key)
                    / f"{safe_name(sequence_key)}_{site_label}_{embedding_label}_same_site_C.png"
                )
                title_site = f"{sequence_key} | {site_label} | LB06 modified C vs LB07 same-site unmodified C | {embedding_label}"
                summary = plot_rows(
                    site_rows,
                    output_png,
                    title=title_site,
                    embedding_label=embedding_label,
                    annotate_site_labels=args.annotate_site_labels,
                )
                summary.update(
                    {
                        "plot_mode": "per-site",
                        "sequence_key": sequence_key,
                        "site_key": site_key,
                        "base_position_1based": base_position,
                        "embedding_source": embedding_source,
                        "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
                        "annotate_site_labels": args.annotate_site_labels,
                        "modified_token_positions": sorted(positions_by_site[sequence_key][site_key]),
                    }
                )
                points_path = output_png.with_suffix(".points.jsonl")
                save_points(site_rows, points_path)
                summary["points_jsonl"] = str(points_path)
                summaries.append(summary)

    if args.plot_mode in {"per-read", "all-modes"}:
        lb06_by_read: dict[tuple[str, str], list[EmbeddingRow]] = defaultdict(list)
        for row in lb06_rows:
            lb06_by_read[(row.sequence_key, row.read_id)].append(row)

        read_items = sorted(lb06_by_read.items(), key=lambda item: (item[0][0], item[0][1]))
        if args.max_lb06_read_plots > 0:
            read_items = read_items[: args.max_lb06_read_plots]

        for (sequence_key, read_id), read_rows in read_items:
            if not read_rows:
                continue
            output_png = (
                output_dir
                / "lb06_per_read_modified_sites"
                / safe_name(sequence_key)
                / f"{safe_name(sequence_key)}_{safe_name(read_id)}_{embedding_label}_modified_sites.png"
            )
            summary = plot_lb06_read_modified_site_rows(
                read_rows,
                output_png,
                title=f"{sequence_key} | {read_id} | LB06 modified-site token embeddings | {embedding_label}",
                embedding_label=embedding_label,
            )
            summary.update(
                {
                    "plot_mode": "per-read",
                    "sequence_key": sequence_key,
                    "read_id": read_id,
                    "embedding_source": embedding_source,
                    "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
                    "max_lb06_read_plots": args.max_lb06_read_plots,
                    "modified_token_positions": sorted(row.token_position for row in read_rows),
                    "modified_sites": sorted({site_label_for_row(row) for row in read_rows}),
                }
            )
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(read_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

    if args.plot_mode in {"lb06-per-sequence-sites", "all-modes"}:
        lb06_by_seq_for_sites: dict[str, list[EmbeddingRow]] = defaultdict(list)
        for row in lb06_rows:
            lb06_by_seq_for_sites[row.sequence_key].append(row)

        for sequence_key in sorted(lb06_by_seq_for_sites):
            seq_rows = sample_rows(lb06_by_seq_for_sites[sequence_key], args.max_lb06_points, rng)
            if not seq_rows:
                continue
            output_png = (
                output_dir
                / "lb06_per_sequence_modified_sites"
                / f"{safe_name(sequence_key)}_{embedding_label}_modified_sites.png"
            )
            summary = plot_lb06_read_modified_site_rows(
                seq_rows,
                output_png,
                title=f"{sequence_key} | LB06 modified-site token embeddings by site | {embedding_label}",
                embedding_label=embedding_label,
                pca_title="LB06 sequence modified-site PCA",
                summary_read_id=False,
            )
            summary.update(
                {
                    "plot_mode": "lb06-per-sequence-sites",
                    "sequence_key": sequence_key,
                    "embedding_source": embedding_source,
                    "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
                    "modified_token_positions": sorted({row.token_position for row in seq_rows}),
                    "modified_sites": sorted({site_label_for_row(row) for row in seq_rows}),
                    "sample_cap": args.max_lb06_points,
                }
            )
            points_path = output_png.with_suffix(".points.jsonl")
            save_points(seq_rows, points_path)
            summary["points_jsonl"] = str(points_path)
            summaries.append(summary)

    summary = {
        "lb07_jsonl": args.lb07_jsonl,
        "lb06_jsonl": args.lb06_jsonl,
        "embedding_source_requested": args.embedding_source,
        "embedding_source_normalized": embedding_source,
        "embedding_label": embedding_label,
        "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
        "annotate_site_labels": args.annotate_site_labels,
        "max_lb06_read_plots": args.max_lb06_read_plots,
        "sequence_key": args.sequence_key,
        "lb06_records": len(lb06_records),
        "lb07_records": len(lb07_records),
        "lb06_modified_points_total": len(lb06_rows),
        "lb07_unmodified_same_site_points_total": len(lb07_rows),
        "lb06_modified_points_by_sequence": dict(Counter(row.sequence_key for row in lb06_rows)),
        "lb07_same_site_points_by_sequence": dict(Counter(row.sequence_key for row in lb07_rows)),
        "modified_positions_by_sequence": summarize_positions(positions_by_sequence),
        "modified_sites_by_sequence": {
            sequence_key: {
                site_key: {
                    "base_position_1based": base_position_by_site.get(site_key),
                    "modified_token_positions": sorted(token_positions),
                }
                for site_key, token_positions in sorted(site_map.items())
            }
            for sequence_key, site_map in sorted(positions_by_site.items())
        },
        "plots": summaries,
    }
    summary_path = output_dir / f"LB06_vs_LB07_same_site_C_{embedding_label}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"LB06 records: {len(lb06_records)}")
    print(f"LB07 records: {len(lb07_records)}")
    print(f"LB06 modified C points: {len(lb06_rows)}")
    print(f"LB07 same-site unmodified C points: {len(lb07_rows)}")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
