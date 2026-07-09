#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
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
    collect_labeled_c_embeddings,
    forward_sequence_hidden,
    iter_records,
    make_batch,
    normalize_embedding_source,
)
from plot_lb06_lb07_same_site_c_embedding_distribution import (  # noqa: E402
    DEFAULT_LB06_JSONL,
    DEFAULT_LB07_JSONL,
    EmbeddingRow,
    build_modified_site_maps,
    collect_dataset_embeddings,
    collect_label_positions,
    collect_mixed_dataset_embeddings,
    modified_positions_from_rows,
    safe_name,
    sequence_key_for_record,
)


def split_records_by_read(
    records: list[ReadRecord],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, str]:
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("--train-frac must be >0, --val-frac >=0, and train+val must be <1.")

    read_ids = sorted({record.record_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(read_ids)

    n = len(read_ids)
    n_train = max(1, int(round(n * train_frac)))
    n_val = int(round(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    split_by_read: dict[str, str] = {}
    for read_id in read_ids[:n_train]:
        split_by_read[read_id] = "train"
    for read_id in read_ids[n_train : n_train + n_val]:
        split_by_read[read_id] = "val"
    for read_id in read_ids[n_train + n_val :]:
        split_by_read[read_id] = "test"
    return split_by_read


def split_dataset_records_by_read(
    datasets: list[tuple[str, list[ReadRecord]]],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, str]:
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("--train-frac must be >0, --val-frac >=0, and train+val must be <1.")

    keys = []
    for dataset_name, records in datasets:
        keys.extend(f"{dataset_name}:{read_id}" for read_id in sorted({record.record_id for record in records}))
    keys = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = max(1, int(round(n * train_frac)))
    n_val = int(round(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    split_by_key: dict[str, str] = {}
    for key in keys[:n_train]:
        split_by_key[key] = "train"
    for key in keys[n_train : n_train + n_val]:
        split_by_key[key] = "val"
    for key in keys[n_train + n_val :]:
        split_by_key[key] = "test"
    return split_by_key


def empty_probe_data() -> dict[str, dict[str, Any]]:
    return {
        split: {
            "x": [],
            "y": [],
            "dataset": [],
            "sequence_key": [],
            "read_id": [],
            "token_position": [],
            "site_key": [],
            "base_position_1based": [],
        }
        for split in ("train", "val", "test")
    }


def finalize_probe_data(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for split in ("train", "val", "test"):
        if data[split]["x"]:
            data[split]["x"] = np.stack(data[split]["x"], axis=0).astype(np.float32)
            data[split]["y"] = np.asarray(data[split]["y"], dtype=np.int64)
        else:
            data[split]["x"] = np.empty((0, 0), dtype=np.float32)
            data[split]["y"] = np.empty((0,), dtype=np.int64)
    return data


def rows_to_probe_data(rows: list[EmbeddingRow], split_by_dataset_read: dict[str, str]) -> dict[str, dict[str, Any]]:
    data = empty_probe_data()
    for row in rows:
        split_key = f"{row.dataset}:{row.read_id}"
        split = split_by_dataset_read.get(split_key)
        if split is None:
            continue
        data[split]["x"].append(row.embedding.astype(np.float32))
        data[split]["y"].append(1 if row.dataset.startswith("LB06") else 0)
        data[split]["dataset"].append(row.dataset)
        data[split]["sequence_key"].append(row.sequence_key)
        data[split]["read_id"].append(row.read_id)
        data[split]["token_position"].append(int(row.token_position))
        data[split]["site_key"].append(row.site_key)
        data[split]["base_position_1based"].append(row.base_position_1based)
    return finalize_probe_data(data)


def extract_embedding_dataset(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    split_by_read: dict[str, str],
    *,
    embedding_source: str,
) -> dict[str, dict[str, Any]]:
    data = empty_probe_data()

    batch: list[ReadRecord] = []
    pbar = tqdm(total=len(records), desc=f"extracting C-token {args.embedding_source} embeddings", unit="read")

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
            sequence_hidden = forward_sequence_hidden(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                backbone_chunk_size=args.backbone_chunk_size,
                embedding_source=embedding_source,
            ).float().detach().cpu().numpy()

        for idx, record in enumerate(batch):
            split = split_by_read[record.record_id]
            sequence_key = sequence_key_for_record(record, args.sequence_key)
            valid_len = min(effective_lengths[idx], sequence_hidden.shape[1])
            embeddings, labels, points = collect_labeled_c_embeddings(
                record,
                sequence_hidden[idx, :valid_len],
                valid_len=valid_len,
            )
            if embeddings.shape[0] == 0:
                continue
            data[split]["x"].append(embeddings)
            data[split]["y"].append(labels)
            data[split]["dataset"].extend(["single_jsonl"] * int(labels.shape[0]))
            data[split]["sequence_key"].extend([sequence_key] * int(labels.shape[0]))
            data[split]["read_id"].extend([record.record_id] * int(labels.shape[0]))
            data[split]["token_position"].extend([int(point["token_position"]) for point in points])
            data[split]["site_key"].extend([None] * int(labels.shape[0]))
            data[split]["base_position_1based"].extend([None] * int(labels.shape[0]))
        pbar.update(len(batch))
        batch = []

    for record in records:
        batch.append(record)
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()
    pbar.close()

    for split in ("train", "val", "test"):
        if data[split]["x"]:
            data[split]["x"] = np.concatenate(data[split]["x"], axis=0).astype(np.float32)
            data[split]["y"] = np.concatenate(data[split]["y"], axis=0).astype(np.int64)
        else:
            data[split]["x"] = np.empty((0, 0), dtype=np.float32)
            data[split]["y"] = np.empty((0,), dtype=np.int64)
    return data


def extract_same_site_embedding_dataset(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    lb07_records: list[ReadRecord],
    lb06_records: list[ReadRecord],
    split_by_dataset_read: dict[str, str],
    *,
    embedding_source: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    positions_by_sequence = collect_label_positions(
        lb06_records,
        sequence_key_mode=args.sequence_key,
        label_values={2},
    )
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
            raise ValueError("No LB06 label=2 modified C embeddings were collected.")
        positions_by_site, site_by_sequence_token, base_position_by_site = build_modified_site_maps(
            lb06_records,
            sequence_key_mode=args.sequence_key,
            fallback_positions_by_sequence=positions_by_sequence,
        )
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
        raise ValueError("No LB06 same-site positive embeddings were collected.")
    if not lb07_rows:
        raise ValueError("No LB07 same-site negative embeddings were collected.")

    rows = lb06_rows + lb07_rows
    data = rows_to_probe_data(rows, split_by_dataset_read)
    metadata = {
        "lb06_same_site_points_total": len(lb06_rows),
        "lb07_same_site_points_total": len(lb07_rows),
        "lb06_points_by_sequence": dict(Counter(row.sequence_key for row in lb06_rows)),
        "lb07_points_by_sequence": dict(Counter(row.sequence_key for row in lb07_rows)),
        "modified_positions_by_sequence": {
            sequence_key: sorted(positions)
            for sequence_key, positions in sorted(positions_by_sequence.items())
        },
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
    }
    return data, metadata


def subsample_split(
    split_data: dict[str, Any],
    *,
    max_negative_tokens: int,
    max_positive_tokens: int,
    seed: int,
) -> dict[str, Any]:
    x = split_data["x"]
    y = split_data["y"]
    if y.size == 0:
        return split_data

    rng = np.random.default_rng(seed)
    neg_idx = np.flatnonzero(y == 0)
    pos_idx = np.flatnonzero(y == 1)
    if max_negative_tokens > 0 and neg_idx.size > max_negative_tokens:
        neg_idx = np.sort(rng.choice(neg_idx, size=max_negative_tokens, replace=False))
    if max_positive_tokens > 0 and pos_idx.size > max_positive_tokens:
        pos_idx = np.sort(rng.choice(pos_idx, size=max_positive_tokens, replace=False))
    keep_idx = np.concatenate([neg_idx, pos_idx])
    keep_idx.sort()

    out = {"x": x[keep_idx], "y": y[keep_idx]}
    keep_list = keep_idx.tolist()
    for key, value in split_data.items():
        if key in {"x", "y"}:
            continue
        if isinstance(value, list):
            out[key] = [value[int(i)] for i in keep_list]
        else:
            out[key] = value
    return out


def standardize_splits(data: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    train_x = data["train"]["x"]
    if train_x.size == 0:
        raise ValueError("No train embeddings were collected.")
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    out = {}
    for split, split_data in data.items():
        out[split] = dict(split_data)
        if split_data["x"].size:
            out[split]["x"] = ((split_data["x"] - mean) / std).astype(np.float32)
    return out, {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def auc_roc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    rank = np.arange(1, len(y_sorted) + 1)
    precision = tp / rank
    return float(precision[y_sorted == 1].sum() / n_pos)


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, *, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    total = max(1, len(y_true))
    pos = tp + fn
    neg = tn + fp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(pos, 1)
    specificity = tn / max(neg, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": int(len(y_true)),
        "positive": int(pos),
        "negative": int(neg),
        "positive_rate": float(pos / total),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / total),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "roc_auc": auc_roc(y_true, prob),
        "average_precision": average_precision(y_true, prob),
    }


def best_threshold_by_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    candidates = np.unique(np.quantile(prob, np.linspace(0.0, 1.0, 201)))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        metrics = binary_metrics(y_true, prob, threshold=float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def train_linear_probe(
    data: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    torch.manual_seed(seed)
    x_train = torch.from_numpy(data["train"]["x"]).float().to(device)
    y_train = torch.from_numpy(data["train"]["y"]).float().to(device)
    x_val = torch.from_numpy(data["val"]["x"]).float().to(device) if data["val"]["x"].size else None
    y_val = torch.from_numpy(data["val"]["y"]).float().to(device) if data["val"]["y"].size else None

    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise ValueError("Train split is empty.")
    if int(y_train.sum().item()) == 0 or int((y_train == 0).sum().item()) == 0:
        raise ValueError("Train split must contain both modified and unmodified C tokens.")

    model = torch.nn.Linear(x_train.shape[1], 1).to(device)
    pos = y_train.sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), min=1.0, max=100.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}

    n = x_train.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_state = None
    best_val = float("inf")

    for _epoch in tqdm(range(1, epochs + 1), desc="training linear probe", unit="epoch"):
        model.train()
        perm = torch.randperm(n, generator=generator, device="cpu").to(device)
        running_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_train[idx]).squeeze(-1)
            loss = loss_fn(logits, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item()) * int(idx.numel())
            seen += int(idx.numel())
        train_loss = running_loss / max(seen, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            if x_val is not None and x_val.shape[0] > 0:
                val_logits = model(x_val).squeeze(-1)
                val_loss = float(loss_fn(val_logits, y_val).item())
            else:
                val_loss = train_loss
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_prob(model: torch.nn.Module, x: np.ndarray, *, device: torch.device, batch_size: int) -> np.ndarray:
    if x.size == 0:
        return np.empty((0,), dtype=np.float32)
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            prob = torch.sigmoid(model(xb).squeeze(-1)).detach().cpu().numpy()
            probs.append(prob.astype(np.float32))
    return np.concatenate(probs, axis=0)


def summarize_split_reads(records: list[ReadRecord], split_by_read: dict[str, str]) -> dict[str, Any]:
    split_counts = Counter(split_by_read.values())
    rows = defaultdict(list)
    for record in records:
        rows[split_by_read[record.record_id]].append(record.record_id)
    return {
        "read_count_by_split": dict(split_counts),
        "example_reads_by_split": {split: values[:5] for split, values in rows.items()},
    }


def summarize_split_dataset_reads(split_by_dataset_read: dict[str, str]) -> dict[str, Any]:
    split_counts = Counter(split_by_dataset_read.values())
    dataset_split_counts: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for key, split in split_by_dataset_read.items():
        dataset = key.split(":", 1)[0]
        dataset_split_counts[dataset][split] += 1
        if len(examples[split]) < 5:
            examples[split].append(key)
    return {
        "read_count_by_split": dict(split_counts),
        "read_count_by_dataset_split": {
            dataset: dict(counts)
            for dataset, counts in sorted(dataset_split_counts.items())
        },
        "example_dataset_reads_by_split": dict(examples),
    }


def token_counts_by_split(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "n": int(data[split]["y"].size),
            "positive": int((data[split]["y"] == 1).sum()),
            "negative": int((data[split]["y"] == 0).sum()),
        }
        for split in ("train", "val", "test")
    }


def sequence_keys_in_data(data: dict[str, dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for split in ("train", "val", "test"):
        keys.update(str(value) for value in data[split].get("sequence_key", []) if value is not None)
    return sorted(keys)


def filter_data_by_sequence(
    data: dict[str, dict[str, Any]],
    sequence_key: str,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        split_data = data[split]
        sequence_values = split_data.get("sequence_key", [])
        keep_idx = [idx for idx, value in enumerate(sequence_values) if str(value) == str(sequence_key)]
        if keep_idx:
            keep_array = np.asarray(keep_idx, dtype=np.int64)
            out_split: dict[str, Any] = {
                "x": split_data["x"][keep_array],
                "y": split_data["y"][keep_array],
            }
        else:
            feature_dim = split_data["x"].shape[1] if split_data["x"].ndim == 2 and split_data["x"].shape[1] else 0
            out_split = {
                "x": np.empty((0, feature_dim), dtype=np.float32),
                "y": np.empty((0,), dtype=np.int64),
            }
        for key, value in split_data.items():
            if key in {"x", "y"}:
                continue
            if isinstance(value, list):
                out_split[key] = [value[idx] for idx in keep_idx]
            else:
                out_split[key] = value
        out[split] = out_split
    return out


def split_has_both_classes(split_data: dict[str, Any]) -> bool:
    y = split_data["y"]
    return bool(y.size and (y == 1).any() and (y == 0).any())


def run_linear_probe_experiment(
    args: argparse.Namespace,
    *,
    raw_data: dict[str, dict[str, Any]],
    classifier_device: torch.device,
    embedding_source: str,
    embedding_label: str,
    output_dir: Path,
    mode_label: str,
    summary_base: dict[str, Any],
    save_name: str,
    sequence_key: str | None = None,
) -> dict[str, Any]:
    sampled_data = {}
    for split, split_data in raw_data.items():
        sampled_data[split] = subsample_split(
            split_data,
            max_negative_tokens=args.max_negative_tokens,
            max_positive_tokens=args.max_positive_tokens,
            seed=args.seed + {"train": 0, "val": 1, "test": 2}[split],
        )

    data, scaler = standardize_splits(sampled_data)
    probe, history = train_linear_probe(
        data,
        device=classifier_device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.classifier_batch_size,
        seed=args.seed,
    )

    val_prob = predict_prob(probe, data["val"]["x"], device=classifier_device, batch_size=args.classifier_batch_size)
    test_prob = predict_prob(probe, data["test"]["x"], device=classifier_device, batch_size=args.classifier_batch_size)
    train_prob = predict_prob(probe, data["train"]["x"], device=classifier_device, batch_size=args.classifier_batch_size)
    threshold = best_threshold_by_f1(data["val"]["y"], val_prob) if val_prob.size else 0.5

    metrics = {
        "train_at_0.5": binary_metrics(data["train"]["y"], train_prob, threshold=0.5),
        "val_at_0.5": binary_metrics(data["val"]["y"], val_prob, threshold=0.5),
        "test_at_0.5": binary_metrics(data["test"]["y"], test_prob, threshold=0.5),
        "val_at_best_val_f1_threshold": binary_metrics(data["val"]["y"], val_prob, threshold=threshold),
        "test_at_best_val_f1_threshold": binary_metrics(data["test"]["y"], test_prob, threshold=threshold),
        "best_val_f1_threshold": float(threshold),
    }

    summary = {
        **summary_base,
        "sequence_key_for_training": sequence_key,
        "raw_token_counts": token_counts_by_split(raw_data),
        "sampled_token_counts": token_counts_by_split(data),
        "classifier": {
            "type": "linear_logistic_probe",
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "classifier_batch_size": args.classifier_batch_size,
            "history": history,
        },
        "metrics": metrics,
    }

    model_path = output_dir / f"{save_name}_{embedding_label}_linear_probe.pt"
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "embedding_source": embedding_source,
            "scaler_mean": scaler["mean"],
            "scaler_std": scaler["std"],
            "threshold": threshold,
            "summary": summary,
        },
        model_path,
    )
    summary["probe_checkpoint"] = str(model_path)

    if args.save_embeddings:
        npz_path = output_dir / f"{save_name}_{embedding_label}_linear_probe_arrays.npz"
        np.savez_compressed(
            npz_path,
            train_x=data["train"]["x"],
            train_y=data["train"]["y"],
            val_x=data["val"]["x"],
            val_y=data["val"]["y"],
            test_x=data["test"]["x"],
            test_y=data["test"]["y"],
        )
        summary["arrays_npz"] = str(npz_path)

    summary_path = output_dir / f"{save_name}_{embedding_label}_linear_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small read-split linear probe for modified-vs-unmodified C-token "
            "classification from BERT/context or DLM/ODE embeddings."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument(
        "--probe-mode",
        choices=("same-site-c", "original-c-labels"),
        default="same-site-c",
        help="same-site-c probes LB06 modified C vs LB07 same-site C; original-c-labels keeps the old single-jsonl label=2 vs label=1 probe.",
    )
    parser.add_argument("--jsonl", default=None, help="Input jsonl/jsonl.gz for --probe-mode original-c-labels.")
    parser.add_argument("--lb07-jsonl", default=DEFAULT_LB07_JSONL, help="LB07 jsonl/jsonl.gz for --probe-mode same-site-c.")
    parser.add_argument("--lb06-jsonl", default=DEFAULT_LB06_JSONL, help="LB06 jsonl/jsonl.gz for --probe-mode same-site-c.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-source", choices=("bert", "dlm", "context_hidden", "ode_hidden"), default="dlm")
    parser.add_argument("--limit-reads", type=int, default=None)
    parser.add_argument("--limit-lb07-reads", type=int, default=None)
    parser.add_argument("--limit-lb06-reads", type=int, default=None)
    parser.add_argument("--sequence-key", choices=("auto", "label", "ref", "seq"), default="label")
    parser.add_argument(
        "--c-mod-site-batch-mode",
        choices=("separate", "mixed"),
        default="separate",
        help="Only affects --probe-mode same-site-c. mixed interleaves LB06/LB07 reads in the same forward batches.",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for embedding extraction.")
    parser.add_argument("--classifier-batch-size", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pad-token-id", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classifier-device", default=None, help="Defaults to --device.")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--backbone-chunk-size", type=int, default=2000)
    parser.add_argument("--elf-ode-steps", type=int, default=4)
    parser.add_argument("--elf-ode-start-t", type=float, default=0.85)
    parser.add_argument("--elf-self-cond-cfg-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-negative-tokens", type=int, default=200000, help="Per split cap; 0 keeps all.")
    parser.add_argument("--max-positive-tokens", type=int, default=0, help="Per split cap; 0 keeps all.")
    parser.add_argument(
        "--train-scope",
        choices=("all-sequences", "per-sequence"),
        default="all-sequences",
        help="all-sequences trains one probe on all sequence keys; per-sequence trains one independent probe per sequence key.",
    )
    parser.add_argument("--save-embeddings", action="store_true", help="Also save standardized probe arrays as .npz.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_source = normalize_embedding_source(args.embedding_source)
    embedding_label = "bert" if embedding_source == "context_hidden" else "dlm"
    device = torch.device(args.device)
    classifier_device = torch.device(args.classifier_device or args.device)

    model = build_model(args, device, embedding_source)
    mode_metadata: dict[str, Any] = {}
    split_summary: dict[str, Any]

    if args.probe_mode == "original-c-labels":
        if args.jsonl is None:
            raise ValueError("--jsonl is required when --probe-mode original-c-labels.")
        records = list(iter_records(Path(args.jsonl)))
        if args.limit_reads is not None:
            records = records[: args.limit_reads]
        if len(records) < 3:
            raise ValueError("Need at least 3 reads for a train/val/test split.")

        split_by_read = split_records_by_read(
            records,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        raw_data = extract_embedding_dataset(
            args,
            model,
            device,
            records,
            split_by_read,
            embedding_source=embedding_source,
        )
        split_summary = summarize_split_reads(records, split_by_read)
        input_summary = {
            "input_jsonl": args.jsonl,
            "records": len(records),
        }
    else:
        lb07_records = list(iter_records(Path(args.lb07_jsonl)))
        lb06_records = list(iter_records(Path(args.lb06_jsonl)))
        limit_lb07 = args.limit_lb07_reads if args.limit_lb07_reads is not None else args.limit_reads
        limit_lb06 = args.limit_lb06_reads if args.limit_lb06_reads is not None else args.limit_reads
        if limit_lb07 is not None:
            lb07_records = lb07_records[:limit_lb07]
        if limit_lb06 is not None:
            lb06_records = lb06_records[:limit_lb06]
        if len(lb07_records) < 2 or len(lb06_records) < 2:
            raise ValueError("Need at least 2 LB07 and 2 LB06 reads for same-site probing.")

        split_by_dataset_read = split_dataset_records_by_read(
            [
                ("LB06_modified", lb06_records),
                ("LB07_unmodified_same_site", lb07_records),
            ],
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
        )
        raw_data, mode_metadata = extract_same_site_embedding_dataset(
            args,
            model,
            device,
            lb07_records,
            lb06_records,
            split_by_dataset_read,
            embedding_source=embedding_source,
        )
        split_summary = summarize_split_dataset_reads(split_by_dataset_read)
        input_summary = {
            "lb07_jsonl": args.lb07_jsonl,
            "lb06_jsonl": args.lb06_jsonl,
            "lb07_records": len(lb07_records),
            "lb06_records": len(lb06_records),
        }

    summary_base = {
        "probe_mode": args.probe_mode,
        **input_summary,
        "model_name_or_path": args.model_name_or_path,
        "embedding_source_requested": args.embedding_source,
        "embedding_source_normalized": embedding_source,
        "embedding_label": embedding_label,
        "limit_reads": args.limit_reads,
        "limit_lb07_reads": args.limit_lb07_reads,
        "limit_lb06_reads": args.limit_lb06_reads,
        "sequence_key": args.sequence_key,
        "c_mod_site_batch_mode": args.c_mod_site_batch_mode,
        "train_scope": args.train_scope,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "max_negative_tokens_per_split": args.max_negative_tokens,
        "max_positive_tokens_per_split": args.max_positive_tokens,
        "same_site_metadata": mode_metadata,
        "split_reads": split_summary,
    }

    mode_label = f"same_site_C_{args.c_mod_site_batch_mode}" if args.probe_mode == "same-site-c" else "c_modification"
    if args.train_scope == "all-sequences":
        summary = run_linear_probe_experiment(
            args,
            raw_data=raw_data,
            classifier_device=classifier_device,
            embedding_source=embedding_source,
            embedding_label=embedding_label,
            output_dir=output_dir,
            mode_label=mode_label,
            summary_base=summary_base,
            save_name=mode_label,
            sequence_key=None,
        )
        print(f"Embedding source: {args.embedding_source} -> {embedding_source}")
        print(f"Summary: {summary['summary_json']}")
        print(f"Probe checkpoint: {summary['probe_checkpoint']}")
        print("Test metrics at 0.5:")
        print(json.dumps(summary["metrics"]["test_at_0.5"], ensure_ascii=False, indent=2))
        print("Test metrics at best val-F1 threshold:")
        print(json.dumps(summary["metrics"]["test_at_best_val_f1_threshold"], ensure_ascii=False, indent=2))
        return

    per_sequence_dir = output_dir / "per_sequence"
    per_sequence_dir.mkdir(parents=True, exist_ok=True)
    per_sequence_results = []
    skipped_sequences = []
    for seq_index, sequence_key in enumerate(sequence_keys_in_data(raw_data)):
        seq_raw_data = filter_data_by_sequence(raw_data, sequence_key)
        if not split_has_both_classes(seq_raw_data["train"]):
            skipped_sequences.append(
                {
                    "sequence_key": sequence_key,
                    "reason": "train split does not contain both positive and negative samples",
                    "raw_token_counts": token_counts_by_split(seq_raw_data),
                }
            )
            continue
        seq_output_dir = per_sequence_dir / safe_name(sequence_key)
        seq_output_dir.mkdir(parents=True, exist_ok=True)
        seq_summary_base = {
            **summary_base,
            "sequence_key_for_training": sequence_key,
        }
        seq_save_name = f"{mode_label}_{safe_name(sequence_key)}"
        try:
            seq_summary = run_linear_probe_experiment(
                args,
                raw_data=seq_raw_data,
                classifier_device=classifier_device,
                embedding_source=embedding_source,
                embedding_label=embedding_label,
                output_dir=seq_output_dir,
                mode_label=mode_label,
                summary_base=seq_summary_base,
                save_name=seq_save_name,
                sequence_key=sequence_key,
            )
        except ValueError as exc:
            skipped_sequences.append(
                {
                    "sequence_key": sequence_key,
                    "reason": str(exc),
                    "raw_token_counts": token_counts_by_split(seq_raw_data),
                }
            )
            continue
        per_sequence_results.append(
            {
                "sequence_key": sequence_key,
                "summary_json": seq_summary["summary_json"],
                "probe_checkpoint": seq_summary["probe_checkpoint"],
                "raw_token_counts": seq_summary["raw_token_counts"],
                "sampled_token_counts": seq_summary["sampled_token_counts"],
                "metrics": seq_summary["metrics"],
            }
        )

    index_summary = {
        **summary_base,
        "available_sequence_keys": sequence_keys_in_data(raw_data),
        "trained_sequence_count": len(per_sequence_results),
        "skipped_sequence_count": len(skipped_sequences),
        "per_sequence_results": per_sequence_results,
        "skipped_sequences": skipped_sequences,
    }
    index_path = output_dir / f"{mode_label}_{embedding_label}_per_sequence_linear_probe_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Embedding source: {args.embedding_source} -> {embedding_source}")
    print(f"Per-sequence index: {index_path}")
    print(f"Trained sequences: {len(per_sequence_results)}")
    print(f"Skipped sequences: {len(skipped_sequences)}")


if __name__ == "__main__":
    main()
