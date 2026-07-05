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


def extract_embedding_dataset(
    args: argparse.Namespace,
    model: BasecallModel,
    device: torch.device,
    records: list[ReadRecord],
    split_by_read: dict[str, str],
    *,
    embedding_source: str,
) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {
        split: {"x": [], "y": [], "read_id": [], "token_position": []}
        for split in ("train", "val", "test")
    }

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
            data[split]["read_id"].extend([record.record_id] * int(labels.shape[0]))
            data[split]["token_position"].extend([int(point["token_position"]) for point in points])
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

    return {
        "x": x[keep_idx],
        "y": y[keep_idx],
        "read_id": [split_data["read_id"][int(i)] for i in keep_idx.tolist()],
        "token_position": [split_data["token_position"][int(i)] for i in keep_idx.tolist()],
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small read-split linear probe for modified-vs-unmodified C-token "
            "classification from BERT/context or DLM/ODE embeddings."
        )
    )
    parser.add_argument("--model-name-or-path", required=True, help="Stage3 HF DLM model directory.")
    parser.add_argument("--jsonl", required=True, help="Input jsonl/jsonl.gz containing input_ids and c_modification_label.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-source", choices=("bert", "dlm", "context_hidden", "ode_hidden"), default="dlm")
    parser.add_argument("--limit-reads", type=int, default=None)
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
    parser.add_argument("--save-embeddings", action="store_true", help="Also save standardized probe arrays as .npz.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_source = normalize_embedding_source(args.embedding_source)
    embedding_label = "bert" if embedding_source == "context_hidden" else "dlm"
    device = torch.device(args.device)
    classifier_device = torch.device(args.classifier_device or args.device)

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

    model = build_model(args, device, embedding_source)
    raw_data = extract_embedding_dataset(
        args,
        model,
        device,
        records,
        split_by_read,
        embedding_source=embedding_source,
    )

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
        "input_jsonl": args.jsonl,
        "model_name_or_path": args.model_name_or_path,
        "embedding_source_requested": args.embedding_source,
        "embedding_source_normalized": embedding_source,
        "embedding_label": embedding_label,
        "limit_reads": args.limit_reads,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "max_negative_tokens_per_split": args.max_negative_tokens,
        "max_positive_tokens_per_split": args.max_positive_tokens,
        "raw_token_counts": {
            split: {
                "n": int(raw_data[split]["y"].size),
                "modified": int((raw_data[split]["y"] == 1).sum()),
                "unmodified": int((raw_data[split]["y"] == 0).sum()),
            }
            for split in ("train", "val", "test")
        },
        "sampled_token_counts": {
            split: {
                "n": int(data[split]["y"].size),
                "modified": int((data[split]["y"] == 1).sum()),
                "unmodified": int((data[split]["y"] == 0).sum()),
            }
            for split in ("train", "val", "test")
        },
        "split_reads": summarize_split_reads(records, split_by_read),
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

    model_path = output_dir / f"c_modification_{embedding_label}_linear_probe.pt"
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

    if args.save_embeddings:
        npz_path = output_dir / f"c_modification_{embedding_label}_linear_probe_arrays.npz"
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

    summary_path = output_dir / f"c_modification_{embedding_label}_linear_probe_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Embedding source: {args.embedding_source} -> {embedding_source}")
    print(f"Summary: {summary_path}")
    print(f"Probe checkpoint: {model_path}")
    print("Test metrics at 0.5:")
    print(json.dumps(metrics["test_at_0.5"], ensure_ascii=False, indent=2))
    print("Test metrics at best val-F1 threshold:")
    print(json.dumps(metrics["test_at_best_val_f1_threshold"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
