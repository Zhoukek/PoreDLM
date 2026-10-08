#!/usr/bin/env python3
"""Train the baseline-conditioned frozen-encoder S18 three-model benchmark."""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


MODELS = ("V003_Stone",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--mix-logit-init", type=float, default=0.0)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--whiten-shrinkage", type=float, default=0.05)
    parser.add_argument("--whiten-components", type=int, default=768)
    return parser.parse_args()


class MLPPlusCosinePCAWhitenedResidualHead:
    """MLP over PCA-whitened [z, abs(z)] plus a cosine branch over z."""

    @staticmethod
    def make(torch, z_dim: int, mix_logit_init: float):
        import torch.nn as nn
        import torch.nn.functional as F

        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.LayerNorm(z_dim * 2),
                    nn.Linear(z_dim * 2, 256),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(256,1),
                )
                self.cosine_weight = nn.Parameter(torch.empty(2, z_dim))
                self.cosine_bias = nn.Parameter(torch.zeros(2))
                self.cosine_log_scale = nn.Parameter(torch.log(torch.tensor(32.0)))
                self.mix_logit = nn.Parameter(torch.tensor(float(mix_logit_init)))
                nn.init.xavier_uniform_(self.cosine_weight)

            def forward(self, x):
                mlp_logit = self.mlp(x)
                z = x[:, :self.cosine_weight.shape[1]]
                z = F.normalize(z, p=2, dim=-1)
                weight = F.normalize(self.cosine_weight, p=2, dim=-1)
                scale = self.cosine_log_scale.exp().clamp(1.0, 100.0)
                cosine_logits = scale * (z @ weight.t()) + self.cosine_bias
                cosine_logit = cosine_logits[:, 1:2] - cosine_logits[:, 0:1]
                # cosine_logit = cosine_logits[:, 1:2]
                alpha = torch.sigmoid(self.mix_logit)
                return mlp_logit + alpha * cosine_logit
                # return mlp_logit

        return Head()


def patch_torch_optimizer(torch):
    """Avoid the broken torch._dynamo import in this torch2d1 environment."""
    import torch.optim

    optimizer_cls = torch.optim.Optimizer
    optimizer_cls.add_param_group = optimizer_cls.add_param_group.__wrapped__
    optimizer_cls.zero_grad = optimizer_cls.zero_grad.__wrapped__

    raw_step = torch.optim.AdamW.step.__wrapped__

    @functools.wraps(raw_step)
    def safe_step(self, closure=None):
        previous_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults["differentiable"])
            return raw_step(self, closure)
        finally:
            torch.set_grad_enabled(previous_grad)

    torch.optim.AdamW.step = safe_step


def load_metadata(path: Path) -> dict[str, np.ndarray]:
    data = pq.read_table(path, columns=[
        "row_index", "window_id", "site_pos0", "label", "ref_7mer",
        "fast5_raw_read_id", "fold", "role",
    ]).to_pydict()
    return {
        "row_index": np.asarray(data["row_index"], dtype=np.int64),
        "window_id": np.asarray(data["window_id"], dtype=object),
        "site_pos0": np.asarray(data["site_pos0"], dtype=np.int64),
        "label": np.asarray(data["label"], dtype=np.int8),
        "ref_7mer": np.asarray(data["ref_7mer"], dtype=object),
        "fast5_raw_read_id": np.asarray(data["fast5_raw_read_id"], dtype=object),
        "fold": np.asarray(data["fold"], dtype=np.int8),
        "role": np.asarray(data["role"], dtype=object),
    }


def load_embedding(path: Path, rows: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path, mmap_mode="r")
    if values.shape != (rows, 768):
        raise RuntimeError(f"Unexpected embedding shape {values.shape} for {path}")
    return values


def robust_centers(
    embedding: np.ndarray,
    indices: np.ndarray,
    kmer_ids: np.ndarray,
    kmer_count: int,
) -> np.ndarray:
    centers = np.zeros((kmer_count, 768), dtype=np.float32)
    for kmer_id in range(kmer_count):
        selected = indices[kmer_ids[indices] == kmer_id]
        if selected.size == 0:
            raise RuntimeError(f"No baseline examples for kmer index {kmer_id}")
        values = np.asarray(embedding[selected], dtype=np.float32)
        ordered = np.sort(values, axis=0)
        trim = int(values.shape[0] * 0.10)
        if trim * 2 >= values.shape[0]:
            trim = 0
        centers[kmer_id] = ordered[trim:values.shape[0] - trim].mean(axis=0)
    return centers


def robust_scale(
    embedding: np.ndarray,
    indices: np.ndarray,
    kmer_ids: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    values = np.asarray(embedding[indices], dtype=np.float32)
    residual = values - centers[kmer_ids[indices]]
    scale = 1.4826 * np.median(np.abs(residual), axis=0)
    return np.maximum(scale.astype(np.float32), 1e-3)


def fit_residual_whitener(
    embedding: np.ndarray,
    indices: np.ndarray,
    kmer_ids: np.ndarray,
    centers: np.ndarray,
    eps: float,
    shrinkage: float,
    components: int,
) -> dict[str, np.ndarray]:
    if components <= 0 or components > 768:
        raise ValueError(f"--whiten-components must be in [1, 768], got {components}")
    values = np.asarray(embedding[indices], dtype=np.float32)
    residual = values - centers[kmer_ids[indices]]
    residual = residual.astype(np.float64, copy=False)
    cov = (residual.T @ residual) / max(residual.shape[0] - 1, 1)
    mean_diag = float(np.mean(np.diag(cov)))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    cov = (1.0 - shrinkage) * cov + shrinkage * mean_diag * np.eye(cov.shape[0], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, float(eps))
    order = np.argsort(eigvals)[::-1][:components]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    inv_sqrt = 1.0 / np.sqrt(eigvals)
    whitener = eigvecs * inv_sqrt[None, :]
    return {
        "whitener": whitener.astype(np.float32),
        "eigvals": eigvals.astype(np.float32),
        "components": np.asarray([components], dtype=np.int32),
    }


def make_features(
    embedding: np.ndarray,
    indices: np.ndarray,
    kmer_ids: np.ndarray,
    centers: np.ndarray,
    whitener: np.ndarray,
) -> np.ndarray:
    values = np.asarray(embedding[indices], dtype=np.float32)
    residual = values - centers[kmer_ids[indices]]
    z = residual @ whitener
    if not np.isfinite(z).all():
        raise RuntimeError("Non-finite whitened residual features")
    z = np.clip(z, -8.0, 8.0).astype(np.float32)
    return np.concatenate((z, np.abs(z)), axis=1)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def train_head(X_train, y_train, X_val, y_val, args, fold: int, model_name: str, device, torch):
    torch.manual_seed(1700 + fold)
    random.seed(1700 + fold)
    head = MLPPlusCosinePCAWhitenedResidualHead.make(torch, X_train.shape[1] // 2, args.mix_logit_init).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_x = torch.from_numpy(np.ascontiguousarray(X_train))
    train_y = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(1700 + fold),
    )
    val_x = torch.from_numpy(np.ascontiguousarray(X_val)).to(device)
    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(batch_x).squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_x.shape[0]
            total_rows += batch_x.shape[0]
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_x).squeeze(-1).float().cpu().numpy()
        val_prob = sigmoid(val_logits)
        val_auc = float(roc_auc_score(y_val, val_prob))
        row = {"model": model_name, "fold": fold, "epoch": epoch, "loss": total_loss / max(total_rows, 1), "val_auroc": val_auc}
        history.append(row)
        print(json.dumps({"stage": "train", **row}), flush=True)
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {model_name} fold {fold}")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        final_val = head(val_x).squeeze(-1).float().cpu().numpy()
    del train_x, train_y, val_x, loader, optimizer, loss_fn
    gc.collect()
    return head, final_val, best_epoch, best_auc, history


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    score = tpr - fpr
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5
    best = np.flatnonzero(valid)[int(np.argmax(score[valid]))]
    return float(np.clip(thresholds[best], 1e-4, 1.0 - 1e-4))


def calculate_metrics(
    dataset: str,
    model: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_reads: int | None = None,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "dataset": dataset,
        "model": model,
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "n_reads": None if n_reads is None else int(n_reads),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def aggregate_sites(meta: dict[str, np.ndarray], probabilities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    groups: dict[int, dict[str, object]] = {}
    for index, site in enumerate(meta["site_pos0"]):
        site = int(site)
        group = groups.setdefault(site, {
            "label": int(meta["label"][index]),
            "kmer": str(meta["ref_7mer"][index]),
            "reads": set(),
            "values": defaultdict(list),
        })
        if int(meta["label"][index]) != group["label"] or str(meta["ref_7mer"][index]) != group["kmer"]:
            raise RuntimeError(f"Inconsistent site metadata for {site}")
        group["reads"].add(str(meta["fast5_raw_read_id"][index]))
        for model, values in probabilities.items():
            group["values"][model].append(float(values[index]))
    output: dict[str, list[object]] = {"site_pos0": [], "label": [], "ref_7mer": [], "n_windows": [], "n_reads": []}
    for model in probabilities:
        output[model] = []
    for site in sorted(groups):
        group = groups[site]
        output["site_pos0"].append(site)
        output["label"].append(group["label"])
        output["ref_7mer"].append(group["kmer"])
        output["n_windows"].append(len(group["values"][next(iter(probabilities))]))
        output["n_reads"].append(len(group["reads"]))
        for model in probabilities:
            output[model].append(float(np.mean(group["values"][model])))
    return {key: np.asarray(value) for key, value in output.items()}


def grouped_bootstrap(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    groups: np.ndarray,
    thresholds: dict[str, float],
    reps: int,
    seed: int,
) -> list[dict[str, object]]:
    unique, inverse = np.unique(groups.astype(object), return_inverse=True)
    rng = np.random.default_rng(seed)
    model_names = list(probabilities)
    pair_names = [
        (f"{left}_minus_{right}", left, right)
        for left_index, left in enumerate(model_names)
        for right in model_names[left_index + 1:]
    ]
    values = {
        model: defaultdict(list)
        for model in (*model_names, *(name for name, _, _ in pair_names))
    }
    for _ in range(reps):
        sampled_groups = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled_groups, minlength=len(unique))
        repeated = np.repeat(np.arange(labels.size), counts[inverse])
        y = labels[repeated]
        if len(np.unique(y)) < 2:
            continue
        for model in model_names:
            prob = probabilities[model][repeated]
            pred = (prob >= thresholds[model]).astype(np.int8)
            values[model]["auroc"].append(float(roc_auc_score(y, prob)))
            values[model]["balanced_accuracy"].append(float(balanced_accuracy_score(y, pred)))
            values[model]["auprc"].append(float(average_precision_score(y, prob)))
        for delta_name, left, right in pair_names:
            for metric in ("auroc", "balanced_accuracy", "auprc"):
                values[delta_name][metric].append(
                    values[left][metric][-1] - values[right][metric][-1]
                )
    output = []
    for model in (*model_names, *(name for name, _, _ in pair_names)):
        for metric, samples in values[model].items():
            if not samples:
                continue
            output.append({
                "model": model, "metric": metric, "bootstrap_replicates": len(samples),
                "estimate": float(np.mean(samples)),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("S18 training requires physical GPU Device 0 via CUDA_VISIBLE_DEVICES=0")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    patch_torch_optimizer(torch)
    out = Path(args.out_dir)
    embedding_dir = out / "embeddings"
    weight_dir = out / "weights"
    out.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    chr19_meta = load_metadata(out / "chr19_split.parquet")
    chr16_meta = load_metadata(out / "chr16_split.parquet")
    common_kmers = sorted(set(chr19_meta["ref_7mer"].tolist()))
    if common_kmers != sorted(set(chr16_meta["ref_7mer"].tolist())) or len(common_kmers) < 700:
        raise RuntimeError(
            f"chr19/chr16 split manifests do not have the same sufficiently large 7mer set: {len(common_kmers)}"
        )
    kmer_to_id = {kmer: index for index, kmer in enumerate(common_kmers)}
    k19 = np.asarray([kmer_to_id[value] for value in chr19_meta["ref_7mer"]], dtype=np.int32)
    k16 = np.asarray([kmer_to_id[value] for value in chr16_meta["ref_7mer"]], dtype=np.int32)
    test16 = np.flatnonzero(chr16_meta["role"] == "test")
    baseline16 = np.flatnonzero(chr16_meta["role"] == "baseline")
    if not np.all(chr16_meta["label"][baseline16] == 0):
        raise RuntimeError("chr16 baseline contains non-negative labels")
    if set(chr16_meta["fast5_raw_read_id"][baseline16]) & set(chr16_meta["fast5_raw_read_id"][test16]):
        raise RuntimeError("chr16 baseline/test read leakage")

    metrics: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    test_scores: dict[str, dict[str, np.ndarray]] = {}
    oof_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    started = time.time()

    for model_name, model_key in (
        # ("V600_Apple", "v600"),
        # ("V610_Apple", "v610"),
        ("V003_Stone", "v007"),
    ):
        # emb19 = load_embedding(embedding_dir / f"chr19_{model_key}_ode_l2_s2_t098_full.npy", chr19_meta["label"].size)
        # emb16 = load_embedding(embedding_dir / f"chr16_{model_key}_ode_l2_s2_t098_full.npy", chr16_meta["label"].size)
        emb19 = load_embedding(embedding_dir / f"chr19_{model_key}_full.npy", chr19_meta["label"].size)
        emb16 = load_embedding(embedding_dir / f"chr16_{model_key}_full.npy", chr16_meta["label"].size)
        oof_logits = np.full(chr19_meta["label"].size, np.nan, dtype=np.float32)
        fold_test_logits: list[np.ndarray] = []
        fold_no_target_logits: list[np.ndarray] = []
        for fold in range(5):
            train_idx = np.flatnonzero(chr19_meta["fold"] != fold)
            val_idx = np.flatnonzero(chr19_meta["fold"] == fold)
            neg_train = train_idx[chr19_meta["label"][train_idx] == 0]
            centers19 = robust_centers(emb19, neg_train, k19, len(common_kmers))
            whiten19 = fit_residual_whitener(
                emb19,
                neg_train,
                k19,
                centers19,
                args.whiten_eps,
                args.whiten_shrinkage,
                args.whiten_components,
            )
            X_train = make_features(emb19, train_idx, k19, centers19, whiten19["whitener"])
            X_val = make_features(emb19, val_idx, k19, centers19, whiten19["whitener"])
            head, val_logits, best_epoch, best_auc, history = train_head(
                X_train, chr19_meta["label"][train_idx], X_val,
                chr19_meta["label"][val_idx], args, fold, model_name,
                torch.device(args.device), torch,
            )
            oof_logits[val_idx] = val_logits
            histories.extend(history)
            torch.save({"state_dict": head.state_dict(), "model": model_name, "fold": fold}, weight_dir / f"{model_key}_fold{fold}.pt")

            centers16 = robust_centers(emb16, baseline16, k16, len(common_kmers))
            X_test = make_features(emb16, test16, k16, centers16, whiten19["whitener"])
            X_no_target = make_features(emb16, test16, k16, centers19, whiten19["whitener"])
            with torch.inference_mode():
                fold_test_logits.append(head(torch.from_numpy(X_test).to(args.device)).squeeze(-1).float().cpu().numpy())
                fold_no_target_logits.append(head(torch.from_numpy(X_no_target).to(args.device)).squeeze(-1).float().cpu().numpy())
            np.savez_compressed(
                weight_dir / f"{model_key}_fold{fold}_baseline.npz",
                centers=centers19,
                whitener=whiten19["whitener"],
                whiten_eigvals=whiten19["eigvals"],
                whiten_components=whiten19["components"],
                target_centers=centers16,
                kmers=np.asarray(common_kmers),
            )
            fold_metrics.append({"model": model_name, "fold": fold, "best_epoch": best_epoch, "val_auroc": best_auc, "train_rows": train_idx.size, "val_rows": val_idx.size})
            del X_train, X_val, X_test, X_no_target, head, centers19, centers16, whiten19
            gc.collect()
            torch.cuda.empty_cache()

        del emb19, emb16
        oof_prob = sigmoid(oof_logits)
        threshold = choose_threshold(chr19_meta["label"], oof_prob)
        thresholds[model_name] = threshold
        test_logits = np.mean(np.stack(fold_test_logits), axis=0)
        no_target_logits = np.mean(np.stack(fold_no_target_logits), axis=0)
        test_scores[model_name] = {
            "prob": sigmoid(test_logits),
            "no_target_prob": sigmoid(no_target_logits),
        }
        oof_scores[model_name] = oof_prob
        metrics.append(calculate_metrics("chr19_oof", model_name, chr19_meta["label"], oof_prob, threshold, len(set(chr19_meta["fast5_raw_read_id"]))))
        metrics.append(calculate_metrics("chr16_window", model_name, chr16_meta["label"][test16], test_scores[model_name]["prob"], threshold, len(set(chr16_meta["fast5_raw_read_id"][test16]))))
        metrics.append(calculate_metrics("chr16_window_no_target_baseline", model_name, chr16_meta["label"][test16], test_scores[model_name]["no_target_prob"], threshold, len(set(chr16_meta["fast5_raw_read_id"][test16]))))

    # Sequence-only control: the common 7mer composition is balanced by construction.
    train_kmer_rate = np.zeros(len(common_kmers), dtype=np.float64)
    for kmer_id in range(len(common_kmers)):
        selected = k19 == kmer_id
        train_kmer_rate[kmer_id] = chr19_meta["label"][selected].mean()
    kmer_prob = train_kmer_rate[k16[test16]]
    metrics.append(calculate_metrics("chr16_window", "7mer_only_control", chr16_meta["label"][test16], kmer_prob, 0.5, len(set(chr16_meta["fast5_raw_read_id"][test16]))))

    # Wide window-level predictions for R and downstream analyses.
    prediction_rows = []
    test_labels = chr16_meta["label"][test16]
    for local_index, row_index in enumerate(test16):
        row = {
            "row_index": int(row_index), "window_id": str(chr16_meta["window_id"][row_index]),
            "site_pos0": int(chr16_meta["site_pos0"][row_index]), "label": int(test_labels[local_index]),
            "ref_7mer": str(chr16_meta["ref_7mer"][row_index]),
            "fast5_raw_read_id": str(chr16_meta["fast5_raw_read_id"][row_index]),
            "kmer_only_prob": float(kmer_prob[local_index]),
        }
        for model_name in MODELS:
            row[f"{model_name}_prob"] = float(test_scores[model_name]["prob"][local_index])
            row[f"{model_name}_no_target_prob"] = float(test_scores[model_name]["no_target_prob"][local_index])
        prediction_rows.append(row)
    pq.write_table(pa.Table.from_pylist(prediction_rows), out / "chr16_window_predictions.parquet", compression="zstd", compression_level=6, use_dictionary=True)
    write_csv(out / "chr16_window_predictions.csv", prediction_rows)

    site_probabilities = {model: test_scores[model]["prob"] for model in MODELS}
    site_values = aggregate_sites({key: value[test16] if value.shape[0] == chr16_meta["label"].size else value for key, value in chr16_meta.items() if key in ("site_pos0", "label", "ref_7mer", "fast5_raw_read_id")}, site_probabilities)
    site_rows = []
    for index in range(site_values["site_pos0"].size):
        row = {key: (value[index].item() if hasattr(value[index], "item") else value[index]) for key, value in site_values.items()}
        row["ref_7mer"] = str(row["ref_7mer"])
        site_rows.append(row)
    pq.write_table(pa.Table.from_pylist(site_rows), out / "chr16_site_predictions.parquet", compression="zstd", compression_level=6, use_dictionary=True)
    write_csv(out / "chr16_site_predictions.csv", site_rows)
    site_labels = np.asarray(site_values["label"], dtype=np.int8)
    site_read_counts = np.asarray(site_values["n_reads"], dtype=np.int32)
    for model_name in MODELS:
        metrics.append(calculate_metrics("chr16_site_all", model_name, site_labels, np.asarray(site_values[model_name]), thresholds[model_name], int(site_read_counts.sum())))
        keep = site_read_counts >= 2
        metrics.append(calculate_metrics("chr16_site_coverage_ge2", model_name, site_labels[keep], np.asarray(site_values[model_name])[keep], thresholds[model_name], int(site_read_counts[keep].sum())))

    bootstrap_rows = []
    test_groups = chr16_meta["fast5_raw_read_id"][test16]
    bootstrap_rows.extend(grouped_bootstrap(test_labels, {model: test_scores[model]["prob"] for model in MODELS}, test_groups, thresholds, args.bootstrap, 1729))
    site_groups = site_values["site_pos0"]
    bootstrap_rows.extend(grouped_bootstrap(site_labels, {model: np.asarray(site_values[model]) for model in MODELS}, site_groups, thresholds, args.bootstrap, 1730))

    write_csv(out / "metrics.csv", metrics)
    write_csv(out / "fold_metrics.csv", fold_metrics)
    write_csv(out / "training_history.csv", histories)
    write_csv(out / "bootstrap_ci.csv", bootstrap_rows)
    summary = {
        "models": list(MODELS), "common_kmers": len(common_kmers),
        "chr19_rows": int(chr19_meta["label"].size), "chr16_baseline_rows": int(baseline16.size),
        "chr16_test_rows": int(test16.size), "chr16_test_sites": int(site_labels.size),
        "chr16_test_sites_coverage_ge2": int((site_read_counts >= 2).sum()),
        "thresholds": thresholds, "device": "physical GPU 0 via CUDA_VISIBLE_DEVICES=0",
        "training": {
            "backbone": "frozen",
            "feature": "PCA-whitened residual z = (embedding - per-7mer unmodified center) @ topK_eigvecs @ diag(1/sqrt(eigvals)), fitted on train negative residuals with covariance shrinkage and eigenvalue floor",
            "head": "MLPPlusCosinePCAWhitenedResidualHead: MLP([z_pca_white, abs(z_pca_white)]) + sigmoid(alpha) * cosine_logit(normalize(z_pca_white)); learned 2-class cosine directions and scale",
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "mix_logit_init": args.mix_logit_init,
            "whiten_eps": args.whiten_eps,
            "whiten_shrinkage": args.whiten_shrinkage,
            "whiten_components": args.whiten_components,
        },
        "v003": "Stone tokens, hf_dlm ode_hidden_state, ode_steps=2, ode_start_t=0.98, self_cond_cfg_scale=0.5",
        "v610": "Apple tokens, OLMo2 base last hidden state",
        "seconds": time.time() - started,
    }
    summary["v600"] = "Apple tokens, HF_RSQ742C12A511_DNAOLMO_V600 base last hidden state"
    summary["v610"] = "Apple tokens, HF_RSQ741C12V523_MIXOLMO_V610 base last hidden state"
    (out / "s18_training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
