#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


DEFAULT_ROOT = Path("/Users/kexuanzhou/project/PoreDLM/data/eval-data/LB06/after-filter")
DEFAULT_MODEL_ROOT = Path("/mnt/zzbnew/poregpt/models/HF_VQE768C08A001_DNADLLM_V001")


@dataclass
class ReadItem:
    site: str
    row_index: int
    read_id: str
    label: str
    motif: str
    ref_start1: int | None
    ref_end1: int | None
    signal: np.ndarray


def add_repo_imports() -> None:
    script = Path(__file__).resolve()
    for parent in [script.parent, *script.parents]:
        if (parent / "src" / "poredlm").exists():
            repo = parent
            break
    else:
        repo = Path.cwd()

    candidates = [
        repo / "src",
        repo / "src" / "poredlm",
        repo / "src" / "poredlm" / "data" / "stage2_BERT_Encoder",
        repo / "src" / "poredlm" / "training_public" / "stage2_BERT_trian" / "token_dataset",
    ]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_reads(input_dir: Path, max_reads_per_site: int | None) -> Iterable[ReadItem]:
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        site = jsonl_path.stem
        kept = 0
        for row_index, obj in enumerate(iter_jsonl(jsonl_path)):
            signal = obj.get("signal")
            if not isinstance(signal, list) or not signal:
                continue
            yield ReadItem(
                site=site,
                row_index=row_index,
                read_id=str(obj.get("read_id") or obj.get("id") or f"{site}:{row_index}"),
                label=str(obj.get("label") or ""),
                motif=str(obj.get("motif") or obj.get("ref_motif") or ""),
                ref_start1=obj.get("ref_start1"),
                ref_end1=obj.get("ref_end1"),
                signal=np.asarray(signal, dtype=np.float32),
            )
            kept += 1
            if max_reads_per_site is not None and kept >= max_reads_per_site:
                break


def pad_signal_batch(signals: list[np.ndarray]) -> torch.Tensor:
    max_len = max(int(x.size) for x in signals)
    batch = torch.zeros((len(signals), max_len), dtype=torch.float32)
    for i, signal in enumerate(signals):
        batch[i, : signal.size] = torch.from_numpy(signal)
    return batch


class HFCodecTokenizer:
    def __init__(self, model_dir: Path, device: torch.device, token_offset: int):
        import modeling_pore_vq_codec  # noqa: F401
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(device).eval()
        self.device = device
        self.token_offset = int(token_offset)

    @torch.inference_mode()
    def encode_batch(self, signals: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
        signal_t = pad_signal_batch(signals).to(self.device)
        if hasattr(self.model, "encode_signal"):
            indices = self.model.encode_signal(signal_t)
        else:
            output = self.model(signal_t)
            indices = output[1] if isinstance(output, tuple) else output["indices"]
        indices = indices.detach().cpu().long().numpy()
        token_ids = [row.astype(np.int64) + self.token_offset for row in indices]
        valid_lengths = [max(1, int(math.ceil(signal.size / 5))) for signal in signals]
        valid_lengths = [min(length, token_ids[i].shape[0]) for i, length in enumerate(valid_lengths)]
        return token_ids, valid_lengths


class LegacyVQETokenizer:
    def __init__(self, ckpt: Path, device: torch.device, token_offset: int):
        from vqe_tokenizer import VQETokenizer

        self.tokenizer = VQETokenizer(model_ckpt=str(ckpt), device=str(device))
        self.token_offset = int(token_offset)

    def encode_batch(self, signals: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
        token_ids = []
        valid_lengths = []
        for signal in signals:
            ids = np.asarray(self.tokenizer._tokenize_chunked_signal(signal), dtype=np.int64)
            token_ids.append(ids + self.token_offset)
            valid_lengths.append(int(ids.size))
        return token_ids, valid_lengths


def build_tokenizer(args: argparse.Namespace, device: torch.device):
    if args.tokenizer_backend == "legacy-vqe":
        return LegacyVQETokenizer(Path(args.encoder), device, args.token_offset)
    return HFCodecTokenizer(Path(args.encoder), device, args.token_offset)


def make_token_batch(
    token_ids: list[np.ndarray],
    valid_lengths: list[int],
    pad_token_id: int,
    max_tokens: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    target_len = max(int(x.size) for x in token_ids)
    if max_tokens is not None:
        target_len = min(target_len, int(max_tokens))
    input_ids = torch.full((len(token_ids), target_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(token_ids), target_len), dtype=torch.long)
    effective_lengths = []
    for i, ids in enumerate(token_ids):
        keep = min(int(ids.size), target_len)
        valid = min(int(valid_lengths[i]), keep)
        if keep:
            input_ids[i, :keep] = torch.from_numpy(ids[:keep]).long()
        if valid:
            attention_mask[i, :valid] = 1
        effective_lengths.append(valid)
    return input_ids.to(device), attention_mask.to(device), effective_lengths


def load_dlm(model_dir: Path, device: torch.device, dtype: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(device).eval()
    if dtype == "float16":
        model = model.half()
    elif dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    elif dtype == "float32":
        model = model.float()
    return model, tokenizer


def get_context_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "context_encoder"):
        out = model.context_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and isinstance(out, dict):
            hidden = out.get("last_hidden_state")
        if hidden is not None:
            return hidden
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_context=True, return_dict=True)
    hidden = out.get("context_hidden_state") if isinstance(out, dict) else getattr(out, "context_hidden_state", None)
    if hidden is None:
        hidden = out.get("last_hidden_state") if isinstance(out, dict) else getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise ValueError("Cannot find context hidden state from hf_dlm output.")
    return hidden


def elf_t_eps(model) -> float:
    cfg = getattr(model, "config", None)
    dlm_cfg = getattr(cfg, "dlm_config", None) or {}
    return float(dlm_cfg.get("t_eps", 0.05))


def elf_net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor, t_eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.view(-1, 1, 1)
    x_pred = net_out
    v_pred = (x_pred - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
    return v_pred, x_pred


@torch.inference_mode()
def ode_from_context_hidden(
    model,
    context: torch.Tensor,
    attention_mask: torch.Tensor,
    ode_steps: int,
    ode_start_t: float,
    self_cond_cfg_scale: float,
) -> torch.Tensor:
    elf = getattr(model, "elf_denoiser", None)
    if elf is None:
        raise ValueError("hf_dlm does not expose elf_denoiser; cannot compute ode_hidden.")
    context = context.to(dtype=next(elf.parameters()).dtype)
    z = context
    x_pred = torch.zeros_like(z)
    t_steps = torch.linspace(float(ode_start_t), 1.0, int(ode_steps) + 1, device=z.device, dtype=z.dtype)

    for idx in range(int(ode_steps)):
        t = t_steps[idx]
        t_next = t_steps[idx + 1]
        t_batch = torch.full((z.shape[0],), float(t.detach().item()), device=z.device, dtype=z.dtype)
        num_self_cond = int(getattr(elf, "num_self_cond_cfg_tokens", 0))
        if num_self_cond > 0:
            model_input = torch.cat([z, x_pred], dim=-1)
            cfg_scale = torch.full((z.shape[0],), float(self_cond_cfg_scale), device=z.device, dtype=z.dtype)
            net_out = elf(
                model_input,
                t_batch,
                attention_mask=attention_mask,
                self_cond_cfg_scale=cfg_scale,
                decoder_step_active=False,
            )
        else:
            net_out = elf(z, t_batch, attention_mask=attention_mask, decoder_step_active=False)
        v_pred, x_pred = elf_net_out_to_v_x(net_out, z, t_batch, elf_t_eps(model))
        z = z + (t_next - t) * v_pred
        valid_mask = attention_mask.to(device=z.device, dtype=torch.bool).unsqueeze(-1)
        z = torch.where(valid_mask, z, context)
        x_pred = torch.where(valid_mask, x_pred, context)
    return z


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (summed / denom).detach().float().cpu().numpy()


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[0], 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    if coords.shape[1] == 1:
        coords = np.pad(coords, ((0, 0), (0, 1)))
    denom = np.maximum((s**2).sum(), 1e-12)
    explained = (s[:2] ** 2 / denom).astype(np.float32)
    if explained.size == 1:
        explained = np.asarray([explained[0], 0.0], dtype=np.float32)
    return coords[:, :2].astype(np.float32), explained


def cluster_metrics(x: np.ndarray, max_pairwise_pairs: int = 200_000) -> dict:
    if x.shape[0] == 0:
        return {
            "n": 0,
            "centroid_mean_l2": float("nan"),
            "centroid_median_l2": float("nan"),
            "centroid_rms_l2": float("nan"),
            "pairwise_mean_l2": float("nan"),
            "pairwise_median_l2": float("nan"),
            "pairwise_mean_cosine": float("nan"),
        }

    centered = x - x.mean(axis=0, keepdims=True)
    centroid_dist = np.linalg.norm(centered, axis=1)
    metrics = {
        "n": int(x.shape[0]),
        "centroid_mean_l2": float(np.mean(centroid_dist)),
        "centroid_median_l2": float(np.median(centroid_dist)),
        "centroid_rms_l2": float(np.sqrt(np.mean(centroid_dist**2))),
        "pairwise_mean_l2": float("nan"),
        "pairwise_median_l2": float("nan"),
        "pairwise_mean_cosine": float("nan"),
    }
    if x.shape[0] < 2:
        return metrics

    n_pairs = x.shape[0] * (x.shape[0] - 1) // 2
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norms, 1e-12)
    if n_pairs <= max_pairwise_pairs:
        upper = np.triu_indices(x.shape[0], k=1)
        upper_l2 = np.linalg.norm(x[upper[0]] - x[upper[1]], axis=1)
        upper_cosine = 1.0 - np.sum(normalized[upper[0]] * normalized[upper[1]], axis=1)
    else:
        rng = np.random.default_rng(20260905)
        left = rng.integers(0, x.shape[0], size=max_pairwise_pairs, endpoint=False)
        right = rng.integers(0, x.shape[0] - 1, size=max_pairwise_pairs, endpoint=False)
        right = right + (right >= left)
        upper_l2 = np.linalg.norm(x[left] - x[right], axis=1)
        upper_cosine = 1.0 - np.sum(normalized[left] * normalized[right], axis=1)
    metrics["pairwise_mean_l2"] = float(np.mean(upper_l2))
    metrics["pairwise_median_l2"] = float(np.median(upper_l2))
    metrics["pairwise_mean_cosine"] = float(np.mean(upper_cosine))
    return metrics


def metric_ratio(ode_value: float, context_value: float) -> float:
    if not np.isfinite(ode_value) or not np.isfinite(context_value) or abs(context_value) < 1e-12:
        return float("nan")
    return float(ode_value / context_value)


def site_separation_metrics(vectors: np.ndarray, sites: list[str]) -> dict:
    unique_sites = sorted(set(sites))
    if not unique_sites:
        return {}

    centers = []
    within_rms = []
    within_pairwise = []
    site_counts = []
    for site in unique_sites:
        idx = np.asarray([i for i, value in enumerate(sites) if value == site], dtype=np.int64)
        site_vectors = vectors[idx]
        site_metrics = cluster_metrics(site_vectors)
        centers.append(site_vectors.mean(axis=0))
        within_rms.append(site_metrics["centroid_rms_l2"])
        within_pairwise.append(site_metrics["pairwise_mean_l2"])
        site_counts.append(int(idx.size))

    centers_arr = np.stack(centers, axis=0)
    within_rms_arr = np.asarray(within_rms, dtype=np.float64)
    within_pairwise_arr = np.asarray(within_pairwise, dtype=np.float64)
    counts_arr = np.asarray(site_counts, dtype=np.float64)
    weights = counts_arr / max(float(counts_arr.sum()), 1.0)

    result = {
        "n_sites": int(len(unique_sites)),
        "within_site_rms_mean": float(np.nanmean(within_rms_arr)),
        "within_site_rms_weighted_mean": float(np.nansum(within_rms_arr * weights)),
        "within_site_pairwise_mean": float(np.nanmean(within_pairwise_arr)),
        "within_site_pairwise_weighted_mean": float(np.nansum(within_pairwise_arr * weights)),
        "between_site_center_l2_mean": float("nan"),
        "between_site_center_l2_median": float("nan"),
        "between_site_center_l2_min": float("nan"),
        "between_site_center_cosine_mean": float("nan"),
        "separation_over_within_rms": float("nan"),
        "min_separation_over_within_rms": float("nan"),
        "separation_over_within_pairwise": float("nan"),
    }
    if centers_arr.shape[0] < 2:
        return result

    upper = np.triu_indices(centers_arr.shape[0], k=1)
    center_l2 = np.linalg.norm(centers_arr[upper[0]] - centers_arr[upper[1]], axis=1)
    center_norms = np.linalg.norm(centers_arr, axis=1, keepdims=True)
    center_normalized = centers_arr / np.maximum(center_norms, 1e-12)
    center_cosine = 1.0 - np.sum(center_normalized[upper[0]] * center_normalized[upper[1]], axis=1)

    mean_within_rms = result["within_site_rms_weighted_mean"]
    mean_within_pairwise = result["within_site_pairwise_weighted_mean"]
    result.update(
        {
            "between_site_center_l2_mean": float(np.mean(center_l2)),
            "between_site_center_l2_median": float(np.median(center_l2)),
            "between_site_center_l2_min": float(np.min(center_l2)),
            "between_site_center_cosine_mean": float(np.mean(center_cosine)),
            "separation_over_within_rms": metric_ratio(float(np.mean(center_l2)), mean_within_rms),
            "min_separation_over_within_rms": metric_ratio(float(np.min(center_l2)), mean_within_rms),
            "separation_over_within_pairwise": metric_ratio(float(np.mean(center_l2)), mean_within_pairwise),
        }
    )
    return result


def write_global_separation_metrics(path: Path, rows: list[dict]) -> None:
    fields = [
        "representation",
        "n_sites",
        "within_site_rms_mean",
        "within_site_rms_weighted_mean",
        "within_site_pairwise_mean",
        "within_site_pairwise_weighted_mean",
        "between_site_center_l2_mean",
        "between_site_center_l2_median",
        "between_site_center_l2_min",
        "between_site_center_cosine_mean",
        "separation_over_within_rms",
        "min_separation_over_within_rms",
        "separation_over_within_pairwise",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def save_plot(coords: np.ndarray, sites: list[str], title: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    unique_sites = sorted(set(sites))
    cmap = plt.get_cmap("tab20", max(1, len(unique_sites)))
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
    colors = [cmap(site_to_idx[site]) for site in sites]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=14, alpha=0.78, linewidths=0)
    for site in unique_sites:
        idx = [i for i, value in enumerate(sites) if value == site]
        if not idx:
            continue
        center = coords[idx].mean(axis=0)
        ax.text(center[0], center[1], site.split("__")[0], fontsize=7, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_global_comparison_plot(
    context_vectors: np.ndarray,
    ode_vectors: np.ndarray,
    sites: list[str],
    out_png: Path,
) -> dict:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    combined = np.concatenate([context_vectors, ode_vectors], axis=0)
    combined_xy, explained = pca_2d(combined)
    n = context_vectors.shape[0]
    context_xy = combined_xy[:n]
    ode_xy = combined_xy[n:]

    unique_sites = sorted(set(sites))
    cmap = plt.get_cmap("tab20", max(1, len(unique_sites)))
    site_to_idx = {site: idx for idx, site in enumerate(unique_sites)}
    colors = [cmap(site_to_idx[site]) for site in sites]

    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=170)
    ax.scatter(
        context_xy[:, 0],
        context_xy[:, 1],
        c=colors,
        s=13,
        alpha=0.58,
        linewidths=0,
        marker="o",
    )
    ax.scatter(
        ode_xy[:, 0],
        ode_xy[:, 1],
        c=colors,
        s=18,
        alpha=0.72,
        linewidths=0,
        marker="^",
    )
    for site in unique_sites:
        idx = np.asarray([i for i, value in enumerate(sites) if value == site], dtype=np.int64)
        if idx.size == 0:
            continue
        context_center = context_xy[idx].mean(axis=0)
        ode_center = ode_xy[idx].mean(axis=0)
        color = cmap(site_to_idx[site])
        ax.plot(
            [context_center[0], ode_center[0]],
            [context_center[1], ode_center[1]],
            color=color,
            alpha=0.7,
            linewidth=1.0,
        )
        ax.scatter(context_center[0], context_center[1], s=62, marker="x", color=color, linewidths=1.4)
        ax.scatter(ode_center[0], ode_center[1], s=54, marker="+", color=color, linewidths=1.4)
        ax.text(
            ode_center[0],
            ode_center[1],
            site.split("__")[0],
            fontsize=7,
            ha="left",
            va="bottom",
            color=color,
        )

    marker_handles = [
        mlines.Line2D([], [], color="#555555", marker="o", linestyle="None", markersize=5, label="context_hidden"),
        mlines.Line2D([], [], color="#555555", marker="^", linestyle="None", markersize=6, label="ode_hidden"),
        mlines.Line2D([], [], color="#555555", marker="x", linestyle="None", markersize=6, label="context center"),
        mlines.Line2D([], [], color="#555555", marker="+", linestyle="None", markersize=7, label="ode center"),
    ]
    ax.legend(handles=marker_handles, frameon=False, loc="best")
    ax.set_title("LB06 all same-site reads: context_hidden and ode_hidden")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return {
        "global_joint_pca_explained_pc1": float(explained[0]),
        "global_joint_pca_explained_pc2": float(explained[1]),
    }


def save_site_comparison_plot(
    context_vectors: np.ndarray,
    ode_vectors: np.ndarray,
    site: str,
    out_png: Path,
) -> dict:
    import matplotlib.pyplot as plt

    context_metrics = cluster_metrics(context_vectors)
    ode_metrics = cluster_metrics(ode_vectors)
    rms_ratio = metric_ratio(ode_metrics["centroid_rms_l2"], context_metrics["centroid_rms_l2"])
    pairwise_ratio = metric_ratio(ode_metrics["pairwise_mean_l2"], context_metrics["pairwise_mean_l2"])
    cosine_ratio = metric_ratio(ode_metrics["pairwise_mean_cosine"], context_metrics["pairwise_mean_cosine"])

    combined = np.concatenate([context_vectors, ode_vectors], axis=0)
    combined_xy, explained = pca_2d(combined)
    n = context_vectors.shape[0]
    context_xy = combined_xy[:n]
    ode_xy = combined_xy[n:]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=170)
    ax.scatter(
        context_xy[:, 0],
        context_xy[:, 1],
        s=18,
        alpha=0.76,
        linewidths=0,
        color="#2f6fdd",
        label="context_hidden",
    )
    ax.scatter(
        ode_xy[:, 0],
        ode_xy[:, 1],
        s=20,
        alpha=0.78,
        linewidths=0,
        marker="^",
        color="#d44a2f",
        label="ode_hidden",
    )
    for a, b in zip(context_xy, ode_xy):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#777777", alpha=0.18, linewidth=0.6)

    ax.scatter(context_xy[:, 0].mean(), context_xy[:, 1].mean(), s=90, marker="x", color="#194b9f")
    ax.scatter(ode_xy[:, 0].mean(), ode_xy[:, 1].mean(), s=90, marker="x", color="#9f2b19")
    ax.set_title(f"{site}: context_hidden vs ode_hidden")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.legend(frameon=False)
    metrics_text = (
        "Compactness in original hidden space\n"
        f"context RMS radius: {context_metrics['centroid_rms_l2']:.4g}\n"
        f"ode RMS radius: {ode_metrics['centroid_rms_l2']:.4g}\n"
        f"ODE/context RMS: {rms_ratio:.3g}\n"
        f"context mean pairwise L2: {context_metrics['pairwise_mean_l2']:.4g}\n"
        f"ode mean pairwise L2: {ode_metrics['pairwise_mean_l2']:.4g}\n"
        f"ODE/context pairwise: {pairwise_ratio:.3g}\n"
        f"ODE/context cosine: {cosine_ratio:.3g}"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.86,
            "linewidth": 0.7,
        },
    )
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return {
        "pca_explained_pc1": float(explained[0]),
        "pca_explained_pc2": float(explained[1]),
    }


def write_site_metrics(path: Path, metrics_rows: list[dict]) -> None:
    fields = [
        "site",
        "n_reads",
        "context_centroid_mean_l2",
        "ode_centroid_mean_l2",
        "ratio_centroid_mean_l2",
        "context_centroid_median_l2",
        "ode_centroid_median_l2",
        "ratio_centroid_median_l2",
        "context_centroid_rms_l2",
        "ode_centroid_rms_l2",
        "ratio_centroid_rms_l2",
        "context_pairwise_mean_l2",
        "ode_pairwise_mean_l2",
        "ratio_pairwise_mean_l2",
        "context_pairwise_median_l2",
        "ode_pairwise_median_l2",
        "ratio_pairwise_median_l2",
        "context_pairwise_mean_cosine",
        "ode_pairwise_mean_cosine",
        "ratio_pairwise_mean_cosine",
        "pca_explained_pc1",
        "pca_explained_pc2",
        "ode_more_compact_by_centroid_rms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_metadata(path: Path, rows: list[ReadItem], valid_token_lengths: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "site", "row_index", "read_id", "label", "motif", "ref_start1", "ref_end1", "valid_tokens"])
        for i, item in enumerate(rows):
            writer.writerow([
                i,
                item.site,
                item.row_index,
                item.read_id,
                item.label,
                item.motif,
                item.ref_start1,
                item.ref_end1,
                valid_token_lengths[i],
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize same-reference-site reads, extract context_hidden/ode_hidden, and plot read-level feature space."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_ROOT), help="Directory containing after-filter *.jsonl files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_ROOT.parent / "embedding-space"), help="Output directory.")
    parser.add_argument("--encoder", default=str(DEFAULT_MODEL_ROOT / "encoder"), help="HF codec encoder directory, or legacy VQE .pth with --tokenizer-backend legacy-vqe.")
    parser.add_argument("--hf-dlm", default=str(DEFAULT_MODEL_ROOT / "hf_dlm"), help="HF DLM model directory.")
    parser.add_argument("--tokenizer-backend", choices=("hf-codec", "legacy-vqe"), default="hf-codec")
    parser.add_argument("--token-offset", type=int, default=128, help="Offset from codec indices to hf_dlm bwav token ids.")
    parser.add_argument("--pad-token-id", type=int, default=None, help="Defaults to hf_dlm tokenizer.pad_token_id, then 1.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional truncation length after tokenization.")
    parser.add_argument("--max-reads-per-site", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--ode-start-t", type=float, default=0.85)
    parser.add_argument("--self-cond-cfg-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_repo_imports()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[setup] input_dir={input_dir}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] device={device}")

    dlm, hf_tokenizer = load_dlm(Path(args.hf_dlm), device, args.dtype)
    pad_token_id = args.pad_token_id
    if pad_token_id is None:
        pad_token_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 1
    codec = build_tokenizer(args, device)

    all_rows: list[ReadItem] = []
    all_valid_lengths: list[int] = []
    context_vectors: list[np.ndarray] = []
    ode_vectors: list[np.ndarray] = []

    batch: list[ReadItem] = []
    for item in iter_reads(input_dir, args.max_reads_per_site):
        batch.append(item)
        if len(batch) < args.batch_size:
            continue
        token_ids, valid_lengths = codec.encode_batch([x.signal for x in batch])
        input_ids, attention_mask, effective_lengths = make_token_batch(token_ids, valid_lengths, pad_token_id, args.max_tokens, device)
        with torch.inference_mode():
            context = get_context_hidden(dlm, input_ids, attention_mask)
            ode = ode_from_context_hidden(dlm, context, attention_mask, args.ode_steps, args.ode_start_t, args.self_cond_cfg_scale)
        context_vectors.append(mean_pool(context, attention_mask))
        ode_vectors.append(mean_pool(ode, attention_mask))
        all_rows.extend(batch)
        all_valid_lengths.extend(effective_lengths)
        print(f"[batch] processed reads={len(all_rows)}", flush=True)
        batch = []

    if batch:
        token_ids, valid_lengths = codec.encode_batch([x.signal for x in batch])
        input_ids, attention_mask, effective_lengths = make_token_batch(token_ids, valid_lengths, pad_token_id, args.max_tokens, device)
        with torch.inference_mode():
            context = get_context_hidden(dlm, input_ids, attention_mask)
            ode = ode_from_context_hidden(dlm, context, attention_mask, args.ode_steps, args.ode_start_t, args.self_cond_cfg_scale)
        context_vectors.append(mean_pool(context, attention_mask))
        ode_vectors.append(mean_pool(ode, attention_mask))
        all_rows.extend(batch)
        all_valid_lengths.extend(effective_lengths)

    if not all_rows:
        raise SystemExit(f"No valid reads found under {input_dir}")

    context_arr = np.concatenate(context_vectors, axis=0)
    ode_arr = np.concatenate(ode_vectors, axis=0)
    sites = [row.site for row in all_rows]

    context_xy, context_exp = pca_2d(context_arr)
    ode_xy, ode_exp = pca_2d(ode_arr)
    np.savez_compressed(
        output_dir / "same_site_read_embeddings.npz",
        context_hidden=context_arr,
        ode_hidden=ode_arr,
        context_pca=context_xy,
        ode_pca=ode_xy,
        context_pca_explained=context_exp,
        ode_pca_explained=ode_exp,
        sites=np.asarray(sites),
        read_ids=np.asarray([row.read_id for row in all_rows]),
        valid_token_lengths=np.asarray(all_valid_lengths, dtype=np.int32),
    )
    write_metadata(output_dir / "same_site_read_metadata.csv", all_rows, all_valid_lengths)
    save_plot(context_xy, sites, "LB06 same-site reads: context_hidden PCA", output_dir / "context_hidden_pca.png")
    save_plot(ode_xy, sites, "LB06 same-site reads: ode_hidden PCA", output_dir / "ode_hidden_pca.png")
    global_joint_metrics = save_global_comparison_plot(
        context_arr,
        ode_arr,
        sites,
        output_dir / "all_sites_context_vs_ode_pca.png",
    )

    per_site_dir = output_dir / "per_site_plots"
    per_site_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    site_to_indices = {site: [i for i, value in enumerate(sites) if value == site] for site in sorted(set(sites))}
    for site, indices in site_to_indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        site_context = context_arr[idx]
        site_ode = ode_arr[idx]
        context_metrics = cluster_metrics(site_context)
        ode_metrics = cluster_metrics(site_ode)
        plot_metrics = save_site_comparison_plot(
            site_context,
            site_ode,
            site,
            per_site_dir / f"{site}.context_vs_ode_pca.png",
        )
        row = {
            "site": site,
            "n_reads": int(idx.size),
            "context_centroid_mean_l2": context_metrics["centroid_mean_l2"],
            "ode_centroid_mean_l2": ode_metrics["centroid_mean_l2"],
            "ratio_centroid_mean_l2": metric_ratio(ode_metrics["centroid_mean_l2"], context_metrics["centroid_mean_l2"]),
            "context_centroid_median_l2": context_metrics["centroid_median_l2"],
            "ode_centroid_median_l2": ode_metrics["centroid_median_l2"],
            "ratio_centroid_median_l2": metric_ratio(ode_metrics["centroid_median_l2"], context_metrics["centroid_median_l2"]),
            "context_centroid_rms_l2": context_metrics["centroid_rms_l2"],
            "ode_centroid_rms_l2": ode_metrics["centroid_rms_l2"],
            "ratio_centroid_rms_l2": metric_ratio(ode_metrics["centroid_rms_l2"], context_metrics["centroid_rms_l2"]),
            "context_pairwise_mean_l2": context_metrics["pairwise_mean_l2"],
            "ode_pairwise_mean_l2": ode_metrics["pairwise_mean_l2"],
            "ratio_pairwise_mean_l2": metric_ratio(ode_metrics["pairwise_mean_l2"], context_metrics["pairwise_mean_l2"]),
            "context_pairwise_median_l2": context_metrics["pairwise_median_l2"],
            "ode_pairwise_median_l2": ode_metrics["pairwise_median_l2"],
            "ratio_pairwise_median_l2": metric_ratio(ode_metrics["pairwise_median_l2"], context_metrics["pairwise_median_l2"]),
            "context_pairwise_mean_cosine": context_metrics["pairwise_mean_cosine"],
            "ode_pairwise_mean_cosine": ode_metrics["pairwise_mean_cosine"],
            "ratio_pairwise_mean_cosine": metric_ratio(ode_metrics["pairwise_mean_cosine"], context_metrics["pairwise_mean_cosine"]),
            "ode_more_compact_by_centroid_rms": bool(
                np.isfinite(context_metrics["centroid_rms_l2"])
                and np.isfinite(ode_metrics["centroid_rms_l2"])
                and ode_metrics["centroid_rms_l2"] < context_metrics["centroid_rms_l2"]
            ),
            **plot_metrics,
        }
        metrics_rows.append(row)

    write_site_metrics(output_dir / "per_site_compactness_metrics.csv", metrics_rows)
    (output_dir / "per_site_compactness_metrics.json").write_text(
        json.dumps(metrics_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    context_separation = {
        "representation": "context_hidden",
        **site_separation_metrics(context_arr, sites),
    }
    ode_separation = {
        "representation": "ode_hidden",
        **site_separation_metrics(ode_arr, sites),
    }
    separation_comparison = {
        "within_site_rms_ratio_ode_over_context": metric_ratio(
            ode_separation["within_site_rms_weighted_mean"],
            context_separation["within_site_rms_weighted_mean"],
        ),
        "within_site_pairwise_ratio_ode_over_context": metric_ratio(
            ode_separation["within_site_pairwise_weighted_mean"],
            context_separation["within_site_pairwise_weighted_mean"],
        ),
        "between_site_center_l2_ratio_ode_over_context": metric_ratio(
            ode_separation["between_site_center_l2_mean"],
            context_separation["between_site_center_l2_mean"],
        ),
        "min_between_site_center_l2_ratio_ode_over_context": metric_ratio(
            ode_separation["between_site_center_l2_min"],
            context_separation["between_site_center_l2_min"],
        ),
        "separation_over_within_rms_ratio_ode_over_context": metric_ratio(
            ode_separation["separation_over_within_rms"],
            context_separation["separation_over_within_rms"],
        ),
        "min_separation_over_within_rms_ratio_ode_over_context": metric_ratio(
            ode_separation["min_separation_over_within_rms"],
            context_separation["min_separation_over_within_rms"],
        ),
    }
    global_separation_rows = [context_separation, ode_separation]
    write_global_separation_metrics(output_dir / "global_site_separation_metrics.csv", global_separation_rows)
    (output_dir / "global_site_separation_metrics.json").write_text(
        json.dumps(
            {
                "representations": global_separation_rows,
                "comparison": separation_comparison,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_reads": len(all_rows),
        "n_sites": len(set(sites)),
        "context_hidden_shape": list(context_arr.shape),
        "ode_hidden_shape": list(ode_arr.shape),
        "context_pca_explained": context_exp.tolist(),
        "ode_pca_explained": ode_exp.tolist(),
        "token_offset": args.token_offset,
        "pad_token_id": int(pad_token_id),
        "per_site_plot_dir": str(per_site_dir),
        "per_site_metrics_csv": str(output_dir / "per_site_compactness_metrics.csv"),
        "global_site_separation_metrics_csv": str(output_dir / "global_site_separation_metrics.csv"),
        "global_joint_plot": str(output_dir / "all_sites_context_vs_ode_pca.png"),
        **separation_comparison,
        **global_joint_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
