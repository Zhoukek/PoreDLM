# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union

import difflib
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 服务器/无显示环境
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

from .metrics import parasail_match_vector


def _align_to_match_vector(pred, ref):
    return parasail_match_vector(pred, ref)

def plot_alignment_heatmap(pred_seqs, ref_seqs, max_reads=32, max_len=80):
    n = min(max_reads, len(ref_seqs), len(pred_seqs))

    mats = np.full((n, max_len), np.nan, dtype=float)  # NaN padding
    for i in range(n):
        m = _align_to_match_vector(pred_seqs[i], ref_seqs[i])
        L = min(len(m), max_len)
        mats[i, :L] = m[:L]

    fig, ax = plt.subplots(figsize=(12, max(3, n * 0.25)))

    mm = np.ma.masked_invalid(mats)
    cmap = ListedColormap(["#d62728", "#2ca02c"])  # mismatch=red, match=green
    cmap.set_bad(color="lightgray")  # padding shows as gray

    ax.imshow(mm, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title("Base alignment (match=1, mismatch/delete=0) — aligned on reference")
    ax.set_xlabel("Reference position (truncated)")
    ax.set_ylabel("Reads")
    return fig

# =========================
# indel-aware alignment
# =========================
def align_sequences_indel_aware(true_seq: str, pred_seq: str) -> Tuple[str, str]:
    """
    用 difflib 做编辑对齐：返回长度一致的对齐序列（gap 用 '-'）
    """
    a = true_seq or ""
    b = pred_seq or ""
    sm = difflib.SequenceMatcher(a=a, b=b)

    true_out = []
    pred_out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        A = a[i1:i2]
        B = b[j1:j2]
        if tag == "equal":
            true_out.append(A)
            pred_out.append(B)
        elif tag == "replace":
            L = max(len(A), len(B))
            true_out.append(A.ljust(L, "-"))
            pred_out.append(B.ljust(L, "-"))
        elif tag == "delete":
            true_out.append(A)
            pred_out.append("-" * len(A))
        elif tag == "insert":
            true_out.append("-" * len(B))
            pred_out.append(B)

    return "".join(true_out), "".join(pred_out)


# =========================
# token ids -> bases
# =========================
def _default_id2base() -> Dict[int, str]:
    """
    默认映射（你可以按自己 vocab 改）：
      0: blank(被丢弃)
      1:A  2:T  3:G  4:C
    """
    return {1: "A", 2: "T", 3: "G", 4: "C"}


def ids_to_bases(
    ids: Union[List[int], np.ndarray],
    id2base: Dict[int, str],
    blank_id: int = 0,
) -> str:
    """
    把 token id 序列转成碱基字符串；去掉 blank；未知 token 用 'N'
    """
    if ids is None:
        return ""
    out = []
    for x in ids:
        xi = int(x)
        if xi == blank_id:
            continue
        out.append(id2base.get(xi, "N"))
    return "".join(out)


def normalize_seq_input(
    x: Any,
    id2base: Dict[int, str],
    blank_id: int,
) -> str:
    """
    允许输入：
      - str: "ATGC..."
      - list[int]/np.ndarray: token ids
      - None: ""
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return ids_to_bases(x, id2base=id2base, blank_id=blank_id)
    # 兜底：转字符串（避免崩）
    try:
        return str(x)
    except Exception:
        return ""


# =========================
# plotting helpers
# =========================
def _base_to_id(ch: str) -> int:
    ch = (ch or " ").upper()
    if ch == "A": return 0
    if ch == "T": return 1
    if ch == "G": return 2
    if ch == "C": return 3
    if ch in ["N", "U"]: return 4
    if ch == "-": return 5
    if ch == " ": return 6
    return 7


def plot_aligned_heatmap_png(
    true_seq: str,
    pred_seq: str,
    out_png: str,
    title: str,
    max_len: int = 300,
    dpi: int = 200,
):
    """
    画 3 行色块图：
      TRUE(aln), PRED(aln), MATCH
    pred 为空也允许：会画成全 '-'（gap）
    """
    true_aln, pred_aln = align_sequences_indel_aware(true_seq, pred_seq)

    L = min(len(true_aln), len(pred_aln))
    L = min(L, max_len)
    if L <= 0:
        L = min(max(len(true_seq), len(pred_seq), 1), max_len)
        # 没对齐信息时就直接 padding
        t = (true_seq[:L]).ljust(L, "-")
        p = (pred_seq[:L]).ljust(L, "-")
    else:
        t = (true_aln[:L]).ljust(L, " ")
        p = (pred_aln[:L]).ljust(L, " ")

    true_ids = np.array([_base_to_id(c) for c in t], dtype=np.int32)
    pred_ids = np.array([_base_to_id(c) for c in p], dtype=np.int32)

    # MATCH: 0 mismatch, 1 match, 2 pad, 3 gap
    match = np.zeros((L,), dtype=np.int32)
    for i in range(L):
        if t[i] == " " or p[i] == " ":
            match[i] = 2
        elif t[i] == "-" or p[i] == "-":
            match[i] = 3
        else:
            match[i] = 1 if t[i] == p[i] else 0

    # 合成 3 行；MATCH 偏移到 10..13
    Z = np.zeros((3, L), dtype=np.int32)
    Z[0, :] = true_ids
    Z[1, :] = pred_ids
    Z[2, :] = 10 + match

    # 颜色表（离散）
    color_table = {
        0: "#1f77b4",  # A
        1: "#ff7f0e",  # T
        2: "#2ca02c",  # G
        3: "#d62728",  # C
        4: "#9467bd",  # N/U
        5: "#8c564b",  # gap '-'
        6: "#7f7f7f",  # pad ' '
        7: "#bbbbbb",  # other
        10: "#d62728", # mismatch
        11: "#2ca02c", # match
        12: "#7f7f7f", # pad
        13: "#8c564b", # gap
    }

    uniq = sorted(color_table.keys())
    idx_map = {v: i for i, v in enumerate(uniq)}
    Zm = np.vectorize(idx_map.get)(Z)

    cmap = ListedColormap([color_table[v] for v in uniq])
    norm = BoundaryNorm(np.arange(len(uniq) + 1) - 0.5, cmap.N)

    fig_w = max(10.0, min(24.0, 0.03 * L + 6.0))
    fig_h = 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.imshow(Zm, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["TRUE(aln)", "PRED(aln)", "MATCH"], fontsize=9)
    ax.set_xlabel("aligned position", fontsize=9)

    step = 10 if L <= 120 else (25 if L <= 300 else 50)
    xticks = list(range(0, L, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=7)

    ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# =========================
# Callback
# =========================
@dataclass
class ValLabelPlotCallbackConfig:
    enabled: bool = False
    max_len: int = 300
    key: str = "val/label_heatmap_aligned"
    save_png: bool = True
    dpi: int = 200

    # ✅ 解决你现在的“true/pred 是 id list”的问题
    blank_id: int = 0
    id2base: Optional[Dict[int, str]] = None  # 不提供就用默认 {1:A,2:T,3:G,4:C}


class ValLabelPlotCallback:
    """
    sample 里允许：
      sample["true_seq"] = List[int] 或 str
      sample["pred_seq"] = List[int] 或 str（可为空）
    """
    def __init__(self, cfg: ValLabelPlotCallbackConfig):
        self.cfg = cfg
        if self.cfg.id2base is None:
            self.cfg.id2base = _default_id2base()

    def on_val_end(
        self,
        epoch: int,
        output_dir: str,
        sample: Optional[Dict[str, Any]] = None,
        wandb_run=None,
        split_name: str = "Val",
    ):
        if not self.cfg.enabled or sample is None:
            return

        # 注意：这里不再要求 pred 非空
        true_raw = sample.get("true_seq", None)
        pred_raw = sample.get("pred_seq", None)

        true_seq = normalize_seq_input(true_raw, id2base=self.cfg.id2base, blank_id=self.cfg.blank_id)
        pred_seq = normalize_seq_input(pred_raw, id2base=self.cfg.id2base, blank_id=self.cfg.blank_id)

        if not true_seq:
            return  # true 为空没法画

        note = sample.get("note", "")
        title = f"{split_name} indel-aware label heatmap @ epoch {epoch}"
        if note:
            title += f" [{note}]"

        png_path = os.path.join(output_dir, f"{split_name.lower()}_label_heatmap_aligned_epoch{epoch}.png")

        try:
            plot_aligned_heatmap_png(
                true_seq=true_seq,
                pred_seq=pred_seq,      # 允许为空字符串
                out_png=png_path,
                title=title,
                max_len=self.cfg.max_len,
                dpi=self.cfg.dpi,
            )
        except Exception as e:
            print(f"[Heatmap] matplotlib export failed: {e}", flush=True)
            return

        # ✅ W&B 上传：保证网页会出现 media
        if wandb_run is not None and wandb is not None and self.cfg.save_png:
            try:
                wandb.log({self.cfg.key + "_png": wandb.Image(png_path)}, step=epoch)
            except Exception as e:
                print(f"[Heatmap] wandb.Image log failed: {e}", flush=True)
