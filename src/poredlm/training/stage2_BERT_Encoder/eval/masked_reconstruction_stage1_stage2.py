"""Masked token reconstruction, then decode BERT tokens back to nanopore signal.

Pipeline:
    selected raw signal chunk -> optional signal crop -> stage1 VQ tokenizer
    -> random BERT masks on stage1 tokens -> BERT MLM
    -> replace only masked positions -> stage1 codebook + CNN decoder -> signal
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def patch_numpy_for_accelerate() -> None:
    """Expose numpy aliases expected by newer accelerate releases."""

    np_core = getattr(np, "_core", None)
    legacy_core = getattr(np, "core", None)
    if np_core is not None and legacy_core is not None:
        for name in ("multiarray", "umath"):
            if not hasattr(np_core, name) and hasattr(legacy_core, name):
                setattr(np_core, name, getattr(legacy_core, name))

    if not hasattr(np, "dtypes"):
        dtype_names = {
            "BoolDType": "bool",
            "Int8DType": "int8",
            "Int16DType": "int16",
            "Int32DType": "int32",
            "Int64DType": "int64",
            "UInt8DType": "uint8",
            "UInt16DType": "uint16",
            "UInt32DType": "uint32",
            "UInt64DType": "uint64",
            "Float16DType": "float16",
            "Float32DType": "float32",
            "Float64DType": "float64",
            "Complex64DType": "complex64",
            "Complex128DType": "complex128",
            "ObjectDType": "object",
            "BytesDType": "bytes",
            "StrDType": "str",
            "VoidDType": "void",
            "DateTime64DType": "datetime64",
            "TimeDelta64DType": "timedelta64",
        }
        np.dtypes = types.SimpleNamespace(
            **{name: type(np.dtype(dtype)) for name, dtype in dtype_names.items()}
        )


patch_numpy_for_accelerate()

from transformers import BertForMaskedLM


REPO_ROOT = Path(__file__).resolve().parents[1]
POREDLM_SRC = REPO_ROOT / "src" / "poredlm"
PROJECT_SRC = REPO_ROOT / "src"
STAGE1_TOKENIZER_SRC = POREDLM_SRC / "training" / "stage1_tokenizer"
for import_path in (STAGE1_TOKENIZER_SRC, POREDLM_SRC, PROJECT_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def install_optional_bonito_stub() -> None:
    """Allow tokenizer inference without the optional Bonito teacher dependency."""

    if "bonito.util" in sys.modules:
        return
    try:
        from bonito.util import load_model as _load_model  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if exc.name not in {"bonito", "bonito.util"}:
            raise

    bonito_module = sys.modules.setdefault("bonito", types.ModuleType("bonito"))
    util_module = types.ModuleType("bonito.util")

    def load_model(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            "bonito is not installed. It is only required when loading a "
            "stage1 tokenizer with teacher_model_path for distillation."
        )

    util_module.load_model = load_model
    bonito_module.util = util_module
    sys.modules["bonito.util"] = util_module


install_optional_bonito_stub()

from tokenizer_model_v0 import Nanopore_Tokenizer_Model_V0
from tokenizer_model_v1 import Nanopore_Tokenizer_Model_V1


BWAV_VOCAB_OFFSET = 129


def default_plot_path(output_npz: str) -> str:
    """Use the npz output path to derive a default comparison figure path."""

    return str(Path(output_npz).with_suffix(".png"))


def resolve_mask_probability(*, mask_probability: float, mask_percentage: float | None) -> float:
    """Resolve token mask probability from either a fraction or a human percentage."""

    if mask_percentage is not None:
        if not (0.0 <= mask_percentage <= 100.0):
            raise ValueError("--mask-percentage must be in [0, 100].")
        return float(mask_percentage) / 100.0

    if not (0.0 <= mask_probability <= 1.0):
        raise ValueError("--mask-probability must be in [0, 1]. Use --mask-percentage for 0..100 values.")
    return float(mask_probability)


def token_mask_to_signal_spans(
    masked_positions: np.ndarray,
    *,
    downsample_rate: int,
    signal_length: int,
) -> np.ndarray:
    """Map masked stage1 token positions back to contiguous signal-sample spans."""

    if downsample_rate <= 0:
        raise ValueError(f"Invalid stage1 downsample_rate={downsample_rate}.")

    mask = np.asarray(masked_positions, dtype=bool).reshape(-1)
    spans: list[tuple[int, int]] = []
    run_start: int | None = None
    for token_index, is_masked in enumerate(mask):
        if is_masked and run_start is None:
            run_start = token_index
        elif not is_masked and run_start is not None:
            spans.append((run_start * downsample_rate, token_index * downsample_rate))
            run_start = None
    if run_start is not None:
        spans.append((run_start * downsample_rate, len(mask) * downsample_rate))

    clipped_spans = [
        (max(0, start), min(signal_length, end))
        for start, end in spans
        if max(0, start) < min(signal_length, end)
    ]
    return np.asarray(clipped_spans, dtype=np.int64).reshape(-1, 2)


def _visible_signal_spans(
    mask_signal_spans: np.ndarray | None,
    *,
    start: int,
    end: int,
) -> np.ndarray:
    """Clip mask spans to the plotted signal window."""

    if mask_signal_spans is None:
        return np.zeros((0, 2), dtype=np.int64)
    spans = np.asarray(mask_signal_spans, dtype=np.int64).reshape(-1, 2)
    visible = [
        (max(start, int(span_start)), min(end, int(span_end)))
        for span_start, span_end in spans
        if max(start, int(span_start)) < min(end, int(span_end))
    ]
    return np.asarray(visible, dtype=np.int64).reshape(-1, 2)


def save_signal_comparison_plot(
    *,
    original_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    output_path: str,
    title: str,
    baseline_signal: np.ndarray | None = None,
    mask_signal_spans: np.ndarray | None = None,
    plot_start: int = 0,
    plot_num_samples: int = 5000,
) -> str:
    """Save one image comparing signals and highlighting masked token-covered regions."""

    original_signal = normalize_signal_array(original_signal)
    reconstructed_signal = normalize_signal_array(reconstructed_signal)
    if original_signal.shape[0] != reconstructed_signal.shape[0]:
        raise ValueError(
            "original_signal and reconstructed_signal must have the same length, "
            f"got {original_signal.shape[0]} and {reconstructed_signal.shape[0]}"
        )
    if baseline_signal is not None:
        baseline_signal = normalize_signal_array(baseline_signal)
        if baseline_signal.shape[0] != original_signal.shape[0]:
            raise ValueError(
                "baseline_signal and original_signal must have the same length, "
                f"got {baseline_signal.shape[0]} and {original_signal.shape[0]}"
            )

    signal_len = original_signal.shape[0]
    start = max(0, int(plot_start))
    if start >= signal_len:
        raise ValueError(f"plot_start={plot_start} is outside signal length {signal_len}.")
    if plot_num_samples <= 0:
        end = signal_len
    else:
        end = min(signal_len, start + int(plot_num_samples))

    x = np.arange(start, end)
    original_window = original_signal[start:end]
    recon_window = reconstructed_signal[start:end]
    residual = recon_window - original_window
    baseline_window = baseline_signal[start:end] if baseline_signal is not None else None
    visible_mask_spans = _visible_signal_spans(mask_signal_spans, start=start, end=end)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_output = output if output.suffix.lower() == ".svg" else output.with_suffix(".svg")
        _save_signal_comparison_svg(
            x=x,
            original_window=original_window,
            recon_window=recon_window,
            residual=residual,
            output_path=svg_output,
            title=f"{title} ({start}:{end})",
            baseline_window=baseline_window,
            mask_signal_spans=visible_mask_spans,
        )
        return str(svg_output)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    for axis in axes:
        label_added = False
        for span_start, span_end in visible_mask_spans:
            axis.axvspan(
                int(span_start),
                int(span_end),
                color="#f0b429",
                alpha=0.22,
                label="Masked token region" if axis is axes[0] and not label_added else None,
            )
            label_added = True

    axes[0].plot(x, original_window, label="Original", color="#1f77b4", linewidth=1.0)
    axes[0].plot(x, recon_window, label="BERT reconstructed", color="#d62728", linewidth=0.9, alpha=0.85)
    if baseline_window is not None:
        axes[0].plot(
            x,
            baseline_window,
            label="Stage1 direct reconstruction",
            color="#7f7f7f",
            linewidth=0.8,
            alpha=0.65,
        )
    axes[0].set_title(f"{title} ({start}:{end})")
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, residual, label="BERT reconstructed - original", color="#9467bd", linewidth=0.8)
    axes[1].axhline(0.0, color="#222222", linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.2)

    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def _downsample_for_svg(*arrays: np.ndarray, max_points: int = 2500) -> tuple[np.ndarray, ...]:
    length = len(arrays[0])
    stride = max(1, int(np.ceil(length / max_points)))
    return tuple(array[::stride] for array in arrays)


def _make_svg_polyline(x: np.ndarray, y: np.ndarray) -> str:
    return " ".join(f"{float(px):.1f},{float(py):.1f}" for px, py in zip(x, y))


def _scale_svg_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0e-6)
    x_scaled = left + (x_values - x_min) / x_span * width
    y_scaled = top + height - (y_values - y_min) / y_span * height
    return x_scaled, y_scaled


def _save_signal_comparison_svg(
    *,
    x: np.ndarray,
    original_window: np.ndarray,
    recon_window: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
    title: str,
    baseline_window: np.ndarray | None = None,
    mask_signal_spans: np.ndarray | None = None,
) -> None:
    """Fallback comparison plot writer that only uses numpy and the Python stdlib."""

    if baseline_window is None:
        x_d, original_d, recon_d, residual_d = _downsample_for_svg(
            x,
            original_window,
            recon_window,
            residual,
        )
        baseline_d = None
    else:
        x_d, original_d, recon_d, baseline_d, residual_d = _downsample_for_svg(
            x,
            original_window,
            recon_window,
            baseline_window,
            residual,
        )

    svg_width = 1400
    svg_height = 760
    left = 82.0
    right = 36.0
    plot_width = svg_width - left - right
    main_top = 76.0
    main_height = 420.0
    residual_top = 586.0
    residual_height = 120.0

    y_candidates = [original_d, recon_d]
    if baseline_d is not None:
        y_candidates.append(baseline_d)
    y_min = float(min(np.min(values) for values in y_candidates))
    y_max = float(max(np.max(values) for values in y_candidates))
    y_pad = max((y_max - y_min) * 0.08, 1.0e-6)
    y_min -= y_pad
    y_max += y_pad

    residual_min = float(np.min(residual_d))
    residual_max = float(np.max(residual_d))
    residual_pad = max((residual_max - residual_min) * 0.15, 1.0e-6)
    residual_min -= residual_pad
    residual_max += residual_pad

    x_min = float(x[0])
    x_max = float(x[-1]) if len(x) > 1 else float(x[0] + 1)
    x_span = max(x_max - x_min, 1.0)
    ox, oy = _scale_svg_points(
        x_d,
        original_d,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=main_top,
        width=plot_width,
        height=main_height,
    )
    rx, ry = _scale_svg_points(
        x_d,
        recon_d,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        left=left,
        top=main_top,
        width=plot_width,
        height=main_height,
    )
    resx, resy = _scale_svg_points(
        x_d,
        residual_d,
        x_min=x_min,
        x_max=x_max,
        y_min=residual_min,
        y_max=residual_max,
        left=left,
        top=residual_top,
        width=plot_width,
        height=residual_height,
    )
    zero_x = np.asarray([x_min, x_max], dtype=np.float32)
    zero_y = np.asarray([0.0, 0.0], dtype=np.float32)
    zx, zy = _scale_svg_points(
        zero_x,
        zero_y,
        x_min=x_min,
        x_max=x_max,
        y_min=residual_min,
        y_max=residual_max,
        left=left,
        top=residual_top,
        width=plot_width,
        height=residual_height,
    )

    visible_mask_spans = _visible_signal_spans(
        mask_signal_spans,
        start=int(x_min),
        end=int(x_max) + 1,
    )
    main_mask_regions = []
    residual_mask_regions = []
    for span_start, span_end in visible_mask_spans:
        span_x0 = left + (float(span_start) - x_min) / x_span * plot_width
        span_x1 = left + (float(span_end) - x_min) / x_span * plot_width
        span_x0 = max(left, min(left + plot_width, span_x0))
        span_x1 = max(left, min(left + plot_width, span_x1))
        span_width = max(0.0, span_x1 - span_x0)
        if span_width <= 0:
            continue
        main_mask_regions.append(
            f'<rect x="{span_x0:.1f}" y="{main_top}" width="{span_width:.1f}" '
            f'height="{main_height}" fill="#f0b429" opacity="0.22"/>'
        )
        residual_mask_regions.append(
            f'<rect x="{span_x0:.1f}" y="{residual_top}" width="{span_width:.1f}" '
            f'height="{residual_height}" fill="#f0b429" opacity="0.22"/>'
        )
    main_mask_regions_svg = "\n  ".join(main_mask_regions)
    residual_mask_regions_svg = "\n  ".join(residual_mask_regions)

    baseline_polyline = ""
    if baseline_d is not None:
        bx, by = _scale_svg_points(
            x_d,
            baseline_d,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            left=left,
            top=main_top,
            width=plot_width,
            height=main_height,
        )
        baseline_polyline = (
            f'<polyline points="{_make_svg_polyline(bx, by)}" '
            'fill="none" stroke="#7f7f7f" stroke-width="1.1" opacity="0.65" />'
        )

    main_bottom = main_top + main_height
    residual_bottom = residual_top + residual_height
    escaped_title = html.escape(title)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{left}" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#202124">{escaped_title}</text>
  <rect x="{left}" y="{main_top}" width="{plot_width}" height="{main_height}" fill="#fbfbfb" stroke="#d8d8d8"/>
  {main_mask_regions_svg}
  <line x1="{left}" y1="{main_bottom}" x2="{left + plot_width}" y2="{main_bottom}" stroke="#444"/>
  <line x1="{left}" y1="{main_top}" x2="{left}" y2="{main_bottom}" stroke="#444"/>
  <polyline points="{_make_svg_polyline(ox, oy)}" fill="none" stroke="#1f77b4" stroke-width="1.4" />
  <polyline points="{_make_svg_polyline(rx, ry)}" fill="none" stroke="#d62728" stroke-width="1.2" opacity="0.86" />
  {baseline_polyline}
  <text x="{left + 16}" y="{main_top + 26}" font-family="Arial, sans-serif" font-size="14" fill="#1f77b4">Original</text>
  <text x="{left + 110}" y="{main_top + 26}" font-family="Arial, sans-serif" font-size="14" fill="#d62728">BERT reconstructed</text>
  <text x="{left + 282}" y="{main_top + 26}" font-family="Arial, sans-serif" font-size="14" fill="#777">Stage1 direct reconstruction</text>
  <text x="{left + 488}" y="{main_top + 26}" font-family="Arial, sans-serif" font-size="14" fill="#9a6700">Masked token regions</text>
  <text x="20" y="{main_top + main_height / 2}" font-family="Arial, sans-serif" font-size="14" fill="#555" transform="rotate(-90 20 {main_top + main_height / 2})">Signal</text>
  <text x="{left}" y="{main_bottom + 24}" font-family="Arial, sans-serif" font-size="13" fill="#555">{x_min:.0f}</text>
  <text x="{left + plot_width - 80}" y="{main_bottom + 24}" font-family="Arial, sans-serif" font-size="13" fill="#555">{x_max:.0f}</text>

  <rect x="{left}" y="{residual_top}" width="{plot_width}" height="{residual_height}" fill="#fbfbfb" stroke="#d8d8d8"/>
  {residual_mask_regions_svg}
  <line x1="{left}" y1="{residual_bottom}" x2="{left + plot_width}" y2="{residual_bottom}" stroke="#444"/>
  <line x1="{left}" y1="{residual_top}" x2="{left}" y2="{residual_bottom}" stroke="#444"/>
  <line x1="{zx[0]:.1f}" y1="{zy[0]:.1f}" x2="{zx[1]:.1f}" y2="{zy[1]:.1f}" stroke="#222" stroke-width="1" opacity="0.65"/>
  <polyline points="{_make_svg_polyline(resx, resy)}" fill="none" stroke="#9467bd" stroke-width="1.1" />
  <text x="{left + 16}" y="{residual_top + 24}" font-family="Arial, sans-serif" font-size="14" fill="#9467bd">BERT reconstructed - original</text>
  <text x="22" y="{residual_top + residual_height / 2}" font-family="Arial, sans-serif" font-size="14" fill="#555" transform="rotate(-90 22 {residual_top + residual_height / 2})">Residual</text>
  <text x="{left + plot_width / 2 - 48}" y="{residual_bottom + 38}" font-family="Arial, sans-serif" font-size="14" fill="#555">Sample index</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def load_stage1_accelerate_checkpoint(model_ckpt_dir: str) -> dict[str, Any]:
    """Load the stage1 checkpoint format saved by Accelerate."""

    ckpt_dir = Path(model_ckpt_dir)
    safetensors_path = ckpt_dir / "model.safetensors"
    bin_path = ckpt_dir / "pytorch_model.bin"
    metadata_path = ckpt_dir / "metadata.json"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        candidates = [
            path
            for path in ckpt_dir.iterdir()
            if path.name.endswith((".bin", ".safetensors")) and "model" in path.name
        ]
        if not candidates:
            raise FileNotFoundError(f"No model weights found under {ckpt_dir}")
        weights_path = candidates[0]
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_path), device="cpu")
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {ckpt_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("cnn_type", "model_type", "codebook_size", "codebook_dim"):
        if key not in metadata:
            raise KeyError(f"Missing {key!r} in {metadata_path}")

    return {
        "model_state_dict": state_dict,
        "cnn_type": int(metadata["cnn_type"]),
        "model_type": int(metadata["model_type"]),
        "codebook_size": int(metadata["codebook_size"]),
        "codebook_dim": int(metadata["codebook_dim"]),
    }


def build_stage1_model(model_ckpt_dir: str, device: torch.device) -> torch.nn.Module:
    """Instantiate the saved stage1 tokenizer model."""

    ckpt = load_stage1_accelerate_checkpoint(model_ckpt_dir)
    model_type = ckpt["model_type"]
    if model_type == 0:
        model_cls = Nanopore_Tokenizer_Model_V0
    elif model_type == 1:
        model_cls = Nanopore_Tokenizer_Model_V1
    else:
        raise ValueError(f"Unsupported stage1 model_type={model_type}; expected 0 or 1.")
    model = model_cls(codebook_size=ckpt["codebook_size"], cnn_type=ckpt["cnn_type"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    model.to(device)
    return model


def normalize_signal_array(array: np.ndarray) -> np.ndarray:
    """Accept common numpy layouts and return a 1D float32 signal."""

    array = np.asarray(array)
    if array.ndim == 1:
        signal = array
    elif array.ndim == 2 and 1 in array.shape:
        signal = array.reshape(-1)
    else:
        raise ValueError(f"Expected a 1D signal or shape [1, T]/[T, 1], got {array.shape}")
    return signal.astype(np.float32, copy=False)


def resolve_signal_npy_path(input_npy: str) -> Path:
    """Use the paired chunks file when a references npy is passed by mistake."""

    path = Path(input_npy)
    if "references" not in path.name:
        return path

    chunks_name = path.name.replace("references", "chunks", 1)
    chunks_path = path.with_name(chunks_name)
    if chunks_path.exists():
        print(
            f"[INFO] Input looks like a references file; using paired signal chunks file: {chunks_path}",
            file=sys.stderr,
        )
        return chunks_path

    print(
        f"[WARN] Input looks like a references file, but paired chunks file was not found: {chunks_path}",
        file=sys.stderr,
    )
    return path


def load_input_signal(
    input_npy: str,
    *,
    input_index: int = 0,
    input_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one signal from a 1D signal, a 2D chunk matrix, or a list-like npy."""

    resolved_path = resolve_signal_npy_path(input_npy)
    array = np.load(resolved_path, allow_pickle=True)
    info: dict[str, Any] = {
        "input_npy": str(input_npy),
        "resolved_input_npy": str(resolved_path),
        "input_shape": tuple(array.shape),
        "input_mode": input_mode,
        "input_index": int(input_index),
    }

    if array.ndim == 1 and array.dtype != object:
        info["selected_signal_shape"] = tuple(array.shape)
        return normalize_signal_array(array), info

    if input_mode not in {"auto", "row", "flatten"}:
        raise ValueError(f"Unsupported input_mode={input_mode!r}; expected auto, row, or flatten.")

    if input_mode == "flatten" and array.dtype != object:
        signal = np.asarray(array).reshape(-1)
        info["selected_signal_shape"] = tuple(signal.shape)
        print(
            f"[INFO] Loaded {resolved_path} with shape {array.shape}; flattened to one signal of length {len(signal)}.",
            file=sys.stderr,
        )
        return normalize_signal_array(signal), info

    if array.ndim == 0:
        raise ValueError(f"Expected a signal array or chunk matrix, got scalar array from {resolved_path}.")

    if input_index < 0 or input_index >= len(array):
        raise IndexError(
            f"--input-index={input_index} is out of range for {resolved_path} with {len(array)} rows/items."
        )

    selected = np.asarray(array[input_index])
    info["selected_signal_shape"] = tuple(selected.shape)
    print(
        f"[INFO] Loaded {resolved_path} with shape {array.shape}; using row/item {input_index} "
        f"as one signal with shape {selected.shape}.",
        file=sys.stderr,
    )
    return normalize_signal_array(selected), info


def crop_signal(
    signal: np.ndarray,
    *,
    signal_start: int,
    signal_length: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Crop a signal window. A missing or non-positive length means use all remaining samples."""

    signal = normalize_signal_array(signal)
    if signal_start < 0:
        raise ValueError("--signal-start must be >= 0.")
    if signal_start >= len(signal):
        raise ValueError(f"--signal-start={signal_start} is outside signal length {len(signal)}.")

    if signal_length is None or signal_length <= 0:
        signal_end = len(signal)
    else:
        signal_end = signal_start + int(signal_length)
        if signal_end > len(signal):
            raise ValueError(
                f"Cannot crop [{signal_start}, {signal_end}) from signal length {len(signal)}. "
                "Choose a smaller --signal-start or --signal-length."
            )

    cropped = signal[signal_start:signal_end].astype(np.float32, copy=False)
    return cropped, {
        "signal_start": int(signal_start),
        "signal_end": int(signal_end),
        "signal_length": int(len(cropped)),
    }


def extract_codebook_indices(indices: torch.Tensor, layer: int = 0) -> torch.Tensor:
    """Normalize VQ indices to shape [B, N] for the single-codebook stage1 model."""

    if indices.ndim == 3:
        return indices[..., layer].long()
    if indices.ndim == 2:
        return indices.long()
    raise ValueError(f"Unexpected VQ index shape: {tuple(indices.shape)}")


def tokenize_signal(
    model: torch.nn.Module,
    signal: np.ndarray,
    *,
    device: torch.device,
    token_batch_size: int,
    layer: int = 0,
) -> np.ndarray:
    """Tokenize a 1D signal with the stage1 model, using the same chunking idea as VQETokenizer."""

    signal = normalize_signal_array(signal)
    downsample_rate = int(model.cnn_stride)
    model_rf = int(model.RF)
    if len(signal) < model_rf:
        return np.array([], dtype=np.int64)

    token_batch_size = max(1, int(token_batch_size))
    chunk_size = token_batch_size * downsample_rate
    margin_stride_count = int(getattr(model, "margin_stride_count", 2))
    margin_samples = margin_stride_count * downsample_rate
    expected_tokens = (len(signal) + downsample_rate - 1) // downsample_rate

    def run_chunk(chunk: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(x)
        return extract_codebook_indices(outputs[1], layer=layer)[0].detach().cpu().numpy().astype(np.int64)

    if len(signal) <= chunk_size:
        padded = np.pad(signal, (0, chunk_size - len(signal)), mode="constant")
        return run_chunk(padded)[:expected_tokens]

    step_samples = chunk_size - 2 * margin_samples
    if step_samples <= 0:
        raise ValueError("token_batch_size is too small for the configured stage1 margin.")

    pieces: list[np.ndarray] = []
    start = 0
    chunk_index = 0
    while start < len(signal):
        real_len = min(chunk_size, len(signal) - start)
        chunk = signal[start : start + real_len]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")

        tokens = run_chunk(chunk)
        valid_tokens = (real_len + downsample_rate - 1) // downsample_rate
        if chunk_index == 0:
            keep = tokens[: max(0, valid_tokens - margin_stride_count)]
        elif start + step_samples >= len(signal):
            keep = tokens[margin_stride_count:valid_tokens]
        else:
            keep = tokens[
                margin_stride_count : max(margin_stride_count, valid_tokens - margin_stride_count)
            ]
        if keep.size:
            pieces.append(keep.astype(np.int64, copy=False))

        start += step_samples
        chunk_index += 1

    if not pieces:
        return np.zeros(expected_tokens, dtype=np.int64)
    token_ids = np.concatenate(pieces, axis=0)
    if len(token_ids) > expected_tokens:
        token_ids = token_ids[:expected_tokens]
    elif len(token_ids) < expected_tokens:
        token_ids = np.pad(token_ids, (0, expected_tokens - len(token_ids)), constant_values=0)
    return token_ids.astype(np.int64, copy=False)


def get_codebook_embed(model: torch.nn.Module) -> torch.Tensor:
    """Return stage1 codebook embedding as [K, D]."""

    embed = model.vq._codebook.embed
    if embed.ndim == 3:
        embed = embed[0]
    if embed.ndim != 2:
        raise RuntimeError(f"Unexpected codebook embedding shape: {tuple(embed.shape)}")
    return embed


def decode_codebook_ids(
    model: torch.nn.Module,
    token_ids: np.ndarray | torch.Tensor,
    *,
    target_signal_len: int,
    device: torch.device,
) -> np.ndarray:
    """Decode stage1 codebook ids back into signal with the CNN decoder."""

    if isinstance(token_ids, np.ndarray):
        token_ids_t = torch.from_numpy(token_ids).long()
    else:
        token_ids_t = token_ids.detach().cpu().long()
    if token_ids_t.ndim == 1:
        token_ids_t = token_ids_t.unsqueeze(0)
    if token_ids_t.ndim != 2:
        raise ValueError(f"token_ids must be [N] or [B, N], got {tuple(token_ids_t.shape)}")

    token_ids_t = token_ids_t.to(device)
    if token_ids_t.numel() == 0:
        raise ValueError("Cannot decode an empty token sequence.")
    if token_ids_t.min() < 0 or token_ids_t.max() >= int(model.codebook_size):
        raise ValueError(
            f"stage1 codebook id out of range [0, {int(model.codebook_size) - 1}], "
            f"got min={int(token_ids_t.min())}, max={int(token_ids_t.max())}"
        )

    codebook = get_codebook_embed(model).to(device)
    z_q = codebook[token_ids_t].permute(0, 2, 1).contiguous()
    with torch.no_grad():
        recon = model.cnn_model.decode(z_q).squeeze(1)

    current_len = recon.shape[-1]
    if current_len > target_signal_len:
        recon = recon[..., :target_signal_len]
    elif current_len < target_signal_len:
        recon = F.pad(recon, (0, target_signal_len - current_len))
    return recon[0].detach().cpu().numpy()


def run_masked_bert_reconstruction(
    bert: BertForMaskedLM,
    codebook_ids: np.ndarray,
    *,
    device: torch.device,
    codebook_size: int,
    mask_probability: float,
    mask_token_id: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict masked BERT tokens and replace only masked positions."""

    repaired_windows: list[np.ndarray] = []
    corrupted_windows: list[np.ndarray] = []
    mask_windows: list[np.ndarray] = []

    for start in range(0, len(codebook_ids), max_length):
        window = codebook_ids[start : start + max_length]
        input_ids = torch.from_numpy(window + BWAV_VOCAB_OFFSET).long().unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        masked_positions = torch.rand(input_ids.shape, device=device) < mask_probability

        corrupted = input_ids.clone()
        corrupted[masked_positions] = mask_token_id

        with torch.no_grad():
            logits = bert(input_ids=corrupted, attention_mask=attention_mask).logits

        logits[..., :BWAV_VOCAB_OFFSET] = -float("inf")
        logits[..., BWAV_VOCAB_OFFSET + codebook_size :] = -float("inf")
        predicted_vocab_ids = logits.argmax(dim=-1)

        repaired_vocab_ids = input_ids.clone()
        repaired_vocab_ids[masked_positions] = predicted_vocab_ids[masked_positions]
        repaired_codebook_ids = repaired_vocab_ids.squeeze(0).detach().cpu().numpy() - BWAV_VOCAB_OFFSET

        repaired_windows.append(repaired_codebook_ids.astype(np.int64, copy=False))
        corrupted_windows.append(corrupted.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False))
        mask_windows.append(masked_positions.squeeze(0).detach().cpu().numpy().astype(bool, copy=False))

    return (
        np.concatenate(repaired_windows, axis=0),
        np.concatenate(corrupted_windows, axis=0),
        np.concatenate(mask_windows, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage1 -> masked BERT MLM -> stage1 decoder reconstruction.")
    parser.add_argument("--stage1-ckpt", required=True, help="Stage1 Accelerate checkpoint directory.")
    parser.add_argument("--stage2-bert", required=True, help="Stage2 BertForMaskedLM checkpoint directory.")
    parser.add_argument("--input-npy", required=True, help="Input .npy file containing a 1D signal or 2D signal chunks.")
    parser.add_argument("--output-npz", required=True, help="Output .npz with tokens, masks, and signals.")
    parser.add_argument("--output-plot", default=None, help="Output comparison image. Defaults to output-npz with .png.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc. Defaults to CUDA when available.")
    parser.add_argument("--input-index", type=int, default=0, help="Row/item index used when input-npy contains many chunks.")
    parser.add_argument(
        "--input-mode",
        choices=("auto", "row", "flatten"),
        default="auto",
        help="How to handle multi-signal input. auto/row selects one row; flatten concatenates numeric arrays.",
    )
    parser.add_argument("--signal-start", type=int, default=0, help="Start sample index for the signal crop.")
    parser.add_argument(
        "--signal-length",
        type=int,
        default=500,
        help="Number of signal samples to keep. Defaults to 500. Use <=0 to use all remaining samples.",
    )
    parser.add_argument("--mask-probability", type=float, default=0.15, help="Token mask probability as a 0..1 fraction.")
    parser.add_argument(
        "--mask-percentage",
        type=float,
        default=None,
        help="Token mask percentage as 0..100. Overrides --mask-probability when provided.",
    )
    parser.add_argument("--mask-token-id", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None, help="BERT window length. Defaults to BERT max positions.")
    parser.add_argument("--token-batch-size", type=int, default=8000, help="Stage1 chunk token count.")
    parser.add_argument("--plot-start", type=int, default=0, help="Start sample index for the comparison figure.")
    parser.add_argument(
        "--plot-num-samples",
        type=int,
        default=500,
        help="Number of samples to draw. Use <=0 to draw all remaining samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mask_probability = resolve_mask_probability(
        mask_probability=args.mask_probability,
        mask_percentage=args.mask_percentage,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loaded_signal, input_info = load_input_signal(
        args.input_npy,
        input_index=args.input_index,
        input_mode=args.input_mode,
    )
    signal, crop_info = crop_signal(
        loaded_signal,
        signal_start=args.signal_start,
        signal_length=args.signal_length,
    )
    stage1 = build_stage1_model(args.stage1_ckpt, device)
    bert = BertForMaskedLM.from_pretrained(args.stage2_bert).to(device).eval()
    required_vocab_size = BWAV_VOCAB_OFFSET + int(stage1.codebook_size)
    if bert.config.vocab_size < required_vocab_size:
        raise ValueError(
            f"Stage2 BERT vocab_size={bert.config.vocab_size} is smaller than "
            f"129 + stage1 codebook_size={required_vocab_size}."
        )
    max_length = int(args.max_length or bert.config.max_position_embeddings)

    codebook_ids = tokenize_signal(
        stage1,
        signal,
        device=device,
        token_batch_size=args.token_batch_size,
    )
    if codebook_ids.size == 0:
        raise RuntimeError("Stage1 produced no tokens. Check signal length and checkpoint configuration.")

    # Masking happens on stage1 tokenizer ids, then the masked token spans are mapped
    # back to signal samples only for visualization.
    repaired_codebook_ids, corrupted_vocab_ids, masked_positions = run_masked_bert_reconstruction(
        bert,
        codebook_ids,
        device=device,
        codebook_size=int(stage1.codebook_size),
        mask_probability=mask_probability,
        mask_token_id=args.mask_token_id,
        max_length=max_length,
    )
    mask_signal_spans = token_mask_to_signal_spans(
        masked_positions,
        downsample_rate=int(stage1.cnn_stride),
        signal_length=len(signal),
    )

    baseline_signal = decode_codebook_ids(stage1, codebook_ids, target_signal_len=len(signal), device=device)
    reconstructed_signal = decode_codebook_ids(
        stage1,
        repaired_codebook_ids,
        target_signal_len=len(signal),
        device=device,
    )

    masked_count = int(masked_positions.sum())
    token_accuracy_masked = float("nan")
    if masked_count > 0:
        token_accuracy_masked = float((repaired_codebook_ids[masked_positions] == codebook_ids[masked_positions]).mean())

    np.savez_compressed(
        args.output_npz,
        original_signal=signal,
        baseline_stage1_signal=baseline_signal,
        reconstructed_signal=reconstructed_signal,
        original_codebook_ids=codebook_ids,
        repaired_codebook_ids=repaired_codebook_ids,
        corrupted_bert_vocab_ids=corrupted_vocab_ids,
        masked_positions=masked_positions,
        mask_signal_spans=mask_signal_spans,
        input_npy=np.asarray(input_info["input_npy"]),
        resolved_input_npy=np.asarray(input_info["resolved_input_npy"]),
        input_shape=np.asarray(input_info["input_shape"], dtype=np.int64),
        selected_signal_shape=np.asarray(input_info["selected_signal_shape"], dtype=np.int64),
        input_index=np.asarray(input_info["input_index"], dtype=np.int64),
        input_mode=np.asarray(input_info["input_mode"]),
        signal_start=np.asarray(crop_info["signal_start"], dtype=np.int64),
        signal_end=np.asarray(crop_info["signal_end"], dtype=np.int64),
        signal_length=np.asarray(crop_info["signal_length"], dtype=np.int64),
        requested_signal_length=np.asarray(-1 if args.signal_length is None else args.signal_length, dtype=np.int64),
        mask_probability=np.asarray(mask_probability, dtype=np.float32),
        mask_percentage=np.asarray(mask_probability * 100.0, dtype=np.float32),
        token_accuracy_masked=np.asarray(token_accuracy_masked, dtype=np.float32),
        signal_mse=np.asarray(float(np.mean((reconstructed_signal - signal) ** 2)), dtype=np.float32),
        stage1_baseline_mse=np.asarray(float(np.mean((baseline_signal - signal) ** 2)), dtype=np.float32),
    )

    output_plot = args.output_plot or default_plot_path(args.output_npz)
    output_plot = save_signal_comparison_plot(
        original_signal=signal,
        reconstructed_signal=reconstructed_signal,
        baseline_signal=baseline_signal,
        output_path=output_plot,
        title=f"Masked BERT reconstruction ({mask_probability * 100.0:.2f}% token mask)",
        mask_signal_spans=mask_signal_spans,
        plot_start=args.plot_start,
        plot_num_samples=args.plot_num_samples,
    )

    print(f"Saved reconstruction artifact: {os.path.abspath(args.output_npz)}")
    print(f"Saved comparison plot: {os.path.abspath(output_plot)}")
    print(
        f"signal_crop={crop_info['signal_start']}:{crop_info['signal_end']} "
        f"(len={crop_info['signal_length']}), tokens={len(codebook_ids)}, "
        f"masked={masked_count}, mask_percentage={mask_probability * 100.0:.2f}, "
        f"masked_token_acc={token_accuracy_masked:.6f}"
    )
    print(f"signal_mse={float(np.mean((reconstructed_signal - signal) ** 2)):.6f}")


if __name__ == "__main__":
    main()
