"""Evaluate masked-token prediction with the current public Stage1/Stage2 pipeline.

raw signal -> PoreVQCodec codebook ids -> offset/special tokens -> Stage2 BERT
-> repair masked content tokens -> PoreVQCodec decoder -> reconstructed signal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModel


HERE = Path(__file__).resolve().parent
STAGE2_DIR = HERE.parent
TOKEN_DATASET_DIR = STAGE2_DIR / "token_dataset"
for path in (STAGE2_DIR, TOKEN_DATASET_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Register the two custom Hugging Face model types before from_pretrained().
import modeling_pore_vq_codec  # noqa: E402,F401
from modeling_stage2_bert import Stage2MaskedSignalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mask current PoreVQCodec tokens and evaluate Stage2 BERT recovery."
    )
    parser.add_argument("--codec", required=True, help="PoreVQCodec save_pretrained directory")
    parser.add_argument("--bert", required=True, help="Stage2MaskedSignalLM checkpoint directory")
    parser.add_argument(
        "--training-config",
        default=None,
        help="Stage2 training YAML. Defaults to <bert>/training_config.yaml.",
    )
    parser.add_argument("--input-npy", required=True, help="1-D signal or array of signal rows")
    parser.add_argument("--input-index", type=int, default=0, help="Row selected from a multi-row npy")
    parser.add_argument("--signal-start", type=int, default=0)
    parser.add_argument("--signal-length", type=int, default=6000, help="<=0 means until the end")
    parser.add_argument("--mask-mode", choices=("contiguous", "random"), default="contiguous")
    parser.add_argument("--mask-token-start", type=int, default=None, help="Content-token index; random if omitted")
    parser.add_argument("--mask-token-length", type=int, default=4)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    parser.add_argument("--no-signal-plot", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "model" not in data or "data" not in data:
        raise ValueError(f"Expected data/model sections in {path}")
    return data


def load_signal(path: Path, index: int, start: int, length: int) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if data.ndim == 1:
        signal = data
    else:
        if not 0 <= index < len(data):
            raise IndexError(f"input-index={index} outside [0, {len(data)})")
        signal = np.asarray(data[index]).reshape(-1)
    if start < 0 or start >= signal.size:
        raise ValueError(f"signal-start={start} outside signal length {signal.size}")
    end = signal.size if length <= 0 else min(signal.size, start + length)
    signal = np.asarray(signal[start:end], dtype=np.float32)
    if signal.size == 0:
        raise ValueError("Selected signal crop is empty")
    return signal


def choose_mask(num_tokens: int, args: argparse.Namespace) -> np.ndarray:
    if num_tokens < 1:
        raise ValueError("Codec produced no content tokens")
    rng = np.random.default_rng(args.seed)
    mask = np.zeros(num_tokens, dtype=bool)
    if args.mask_mode == "contiguous":
        length = min(max(1, args.mask_token_length), num_tokens)
        if args.mask_token_start is None or args.mask_token_start < 0:
            start = int(rng.integers(0, num_tokens - length + 1))
        else:
            start = args.mask_token_start
        if start + length > num_tokens:
            raise ValueError(f"Mask [{start}, {start + length}) exceeds {num_tokens} content tokens")
        mask[start : start + length] = True
    else:
        if not 0.0 < args.mask_probability <= 1.0:
            raise ValueError("mask-probability must be in (0, 1]")
        mask = rng.random(num_tokens) < args.mask_probability
        if not mask.any():
            mask[int(rng.integers(num_tokens))] = True
    return mask


def decode(codec: torch.nn.Module, ids: np.ndarray, signal_len: int, device: torch.device) -> np.ndarray:
    token_tensor = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        reconstructed = codec.decode_token(token_tensor).squeeze().float().cpu().numpy()
    reconstructed = np.asarray(reconstructed).reshape(-1)
    if reconstructed.size < signal_len:
        reconstructed = np.pad(reconstructed, (0, signal_len - reconstructed.size))
    return reconstructed[:signal_len].astype(np.float32, copy=False)


def save_plots(
    output_dir: Path,
    signal: np.ndarray,
    baseline: np.ndarray,
    repaired: np.ndarray,
    original_ids: np.ndarray,
    predicted_ids: np.ndarray,
    mask: np.ndarray,
    stride: int,
    save_signal_plot: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(original_ids.size)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, original_ids, ".-", ms=3, lw=0.7, label="codec id (target)")
    ax.plot(x, predicted_ids, ".-", ms=3, lw=0.7, alpha=0.8, label="BERT repaired id")
    ax.scatter(x[mask], predicted_ids[mask], c="tab:red", s=24, label="masked positions", zorder=3)
    ax.set(xlabel="content token index", ylabel="codebook id", title="Masked-token reconstruction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "token_comparison.png", dpi=180)
    plt.close(fig)

    if not save_signal_plot:
        return
    sx = np.arange(signal.size)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(sx, signal, lw=0.8, label="original signal")
    ax.plot(sx, baseline, lw=0.8, alpha=0.8, label="codec baseline")
    ax.plot(sx, repaired, lw=0.8, alpha=0.8, label="BERT repaired")
    for token_index in np.flatnonzero(mask):
        ax.axvspan(token_index * stride, min((token_index + 1) * stride, signal.size), color="gold", alpha=0.18)
    ax.set(xlabel="signal sample", ylabel="value", title="Signal reconstruction (yellow = masked tokens)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "signal_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    bert_dir = Path(args.bert).expanduser().resolve()
    config_path = Path(args.training_config).expanduser().resolve() if args.training_config else bert_dir / "training_config.yaml"
    training_config = load_yaml(config_path)
    data_cfg, model_cfg = training_config["data"], training_config["model"]

    codec = AutoModel.from_pretrained(args.codec, trust_remote_code=True).to(device).eval()
    bert = Stage2MaskedSignalLM.from_pretrained(bert_dir).to(device).eval()
    signal = load_signal(Path(args.input_npy), args.input_index, args.signal_start, args.signal_length)

    signal_tensor = torch.from_numpy(signal).to(device).view(1, 1, -1)
    with torch.inference_mode():
        codebook_ids = codec.encode_signal(signal_tensor).reshape(-1).long().cpu().numpy()
    codebook_ids = codebook_ids.astype(np.int64, copy=False)
    content_mask = choose_mask(codebook_ids.size, args)

    token_offset = int(data_cfg.get("token_offset", 128))
    tokens_are_shifted = bool(data_cfg.get("tokens_are_shifted", True))
    add_cls = bool(data_cfg.get("add_cls_token", False))
    prefix = ([int(model_cfg.get("cls_token_id", 5))] if add_cls else []) + [int(model_cfg.get("bos_token_id", 2))]
    suffix = [int(model_cfg.get("eos_token_id", 3))]
    shifted = codebook_ids + token_offset if tokens_are_shifted else codebook_ids.copy()
    input_ids = np.asarray(prefix + shifted.tolist() + suffix, dtype=np.int64)
    content_start = len(prefix)
    sequence_mask = np.zeros(input_ids.size, dtype=bool)
    sequence_mask[content_start : content_start + codebook_ids.size] = content_mask
    corrupted = input_ids.copy()
    corrupted[sequence_mask] = int(model_cfg.get("mask_token_id", bert.config.mask_token_id))

    max_positions = int(bert.config.max_position_embeddings)
    if input_ids.size > max_positions:
        raise ValueError(
            f"BERT input has {input_ids.size} tokens but max_position_embeddings={max_positions}. "
            "Reduce --signal-length so the complete training-format sequence fits."
        )
    if int(input_ids.max()) >= int(bert.config.vocab_size):
        raise ValueError(f"Token id {int(input_ids.max())} exceeds BERT vocab_size={bert.config.vocab_size}")

    ids_tensor = torch.from_numpy(corrupted).to(device).unsqueeze(0)
    attention_mask = torch.ones_like(ids_tensor)
    with torch.inference_mode():
        logits = bert(input_ids=ids_tensor, attention_mask=attention_mask).logits[0]
    # Prevent special-token predictions: content targets use exactly this codebook range.
    codebook_size = int(codec.config.codebook_size)
    low = token_offset if tokens_are_shifted else 0
    high = low + codebook_size
    sequence_mask_tensor = torch.from_numpy(sequence_mask).to(device=device)
    masked_logits = logits[sequence_mask_tensor, low:high]
    predicted_codebook = masked_logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
    repaired_ids = codebook_ids.copy()
    repaired_ids[content_mask] = predicted_codebook

    baseline_signal = decode(codec, codebook_ids, signal.size, device)
    repaired_signal = decode(codec, repaired_ids, signal.size, device)
    masked_accuracy = float(np.mean(repaired_ids[content_mask] == codebook_ids[content_mask]))
    overall_accuracy = float(np.mean(repaired_ids == codebook_ids))
    baseline_mse = float(np.mean((baseline_signal - signal) ** 2))
    repaired_mse = float(np.mean((repaired_signal - signal) ** 2))
    stride = int(getattr(codec, "cnn_stride", getattr(codec, "stride", 1)))
    masked_samples = np.zeros(signal.size, dtype=bool)
    for i in np.flatnonzero(content_mask):
        masked_samples[i * stride : min((i + 1) * stride, signal.size)] = True
    masked_signal_mse = float(np.mean((repaired_signal[masked_samples] - signal[masked_samples]) ** 2))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "num_signal_samples": int(signal.size),
        "num_content_tokens": int(codebook_ids.size),
        "num_masked_tokens": int(content_mask.sum()),
        "masked_token_accuracy": masked_accuracy,
        "overall_token_accuracy": overall_accuracy,
        "codec_baseline_mse": baseline_mse,
        "bert_repaired_signal_mse": repaired_mse,
        "masked_signal_region_mse": masked_signal_mse,
        "token_offset": token_offset,
        "tokens_are_shifted": tokens_are_shifted,
        "add_cls_token": add_cls,
        "codec_stride": stride,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "result.npz",
        original_signal=signal,
        codec_baseline_signal=baseline_signal,
        bert_repaired_signal=repaired_signal,
        original_codebook_ids=codebook_ids,
        repaired_codebook_ids=repaired_ids,
        bert_input_ids=input_ids,
        corrupted_bert_input_ids=corrupted,
        masked_content_positions=content_mask,
        masked_sequence_positions=sequence_mask,
    )
    save_plots(
        output_dir, signal, baseline_signal, repaired_signal, codebook_ids, repaired_ids,
        content_mask, stride, not args.no_signal_plot,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
