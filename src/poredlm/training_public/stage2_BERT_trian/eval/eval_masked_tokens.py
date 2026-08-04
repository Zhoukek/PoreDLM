"""Evaluate Stage2 masked-token prediction with the exact current FlowMap input path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModel


HERE = Path(__file__).resolve().parent
STAGE2_DIR = HERE.parent
TOKEN_DATASET_DIR = STAGE2_DIR / "token_dataset"
for import_path in (STAGE2_DIR, TOKEN_DATASET_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import modeling_pore_vq_codec  # noqa: E402,F401
from flowmap import FlowMapDataset  # noqa: E402
from modeling_stage2_bert import Stage2MaskedSignalLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read current FlowMap token shards and evaluate Stage2 BERT masked-token recovery."
    )
    parser.add_argument("--bert", required=True, help="Stage2MaskedSignalLM checkpoint directory")
    parser.add_argument("--codec", required=True, help="PoreVQCodec tokenizer checkpoint directory")
    parser.add_argument(
        "--training-config",
        default=str(STAGE2_DIR / "runs" / "test" / "train_config.yaml"),
        help="Config that defines the current data/model pipeline",
    )
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--data-dir", default=None, help="Optional override for data.<split>_dir")
    parser.add_argument("--sample-index", type=int, default=0, help="Skip this many FlowMap samples")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to evaluate")
    parser.add_argument("--mask-mode", choices=("contiguous", "random"), default="contiguous")
    parser.add_argument("--mask-token-start", type=int, default=None, help="Content index; random when omitted/<0")
    parser.add_argument("--mask-token-length", type=int, default=4)
    parser.add_argument(
        "--plot-context-tokens",
        type=int,
        default=20,
        help="For contiguous masks, show this many context tokens on each side; <0 shows all",
    )
    parser.add_argument("--mask-probability", type=float, default=None, help="Defaults to model.mask_probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available")
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "data" not in config or "model" not in config:
        raise ValueError(f"Expected data/model sections in {path}")
    return config


def build_flowmap_dataset(
    config: dict[str, Any], split: str, data_dir_override: str | None, max_samples: int
) -> FlowMapDataset:
    data_cfg = config["data"]
    paths = [data_dir_override] if data_dir_override else data_cfg.get(f"{split}_paths")
    if paths is None:
        data_dir = data_cfg.get(f"{split}_dir")
        paths = [data_dir] if data_dir else None
    if not paths:
        raise ValueError(f"Missing data.{split}_paths/data.{split}_dir and no --data-dir override")
    return FlowMapDataset(
        shard_paths=paths,
        buffer_size=0,  # Evaluation order stays deterministic and does not fill the training shuffle buffer.
        memmap_dtype=str(data_cfg.get("token_dtype", data_cfg.get("memmap_dtype", "uint32"))),
        shuffle_buffer=False,
        rank=0,
        world_size=1,
        is_repeat=False,
        seed=int(config.get("reproducibility", {}).get("seed", config.get("seed", 42))),
        data_name=str(data_cfg.get("data_name", "data")),
        memmap_cache_capacity=int(data_cfg.get("memmap_cache_capacity", 16)),
        max_samples=max_samples,
        verbose=bool(data_cfg.get("verbose", False)),
    )


def normalize_flowmap_sample(
    sample: torch.Tensor, data_cfg: dict[str, Any], model_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror train_stage2_bert_memmap.normalize_flowmap_sample exactly."""

    raw = torch.as_tensor(sample, dtype=torch.long).flatten()
    pad_id = int(model_cfg.get("pad_token_id", 0))
    bos_id = int(model_cfg.get("bos_token_id", 2))
    eos_id = int(model_cfg.get("eos_token_id", 3))
    cls_id = int(model_cfg.get("cls_token_id", 4))
    token_offset = int(data_cfg.get("token_offset", 128))
    max_length = int(model_cfg.get("max_position_embeddings", 1280))
    input_has_special = bool(data_cfg.get("input_has_special_tokens", True))
    tokens_are_shifted = bool(data_cfg.get("tokens_are_shifted", True))
    add_cls = bool(data_cfg.get("add_cls_token", True))
    strict_special = bool(data_cfg.get("strict_special_tokens", True))

    if raw.numel() == 0:
        content = raw
    elif input_has_special:
        if raw.numel() < 2:
            raise ValueError("FlowMap sample is too short to contain BOS/EOS")
        if strict_special and (int(raw[0]) != bos_id or int(raw[-1]) != eos_id):
            raise ValueError(
                f"FlowMap boundary mismatch: expected BOS/EOS={bos_id}/{eos_id}, "
                f"got {int(raw[0])}/{int(raw[-1])}"
            )
        content = raw[1:-1]
    else:
        content = raw

    if not tokens_are_shifted:
        content = content + token_offset
    reserve = 2 + int(add_cls)
    if max_length < reserve:
        raise ValueError(f"max_position_embeddings={max_length} is too short for special tokens")
    content = content[: max_length - reserve]
    pieces = []
    if add_cls:
        pieces.append(torch.tensor([cls_id], dtype=torch.long))
    pieces.extend((torch.tensor([bos_id]), content, torch.tensor([eos_id])))
    sequence = torch.cat(pieces)
    valid_length = int(sequence.numel())
    if valid_length < max_length:
        sequence = F.pad(sequence, (0, max_length - valid_length), value=pad_id)
    attention_mask = (sequence != pad_id).long()
    return sequence, attention_mask


def iter_selected_samples(
    dataset: FlowMapDataset, data_name: str, start: int, count: int
) -> Iterator[tuple[int, int | None, torch.Tensor]]:
    if start < 0 or count < 1:
        raise ValueError("sample-index must be >=0 and num-samples must be >=1")
    found = 0
    for stream_index, item in enumerate(dataset):
        if stream_index < start:
            continue
        if data_name not in item:
            raise KeyError(f"FlowMap item has no {data_name!r}; keys={list(item)}")
        sample_id = int(item["id"]) if "id" in item else None
        yield stream_index, sample_id, item[data_name]
        found += 1
        if found >= count:
            return
    if found == 0:
        raise IndexError(f"No sample exists at/after sample-index={start}")


def choose_content_mask(num_content: int, args: argparse.Namespace, probability: float, seed: int) -> np.ndarray:
    if num_content < 1:
        raise ValueError("Selected sequence contains no content tokens")
    rng = np.random.default_rng(seed)
    mask = np.zeros(num_content, dtype=bool)
    if args.mask_mode == "contiguous":
        length = min(max(1, args.mask_token_length), num_content)
        if args.mask_token_start is None or args.mask_token_start < 0:
            start = int(rng.integers(0, num_content - length + 1))
        else:
            start = args.mask_token_start
        if start + length > num_content:
            raise ValueError(f"Mask [{start}, {start + length}) exceeds {num_content} content tokens")
        mask[start : start + length] = True
    else:
        if not 0.0 < probability <= 1.0:
            raise ValueError("mask-probability must be in (0, 1]")
        mask = rng.random(num_content) < probability
        if not mask.any():
            mask[int(rng.integers(num_content))] = True
    return mask


def save_token_plot(
    path: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    sample_id: int | None,
    mask_mode: str,
    context_tokens: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked_indices = np.flatnonzero(mask)
    if mask_mode == "contiguous" and context_tokens >= 0 and masked_indices.size:
        plot_start = max(0, int(masked_indices[0]) - context_tokens)
        plot_end = min(targets.size, int(masked_indices[-1]) + context_tokens + 1)
    else:
        plot_start, plot_end = 0, targets.size

    x = np.arange(plot_start, plot_end)
    plot_targets = targets[plot_start:plot_end]
    plot_predictions = predictions[plot_start:plot_end]
    plot_mask = mask[plot_start:plot_end]
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(x, plot_targets, ".-", ms=4, lw=0.8, label="target shifted token id")
    ax.plot(x, plot_predictions, ".-", ms=4, lw=0.8, alpha=0.8, label="BERT repaired token id")
    ax.scatter(
        x[plot_mask],
        plot_predictions[plot_mask],
        c="tab:red",
        s=32,
        label="BERT prediction at masked position",
        zorder=3,
    )
    if mask_mode == "contiguous" and masked_indices.size:
        ax.axvspan(
            float(masked_indices[0]) - 0.5,
            float(masked_indices[-1]) + 0.5,
            color="gold",
            alpha=0.16,
            label="masked region",
        )
    ax.set(
        xlabel="content token index",
        ylabel="BERT vocabulary id",
        title=(
            f"FlowMap masked-token reconstruction (sample id={sample_id}, "
            f"masked={int(masked_indices[0])}:{int(masked_indices[-1]) + 1})"
            if mask_mode == "contiguous" and masked_indices.size
            else f"FlowMap masked-token reconstruction (sample id={sample_id}, masked_count={masked_indices.size})"
        ),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def decode_codebook_ids(codec: torch.nn.Module, codebook_ids: np.ndarray, device: torch.device) -> np.ndarray:
    ids = torch.as_tensor(codebook_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        waveform = codec.decode_token(ids).squeeze().float().cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def save_waveform_plot(
    path: Path,
    target_waveform: np.ndarray,
    repaired_waveform: np.ndarray,
    content_mask: np.ndarray,
    stride: int,
    context_tokens: int,
    sample_id: int | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked_indices = np.flatnonzero(content_mask)
    if masked_indices.size and context_tokens >= 0:
        sample_start = max(0, (int(masked_indices[0]) - context_tokens) * stride)
        sample_end = min(
            target_waveform.size,
            (int(masked_indices[-1]) + context_tokens + 1) * stride,
        )
    else:
        sample_start, sample_end = 0, target_waveform.size
    x = np.arange(sample_start, sample_end)
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(x, target_waveform[sample_start:sample_end], lw=1.0, label="original-token reconstruction")
    ax.plot(
        x,
        repaired_waveform[sample_start:sample_end],
        lw=1.0,
        alpha=0.85,
        label="BERT-token reconstruction",
    )
    if masked_indices.size:
        ax.axvspan(
            int(masked_indices[0]) * stride,
            min((int(masked_indices[-1]) + 1) * stride, target_waveform.size),
            color="gold",
            alpha=0.16,
            label="masked token region",
        )
    ax.set(
        xlabel="reconstructed waveform sample",
        ylabel="signal value",
        title=f"Tokenizer-decoded waveform comparison (sample id={sample_id})",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = load_config(Path(args.training_config).expanduser().resolve())
    data_cfg, model_cfg = config["data"], config["model"]
    if str(data_cfg.get("dataset_type", "flowmap")).lower() != "flowmap":
        raise ValueError("This evaluator is for the current data.dataset_type=flowmap pipeline")

    dataset = build_flowmap_dataset(config, args.split, args.data_dir, args.sample_index + args.num_samples)
    bert = Stage2MaskedSignalLM.from_pretrained(Path(args.bert).expanduser().resolve()).to(device).eval()
    codec = AutoModel.from_pretrained(args.codec, trust_remote_code=True).to(device).eval()
    mask_token_id = int(model_cfg.get("mask_token_id", bert.config.mask_token_id))
    pad_id = int(model_cfg.get("pad_token_id", bert.config.pad_token_id))
    add_cls = bool(data_cfg.get("add_cls_token", True))
    content_start = 2 if add_cls else 1
    probability = float(
        args.mask_probability if args.mask_probability is not None else model_cfg.get("mask_probability", 0.15)
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_results: list[dict[str, Any]] = []
    total_correct = total_masked = total_invalid = 0
    first_arrays: dict[str, np.ndarray] | None = None

    data_name = str(data_cfg.get("data_name", "data"))
    for ordinal, (stream_index, sample_id, raw) in enumerate(
        iter_selected_samples(dataset, data_name, args.sample_index, args.num_samples)
    ):
        input_ids, attention_mask = normalize_flowmap_sample(raw, data_cfg, model_cfg)
        valid_length = int(attention_mask.sum())
        num_content = valid_length - content_start - 1  # final valid token is EOS
        content_mask = choose_content_mask(num_content, args, probability, args.seed + ordinal)
        sequence_mask = np.zeros(input_ids.numel(), dtype=bool)
        sequence_mask[content_start : content_start + num_content] = content_mask
        corrupted = input_ids.clone()
        corrupted[torch.from_numpy(sequence_mask)] = mask_token_id

        with torch.inference_mode():
            logits = bert(
                input_ids=corrupted.unsqueeze(0).to(device),
                attention_mask=attention_mask.unsqueeze(0).to(device),
            ).logits[0].cpu()
        predicted = logits[torch.from_numpy(sequence_mask)].argmax(dim=-1)
        targets = input_ids[torch.from_numpy(sequence_mask)]
        correct = int((predicted == targets).sum())
        masked_count = int(targets.numel())
        low = int(model_cfg.get("random_token_min_id", data_cfg.get("token_offset", 128)))
        high = int(model_cfg.get("random_token_max_id", model_cfg["vocab_size"]))
        invalid = int(((predicted < low) | (predicted >= high)).sum())
        total_correct += correct
        total_masked += masked_count
        total_invalid += invalid

        repaired = input_ids.clone()
        repaired[torch.from_numpy(sequence_mask)] = predicted
        result = {
            "stream_index": stream_index,
            "sample_id": sample_id,
            "valid_sequence_length": valid_length,
            "num_content_tokens": num_content,
            "num_masked_tokens": masked_count,
            "masked_token_accuracy": correct / masked_count,
            "invalid_prediction_rate": invalid / masked_count,
        }
        sample_results.append(result)
        if first_arrays is None:
            content_slice = slice(content_start, content_start + num_content)
            target_content = input_ids[content_slice].numpy()
            repaired_content = repaired[content_slice].numpy()
            # Decoder accepts codebook ids, whereas Stage2 operates in the shifted vocabulary.
            token_offset = int(data_cfg.get("token_offset", 128))
            low = token_offset
            high = token_offset + int(codec.config.codebook_size)
            decode_content = repaired_content.copy()
            invalid_decode_positions = (decode_content < low) | (decode_content >= high)
            if invalid_decode_positions.any():
                # Keep MLM metrics as true full-vocabulary top-1, but use the best legal
                # codebook prediction where an invalid special token cannot be decoded.
                content_sequence_mask = torch.from_numpy(sequence_mask)
                legal_predictions = logits[content_sequence_mask, low:high].argmax(dim=-1).numpy() + low
                decode_content[content_mask] = legal_predictions
            target_codebook_ids = target_content - token_offset
            repaired_codebook_ids = decode_content - token_offset
            codebook_size = int(codec.config.codebook_size)
            if (
                target_codebook_ids.min() < 0
                or target_codebook_ids.max() >= codebook_size
                or repaired_codebook_ids.min() < 0
                or repaired_codebook_ids.max() >= codebook_size
            ):
                raise ValueError(
                    "Cannot decode tokens: shifted content ids do not match "
                    f"token_offset={token_offset} and codec codebook_size={codebook_size}"
                )
            target_waveform = decode_codebook_ids(codec, target_codebook_ids, device)
            repaired_waveform = decode_codebook_ids(codec, repaired_codebook_ids, device)
            stride = int(getattr(codec, "cnn_stride", getattr(codec, "stride", 1)))
            first_arrays = {
                "raw_flowmap_tokens": torch.as_tensor(raw).cpu().numpy(),
                "normalized_input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
                "corrupted_input_ids": corrupted.numpy(),
                "repaired_input_ids": repaired.numpy(),
                "target_content_ids": target_content,
                "repaired_content_ids": repaired_content,
                "decoder_repaired_content_ids": decode_content,
                "target_codebook_ids": target_codebook_ids,
                "repaired_codebook_ids": repaired_codebook_ids,
                "original_token_waveform": target_waveform,
                "bert_token_waveform": repaired_waveform,
                "masked_content_positions": content_mask,
                "masked_sequence_positions": sequence_mask,
            }
            save_token_plot(
                output_dir / "token_comparison.png",
                first_arrays["target_content_ids"],
                first_arrays["repaired_content_ids"],
                content_mask,
                sample_id,
                args.mask_mode,
                args.plot_context_tokens,
            )
            save_waveform_plot(
                output_dir / "waveform_comparison.png",
                target_waveform,
                repaired_waveform,
                content_mask,
                stride,
                args.plot_context_tokens if args.mask_mode == "contiguous" else -1,
                sample_id,
            )

    metrics = {
        "dataset_type": "flowmap",
        "split": args.split,
        "data_paths": [args.data_dir] if args.data_dir else data_cfg.get(f"{args.split}_paths", [data_cfg.get(f"{args.split}_dir")]),
        "sample_index": args.sample_index,
        "num_samples_evaluated": len(sample_results),
        "mask_mode": args.mask_mode,
        "masked_token_accuracy": total_correct / total_masked,
        "invalid_prediction_rate": total_invalid / total_masked,
        "total_masked_tokens": total_masked,
        "token_offset": int(data_cfg.get("token_offset", 128)),
        "input_has_special_tokens": bool(data_cfg.get("input_has_special_tokens", True)),
        "tokens_are_shifted": bool(data_cfg.get("tokens_are_shifted", True)),
        "add_cls_token": add_cls,
        "samples": sample_results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if first_arrays is not None:
        np.savez_compressed(output_dir / "first_sample_result.npz", **first_arrays)
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
