#!/usr/bin/env python3
"""Decode ``<|bwav:N|>`` corpora back to nanopore signal arrays.

One input ``*.jsonl.gz`` produces one two-dimensional ``*.npy`` by default.
Rows in the output have exactly the same order as records in the input file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


BWAV_TOKEN = re.compile(r"<\|bwav:(\d+)\|>")


def parse_bwav_tokens(text: str) -> np.ndarray:
    """Extract codebook indices from a serialized bwav token string."""
    values = BWAV_TOKEN.findall(text)
    if not values:
        raise ValueError("the 'text' field contains no <|bwav:N|> tokens")
    return np.asarray(values, dtype=np.int64)


def iter_records(path: Path) -> Iterator[tuple[int, str, np.ndarray]]:
    """Yield ``(line_number, record_id, token_ids)`` from a JSONL gzip file."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = str(record.get("id", line_number))
                text = record["text"]
                if not isinstance(text, str):
                    raise TypeError("'text' is not a string")
                tokens = parse_bwav_tokens(text)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid record on line {line_number}: {exc}") from exc
            yield line_number, record_id, tokens


def inspect_input(path: Path) -> tuple[int, int]:
    """Return record count and the common token count, validating all rows."""
    count = 0
    token_count = -1
    for line_number, record_id, tokens in iter_records(path):
        if token_count < 0:
            token_count = len(tokens)
        elif len(tokens) != token_count:
            raise ValueError(
                f"{path}: record {record_id!r} on line {line_number} has {len(tokens)} "
                f"tokens, expected {token_count}; a rectangular .npy cannot store variable lengths"
            )
        count += 1
    if count == 0:
        raise ValueError(f"{path}: input contains no records")
    return count, token_count


def _select_device(requested: str | None) -> str:
    import torch

    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({requested}) but is not available")
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_hf_model_path(path: Path) -> Path:
    if (path / "config.json").is_file():
        return path
    run_config = path / "config.yaml"
    if not run_config.is_file():
        return path
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is needed when --model points to a Stage1 run directory") from exc
    with run_config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    save_folder = config.get("save_folder")
    if not save_folder:
        raise ValueError(f"save_folder is missing in {run_config}")
    resolved = Path(save_folder).expanduser()
    return resolved if resolved.is_absolute() else path / resolved


class SignalDecoder:
    """Uniform decoder interface for current HF codecs and legacy checkpoints."""

    def __init__(self, model_path: Path, device: str | None = None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load and run the signal decoder") from exc
        self.torch = torch
        self.device = _select_device(device)
        resolved = _resolve_hf_model_path(model_path.expanduser())
        if resolved.is_dir() and (resolved / "config.json").is_file():
            public_dir = Path(__file__).resolve().parents[2] / "training_public" / "stage1_tokenizer_train"
            sys.path.insert(0, str(public_dir))
            import modeling_pore_codec  # noqa: F401
            import modeling_pore_vq_codec  # noqa: F401
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                str(resolved), trust_remote_code=True
            ).to(self.device).eval()
            self._decode = self._decode_hf
        else:
            stage2_dir = Path(__file__).resolve().parent.parent / "stage2_BERT_Encoder"
            sys.path.insert(0, str(stage2_dir))
            from vqe_tokenizer import VQETokenizer

            self.model = VQETokenizer(model_ckpt=str(resolved), device=self.device)
            self._decode = self._decode_legacy

    def _decode_hf(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "decode_token"):
            raise AttributeError("loaded HF codec does not implement decode_token()")
        return self.model.decode_token(token_ids)

    def _decode_legacy(self, token_ids: torch.Tensor) -> torch.Tensor:
        result = self.model.decode_token_ids(token_ids, return_numpy=False)
        return result.unsqueeze(1) if result.ndim == 2 else result

    def decode(self, token_ids: np.ndarray, target_signal_length: int | None = None) -> np.ndarray:
        ids = self.torch.from_numpy(token_ids).long().to(self.device)
        with self.torch.inference_mode():
            signal = self._decode(ids)
        if isinstance(signal, (tuple, list)):
            signal = signal[0]
        signal = signal.detach().float().cpu()
        if signal.ndim == 3 and signal.shape[1] == 1:
            signal = signal[:, 0]
        if signal.ndim != 2:
            raise RuntimeError(f"decoder returned unexpected shape {tuple(signal.shape)}")
        result = signal.numpy()
        if target_signal_length is None:
            stride = getattr(self.model, "cnn_stride", None)
            if stride is None and hasattr(self.model, "model"):
                stride = getattr(self.model.model, "cnn_stride", None)
            if stride is not None:
                target_signal_length = token_ids.shape[1] * int(stride)
        if target_signal_length is not None:
            if result.shape[1] > target_signal_length:
                result = result[:, :target_signal_length]
            elif result.shape[1] < target_signal_length:
                result = np.pad(result, ((0, 0), (0, target_signal_length - result.shape[1])))
        return result


def decode_file(
    input_path: Path,
    output_path: Path,
    decoder: SignalDecoder,
    batch_size: int,
    overwrite: bool,
    signal_length: int | None = None,
) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path} (use --overwrite to replace it)")
    record_count, token_count = inspect_input(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = None
    batch: list[np.ndarray] = []
    row = 0
    for _, _, tokens in iter_records(input_path):
        batch.append(tokens)
        if len(batch) < batch_size and row + len(batch) < record_count:
            continue
        decoded = decoder.decode(np.stack(batch), target_signal_length=signal_length)
        if output is None:
            output = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32, shape=(record_count, decoded.shape[1])
            )
        elif decoded.shape[1] != output.shape[1]:
            raise RuntimeError("decoder returned different signal lengths between batches")
        output[row : row + len(batch)] = decoded
        row += len(batch)
        batch.clear()
        print(f"\r{input_path.name}: {row}/{record_count}", end="", flush=True)
    print()
    assert output is not None
    signal_length = int(output.shape[1])
    output.flush()
    del output
    return record_count, signal_length


def find_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl.gz"))
        if files:
            return files
        raise FileNotFoundError(f"no *.jsonl.gz files found in {path}")
    raise FileNotFoundError(path)


def output_name(input_path: Path) -> str:
    name = input_path.name
    return name[:-9] + ".npy" if name.endswith(".jsonl.gz") else name + ".npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="A .jsonl.gz file or directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="HF codec directory, Stage1 run, or legacy checkpoint")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:1, or cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--signal-length",
        type=int,
        default=None,
        help="Target points per reconstructed signal; default: token count times codec stride",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.signal_length is not None and args.signal_length < 1:
        raise ValueError("--signal-length must be at least 1")
    inputs = find_inputs(args.input.expanduser())
    decoder = SignalDecoder(args.model, args.device)
    for input_path in inputs:
        output_path = args.output_dir.expanduser() / output_name(input_path)
        rows, length = decode_file(
            input_path,
            output_path,
            decoder,
            args.batch_size,
            args.overwrite,
            args.signal_length,
        )
        print(f"saved {output_path} shape=({rows}, {length}) dtype=float32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
