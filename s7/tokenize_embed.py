#!/usr/bin/env python3
"""Tokenize Apple-normalized signal windows and embed them with the supplied PoreGPT model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_OFFSET = 128
CODEBOOK_SIZE = 2401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def patch_torch_transformers_compatibility(torch: Any) -> None:
    if (
        hasattr(torch.utils, "_pytree")
        and not hasattr(torch.utils._pytree, "register_pytree_node")
        and hasattr(torch.utils._pytree, "_register_pytree_node")
    ):
        def register_pytree_node(node_type, flatten_fn, unflatten_fn, **_: Any):
            return torch.utils._pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

        torch.utils._pytree.register_pytree_node = register_pytree_node
    try:
        import torch.distributed.tensor as torch_dtensor

        for name in ("Replicate", "Shard", "Partial"):
            if not hasattr(torch_dtensor, name):
                setattr(torch_dtensor, name, type(name, (), {}))
    except Exception:
        pass
    if hasattr(torch, "amp") and not hasattr(torch.amp, "GradScaler"):
        torch.amp.GradScaler = torch.cuda.amp.GradScaler

    class FakeOnnxExporter(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            value = type(name, (), {})
            setattr(self, name, value)
            return value

    exporter = FakeOnnxExporter("torch.onnx._internal.exporter")
    for name in ("ExportOptions", "ExportOutput", "ExportOutputSerializer", "ONNXProgram"):
        setattr(exporter, name, type(name, (), {}))
    sys.modules["torch.onnx._internal.exporter"] = exporter

    # This environment has torch 2.1 binaries alongside newer torch.nn.attention
    # Python files. Transformers only imports Flex Attention for registration; this
    # model uses eager/SDPA attention, so a dormant compatibility stub is sufficient.
    attention_module = types.ModuleType("torch.nn.attention")
    attention_module.__path__ = []
    flex_module = types.ModuleType("torch.nn.attention.flex_attention")

    def unavailable_flex_attention(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Flex Attention is unavailable with torch 2.1")

    flex_module.flex_attention = unavailable_flex_attention
    attention_module.flex_attention = flex_module
    sys.modules["torch.nn.attention"] = attention_module
    sys.modules["torch.nn.attention.flex_attention"] = flex_module
    torch.nn.attention = attention_module


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(out_dir / "hf_cache"))
    os.environ.setdefault("HF_MODULES_CACHE", str(out_dir / "hf_cache" / "modules"))
    model_root = Path(args.model_dir)
    encoder_dir = model_root / "encoder"
    base_dir = model_root / "base"

    import torch

    patch_torch_transformers_compatibility(torch)
    from transformers import AutoModel, AutoModelForCausalLM

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    records = [json.loads(line) for line in Path(args.windows).open() if line.strip()]
    if not records:
        raise RuntimeError("No signal windows were loaded")
    if not all(record["normal_mode"] == "apple" for record in records):
        raise RuntimeError("At least one window is not marked apple-normalized")
    if not all(int(record["signal_base_shift"]) == -4 for record in records):
        raise RuntimeError("At least one window does not use signal_base_shift=-4")

    print(json.dumps({"stage": "load_encoder", "path": str(encoder_dir)}), flush=True)
    encoder = AutoModel.from_pretrained(str(encoder_dir), trust_remote_code=True)
    encoder.to(device).eval()

    raw_tokens: list[np.ndarray] = []
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            # Input is already Apple-normalized; do not call AutoFeatureExtractor here.
            signal = torch.as_tensor(record["signal"], dtype=torch.float32, device=device).view(1, 1, -1)
            token_ids = encoder.encode_signal(signal, layer=1).squeeze(0)
            tokens = token_ids.detach().cpu().numpy().astype(np.int64, copy=False)
            if tokens.size == 0 or tokens.min() < 0 or tokens.max() >= CODEBOOK_SIZE:
                raise ValueError(f"Invalid encoder token range for record {index}: {tokens}")
            raw_tokens.append(tokens)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    max_length = max(len(tokens) for tokens in raw_tokens)
    input_ids = torch.full((len(records), max_length), 1, dtype=torch.long)
    attention_mask = torch.zeros((len(records), max_length), dtype=torch.long)
    for index, tokens in enumerate(raw_tokens):
        length = len(tokens)
        input_ids[index, :length] = torch.from_numpy(tokens + TOKEN_OFFSET)
        attention_mask[index, :length] = 1
    if input_ids[attention_mask.bool()].min() < TOKEN_OFFSET or input_ids.max() >= 2560:
        raise ValueError("Shifted model token IDs fall outside the OLMo vocabulary")

    print(
        json.dumps(
            {
                "stage": "load_base",
                "path": str(base_dir),
                "dtype": str(dtype),
                "records": len(records),
                "max_tokens": max_length,
            }
        ),
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.to(device).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    pooled: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            end = min(start + args.batch_size, len(records))
            ids = input_ids[start:end].to(device)
            mask = attention_mask[start:end].to(device)
            outputs = base(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1].float()
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            mean_hidden = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            pooled.append(mean_hidden.cpu().numpy())
            print(json.dumps({"stage": "embed", "completed": end, "total": len(records)}), flush=True)
    embeddings = np.concatenate(pooled, axis=0).astype(np.float32, copy=False)

    token_out = out_dir / "signal_windows.tokens.jsonl"
    with token_out.open("w") as handle:
        for record, tokens in zip(records, raw_tokens):
            row = {key: value for key, value in record.items() if key != "signal"}
            row["encoder_layer"] = 1
            row["raw_signal_tokens"] = tokens.tolist()
            row["model_input_ids"] = (tokens + TOKEN_OFFSET).tolist()
            row["token_count"] = int(tokens.size)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    np.savez_compressed(
        out_dir / "signal_windows.embeddings.npz",
        embeddings=embeddings,
        labels=np.asarray([record["label"] for record in records], dtype=np.int8),
        datasets=np.asarray([record["dataset"] for record in records]),
        site_ids=np.asarray([record["site_id"] for record in records]),
        read_ids=np.asarray([record["read_id"] for record in records]),
        site_pos0=np.asarray([record["site_pos0"] for record in records], dtype=np.int64),
        strands=np.asarray([record["strand"] for record in records]),
        token_counts=np.asarray([len(tokens) for tokens in raw_tokens], dtype=np.int32),
        signal_lengths=np.asarray([record["signal_len"] for record in records], dtype=np.int32),
    )
    summary = {
        "model_dir": str(model_root),
        "encoder_layer": 1,
        "encoder_codebook_size": CODEBOOK_SIZE,
        "model_token_offset": TOKEN_OFFSET,
        "embedding_source": "mean-pooled final hidden state of 24-layer OLMo2 base model",
        "embedding_dim": int(embeddings.shape[1]),
        "records": len(records),
        "token_count_min": min(len(tokens) for tokens in raw_tokens),
        "token_count_max": max(len(tokens) for tokens in raw_tokens),
        "token_count_mean": float(np.mean([len(tokens) for tokens in raw_tokens])),
        "raw_token_min": int(min(tokens.min() for tokens in raw_tokens)),
        "raw_token_max": int(max(tokens.max() for tokens in raw_tokens)),
        "apple_normalization_reapplied": False,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
    }
    (out_dir / "token_embedding_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
