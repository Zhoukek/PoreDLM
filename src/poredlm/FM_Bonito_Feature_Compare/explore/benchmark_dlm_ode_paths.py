#!/usr/bin/env python3
"""Benchmark the current DLM ODE path and its frozen-context equivalent.

The benchmark uses real 1,500-code chunks and the exact one-step ODE
parameters used in the Stage-4 experiments.  It separates the time spent in
the context encoder from ODE refinement, so that the benefit of a compatible
context-hidden cache can be measured without running a full training job.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/mnt/zzbnew/poregpt/models/HF_RSQ741C12V523_DNADLLM_V011/"
            "hf_dlm_elf_mixed_with_cls_token_step65000-unsharded"
        ),
    )
    parser.add_argument(
        "--tokens-jsonl-gz",
        type=Path,
        default=Path("../data/tokens/eval00001_chunks.jsonl.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("dlm_ode_path_benchmark.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=1)
    parser.add_argument("--ode-start-t", type=float, default=0.97)
    parser.add_argument("--ode-self-cond-cfg-scale", type=float, default=0.5)
    return parser.parse_args()


def read_input_ids(path: Path, batch_size: int) -> list[list[int]]:
    token_re = re.compile(r"<\|bwav:(\d+)\|>")
    rows: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            codes = [int(value) for value in token_re.findall(record["text"])]
            if len(codes) != 1500:
                raise ValueError(f"Expected 1,500 signal codes, received {len(codes)}")
            rows.append([5, 2, *(code + 128 for code in codes), 3])
            if len(rows) == batch_size:
                return rows
    raise ValueError(f"Only found {len(rows)} rows; batch size is {batch_size}")


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; refusing a CPU measurement.")
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--batch-size and --iterations must be positive; --warmup must be non-negative.")
    if not (0.0 < args.ode_start_t <= 1.0):
        raise ValueError("--ode-start-t must be in (0, 1].")

    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    elf_src = args.model_dir / "ELF-pytorch-port" / "src"
    if elf_src.is_dir():
        sys.path.insert(0, str(elf_src))
    else:
        configured_src = config.get("elf_src_path")
        if configured_src and Path(configured_src).is_dir():
            sys.path.insert(0, configured_src)

    device = torch.device("cuda")
    input_ids = torch.tensor(read_input_ids(args.tokens_jsonl_gz, args.batch_size), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    print(f"Loading DLM checkpoint: {args.model_dir}", flush=True)
    model = AutoModel.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True, torch_dtype="auto"
    ).to(device).eval()

    def sync() -> None:
        torch.cuda.synchronize(device)

    def elapsed_ms(callable_) -> float:
        sync()
        start = time.perf_counter()
        callable_()
        sync()
        return (time.perf_counter() - start) * 1000.0

    def full_ode_call():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_ode_hidden=True,
                ode_steps=args.ode_steps,
                ode_start_t=args.ode_start_t,
                ode_self_cond_cfg_scale=args.ode_self_cond_cfg_scale,
            )["ode_hidden_state"]

    def context_call():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return model.context_encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state

    # Materialize one context feature set before the cache-equivalent timing.
    context = context_call().to(dtype=next(model.elf_denoiser.parameters()).dtype)

    def ode_from_context_call():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            return model.ode_from_context_hidden(
                context,
                attention_mask=attention_mask,
                ode_steps=args.ode_steps,
                ode_start_t=args.ode_start_t,
                self_cond_cfg_scale=args.ode_self_cond_cfg_scale,
            )

    for _ in range(args.warmup):
        full_ode_call()
        context_call()
        ode_from_context_call()
    sync()

    full_times = [elapsed_ms(full_ode_call) for _ in range(args.iterations)]
    context_times = [elapsed_ms(context_call) for _ in range(args.iterations)]
    ode_from_context_times = [elapsed_ms(ode_from_context_call) for _ in range(args.iterations)]

    full_output = full_ode_call()
    cached_output = ode_from_context_call()
    max_abs_difference = float((full_output.float() - cached_output.float()).abs().max().cpu())
    import numpy as np

    full_median = float(np.median(full_times))
    cached_median = float(np.median(ode_from_context_times))
    result = {
        "model_dir": str(args.model_dir),
        "tokens_jsonl_gz": str(args.tokens_jsonl_gz),
        "device": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "sequence_length_with_special_tokens": int(input_ids.shape[1]),
        "precision": "torch.autocast(cuda, float16), inference_mode",
        "ode_parameters": {
            "ode_steps": args.ode_steps,
            "ode_start_t": args.ode_start_t,
            "ode_self_cond_cfg_scale": args.ode_self_cond_cfg_scale,
        },
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "milliseconds_per_batch": {
            "direct_dlm_ode_median": full_median,
            "context_encoder_only_median": float(np.median(context_times)),
            "ode_from_context_median": cached_median,
            "direct_dlm_ode_values": full_times,
            "context_encoder_only_values": context_times,
            "ode_from_context_values": ode_from_context_times,
        },
        "cache_equivalent_speedup_direct_over_ode_from_context": full_median / cached_median,
        "full_vs_context_plus_ode_ratio": full_median / (
            float(np.median(context_times)) + cached_median
        ),
        "full_vs_cached_ode_max_abs_difference": max_abs_difference,
        "interpretation": (
            "The ode_from_context measurement represents computation after reading a compatible frozen "
            "context-hidden cache. Disk transfer and TCN/CTC head time are excluded from both paths."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
