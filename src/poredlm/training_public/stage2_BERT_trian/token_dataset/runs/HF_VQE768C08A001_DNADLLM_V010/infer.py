"""Tokenize raw nanopore signal with the V010 BlockMLP VQ tokenizer."""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

# This import registers model_type="pore_block_mlp_vq" before loading.
import modeling_pore_block_mlp_vq  # noqa: F401,E402
from dataset import PoreSignalDataset  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def flush_chunk(save_folder, output_prefix, rank, chunk_idx, token_buffer, meta_buffer, token_dtype):
    if not token_buffer:
        return
    stem = f"{output_prefix}_rank{rank}_{chunk_idx:04d}"
    np.asarray(token_buffer, dtype=token_dtype).tofile(Path(save_folder) / f"{stem}.npy")
    with gzip.open(Path(save_folder) / f"{stem}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        pd.DataFrame(meta_buffer).to_csv(handle, index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="V010 BlockMLP signal tokenization")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    dolma_cfg = cfg["dolma_spec"]

    input_paths = dataset_cfg["input_data_paths"]
    save_folder = os.environ.get("TOKENIZE_OUTPUT_DIR", dataset_cfg["save_folder"])
    model_dir = os.environ.get("TOKENIZER_CHECKPOINT", model_cfg["model_dir"])
    logic_chunk_size = int(dataset_cfg.get("logic_chunk_size", 6000))
    batch_size = int(dataset_cfg.get("batch_size", 1))
    num_workers = int(dataset_cfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 6198))

    if bool(dataset_cfg.get("feature_extraction", False)):
        raise ValueError("V010 BlockMLP expects raw signal; set feature_extraction=false.")

    dtype_name = str(dolma_cfg.get("memmap_dtype", "uint32"))
    dtype_map = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32}
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dolma_spec.memmap_dtype: {dtype_name}")
    token_dtype = np.dtype(dtype_map[dtype_name])
    token_max = np.iinfo(token_dtype).max
    token_offset = int(dolma_cfg.get("token_offset", 128))
    bos_token_id = int(dolma_cfg.get("bos_token_id", 2))
    eos_token_id = int(dolma_cfg.get("eos_token_id", 3))
    items_per_file = int(dolma_cfg.get("items_per_file", 262144))
    output_prefix = str(dataset_cfg.get("output_prefix", "pore_tokens"))

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device)
    model.eval()
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "pore_block_mlp_vq":
        raise ValueError(f"Expected BlockMLP checkpoint, got model_type={model_type!r}")

    dataset = PoreSignalDataset(
        shard_paths=input_paths,
        logic_chunk_size=logic_chunk_size,
        buffer_size=int(dataset_cfg.get("buffer_size", 0)),
        memmap_dtype=str(dataset_cfg.get("memmap_dtype", "float32")),
        shuffle_buffer=bool(dataset_cfg.get("shuffle_buffer", False)),
        rank=rank,
        world_size=world_size,
        is_repeat=False,
        seed=seed,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(dataset_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(dataset_cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(dataset_cfg.get("persistent_workers", False))
    dataloader = DataLoader(dataset, **loader_kwargs)

    token_buffer, meta_buffer = [], []
    chunk_idx = 0
    item_count = 0
    current_pointer = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"GPU {rank}", position=rank):
            signals = batch["signal"].to(device=device, dtype=torch.float32, non_blocking=True)
            if signals.ndim == 2:
                signals = signals.unsqueeze(1)
            token_ids = model.encode_signal(signals).detach().cpu().numpy()
            batch_ids = batch.get("id")

            for index, raw_tokens in enumerate(token_ids):
                shifted = raw_tokens.reshape(-1).astype(np.int64) + token_offset
                if shifted.size and (shifted.min() < 0 or shifted.max() > token_max):
                    raise ValueError(f"Shifted token id exceeds {dtype_name}: {int(shifted.max())}")
                sequence = [bos_token_id, *shifted.tolist(), eos_token_id]
                start = current_pointer
                end = start + len(sequence)
                token_buffer.extend(sequence)
                raw_id = str(batch_ids[index].item()) if batch_ids is not None else f"unknown_{item_count}"
                meta_buffer.append([start, end, len(sequence), "token", raw_id, raw_id])
                current_pointer = end
                item_count += 1

                if item_count % items_per_file == 0:
                    flush_chunk(save_folder, output_prefix, rank, chunk_idx, token_buffer, meta_buffer, token_dtype)
                    chunk_idx += 1
                    token_buffer, meta_buffer = [], []
                    current_pointer = 0

    flush_chunk(save_folder, output_prefix, rank, chunk_idx, token_buffer, meta_buffer, token_dtype)
    print(f"[Rank {rank}] finished: {item_count} samples -> {save_folder}", flush=True)


if __name__ == "__main__":
    main()
