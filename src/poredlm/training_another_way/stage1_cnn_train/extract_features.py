"""Extract frozen continuous CNN features into public-style memmap shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from dataset import PoreSignalDataset
from modeling_continuous_cnn import ContinuousCNNConfig, ContinuousSignalCNN


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "valid"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        model = ContinuousSignalCNN.from_pretrained(checkpoint_path)
    else:
        # Backward compatibility with the old single-file .pt checkpoints.
        model = ContinuousSignalCNN(ContinuousCNNConfig(**cfg.get("model", {})))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=True)
    model = model.to(device)
    model.eval()
    data_cfg = cfg["data"][args.split]
    dataset = PoreSignalDataset(data_cfg["paths"], chunk_size=data_cfg.get("chunk_size", 6000),
                                memmap_dtype=data_cfg.get("memmap_dtype", "float32"), buffer_size=0,
                                shuffle_buffer=False, repeat=False, seed=cfg.get("seed", 42))
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["training"]["device_micro_batch_size"],
                                         num_workers=0, pin_memory=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    suffix = f"_rank{rank:05d}" if world_size > 1 else ""
    data_path = out / f"features{suffix}.npy"
    index_path = out / f"features{suffix}.csv.gz"
    output_dtype = np.dtype(args.dtype)
    offset = 0; sample_count = 0; feature_length = None
    with data_path.open("wb") as data_handle, gzip.open(index_path, "wt", newline="", encoding="utf-8") as index_handle:
        writer = csv.writer(index_handle)
        for batch in tqdm(loader, desc=f"extract-{args.split}"):
            if args.max_samples is not None and sample_count >= args.max_samples: break
            signals = batch["signal"]
            if args.max_samples is not None:
                keep = min(signals.shape[0], args.max_samples - sample_count)
                signals = signals[:keep]; ids = batch["id"][:keep]
            else:
                ids = batch["id"]
            latent = model.encode(signals.to(device, non_blocking=True)).transpose(1, 2).cpu().numpy()
            if feature_length is None: feature_length = int(latent.shape[1])
            if latent.shape[1] != feature_length: raise ValueError("CNN produced variable feature lengths.")
            latent = latent.astype(output_dtype, copy=False)
            for row, sample_id in zip(latent, ids.tolist()):
                flat = row.reshape(-1)
                flat.tofile(data_handle)
                writer.writerow([offset, offset + flat.size, int(sample_id)])
                offset += flat.size; sample_count += 1
    metadata = {"feature_dim": int(model.hidden_size), "feature_length": feature_length,
                "cnn_stride": int(model.stride), "dtype": args.dtype,
                "num_samples": sample_count, "rank": rank, "world_size": world_size}
    (out / f"metadata{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
