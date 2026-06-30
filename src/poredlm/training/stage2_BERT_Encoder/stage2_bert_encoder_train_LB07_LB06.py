"""Train a BERT encoder for Stage 2 representation learning by optimizer step."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from dataset import (
    Stage2Batch,
    build_bwav_vocab_lookup,
    load_tokenizer_vocab,
    parse_bwav_token_text,
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_number(value: int) -> str:
    """Format large integer counts for logs."""

    return f"{value:,}"


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


class MaskedSignalLM(nn.Module):
    """Jiaheng's TransformerEncoder masked-token LM, with configurable vocab size."""

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(pos)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return self.lm_head(self.norm(x))


def build_masked_signal_lm(config: dict[str, Any]) -> MaskedSignalLM:
    model_cfg = config["model"]
    return MaskedSignalLM(
        max_seq_len=int(model_cfg.get("max_position_embeddings", model_cfg.get("max_seq_len", 1024))),
        d_model=int(model_cfg.get("d_model", model_cfg.get("hidden_size", 512))),
        layers=int(model_cfg.get("layers", model_cfg.get("num_hidden_layers", 8))),
        heads=int(model_cfg.get("heads", model_cfg.get("num_attention_heads", 8))),
        dropout=float(model_cfg.get("dropout", model_cfg.get("hidden_dropout_prob", 0.1))),
        vocab_size=int(model_cfg.get("vocab_size", 2056)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
    )


def print_startup_summary(
    accelerator: Accelerator,
    config: dict[str, Any],
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader | None,
    seed: int,
) -> None:
    """Print model, data, and runtime information before training starts."""

    if not accelerator.is_main_process:
        return

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset if valid_loader is not None else None
    total_params, trainable_params = count_parameters(model)

    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    device_micro_batch_size = int(training_cfg.get("device_micro_batch_size", 8))
    effective_global_batch_size = (
        device_micro_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    train_files = getattr(train_dataset, "files", [])
    valid_files = getattr(valid_dataset, "files", []) if valid_dataset is not None else []
    train_line_counts = getattr(train_dataset, "file_line_counts", [])
    valid_line_counts = getattr(valid_dataset, "file_line_counts", []) if valid_dataset is not None else []
    train_samples = len(train_dataset) if hasattr(train_dataset, "__len__") else None
    valid_samples = len(valid_dataset) if valid_dataset is not None and hasattr(valid_dataset, "__len__") else None

    print("\n" + "=" * 80)
    print("Starting Stage 2 BERT Encoder Step Training")
    print("=" * 80)
    print(
        pformat(
            {
                "seed": seed,
                "distributed": {
                    "num_processes": accelerator.num_processes,
                    "process_index": accelerator.process_index,
                    "local_process_index": accelerator.local_process_index,
                    "device": str(accelerator.device),
                    "mixed_precision": accelerator.mixed_precision,
                },
                "data": {
                    "train_dir": data_cfg.get("train_dir"),
                    "valid_dir": data_cfg.get("valid_dir") or None,
                    "file_pattern": data_cfg.get("file_pattern", "*.jsonl.gz"),
                    "streaming": bool(data_cfg.get("streaming", True)),
                    "train_files": len(train_files),
                    "valid_files": len(valid_files),
                    "train_samples": train_samples if train_samples is not None else "streaming_unknown",
                    "valid_samples": valid_samples
                    if valid_samples is not None
                    else ("streaming_unknown" if valid_dataset is not None else 0),
                    "train_lines_per_file_head": train_line_counts[:5],
                    "valid_lines_per_file_head": valid_line_counts[:5],
                    "num_workers": data_cfg.get("num_workers", 8),
                    "prefetch_factor": data_cfg.get("prefetch_factor", 2),
                },
                "model": {
                    "type": model.__class__.__name__,
                    "vocab_size": model_cfg.get("vocab_size"),
                    "tokenizer_path": model_cfg.get("tokenizer_path"),
                    "mask_token_id": model_cfg.get("mask_token_id"),
                    "pad_token_id": model_cfg.get("pad_token_id"),
                    "unk_token_id": model_cfg.get("unk_token_id"),
                    "random_token_min_id": model_cfg.get("random_token_min_id"),
                    "random_token_max_id": model_cfg.get("random_token_max_id"),
                    "hidden_size": model_cfg.get("hidden_size"),
                    "num_hidden_layers": model_cfg.get("num_hidden_layers"),
                    "num_attention_heads": model_cfg.get("num_attention_heads"),
                    "intermediate_size": model_cfg.get("intermediate_size"),
                    "max_position_embeddings": model_cfg.get("max_position_embeddings"),
                    "mask_probability": model_cfg.get("mask_probability"),
                    "total_parameters": format_number(total_params),
                    "trainable_parameters": format_number(trainable_params),
                },
                "training": {
                    "max_train_steps": training_cfg.get("max_train_steps", training_cfg.get("max_steps")),
                    "scheduler_num_training_steps": training_cfg.get("scheduler_num_training_steps"),
                    "learning_rate": training_cfg.get("learning_rate"),
                    "weight_decay": training_cfg.get("weight_decay"),
                    "warmup_steps": training_cfg.get("warmup_steps"),
                    "lr_scheduler_type": training_cfg.get("lr_scheduler_type"),
                    "device_micro_batch_size": device_micro_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "effective_global_batch_size": effective_global_batch_size,
                    "gradient_clipping": training_cfg.get("gradient_clipping"),
                    "output_dir": training_cfg.get("output_dir"),
                    "log_every_steps": training_cfg.get("log_every_steps"),
                    "eval_every_steps": training_cfg.get("eval_every_steps"),
                    "max_eval_batches": training_cfg.get("max_eval_batches"),
                    "save_every_steps": training_cfg.get("save_every_steps"),
                    "resume_from_checkpoint": training_cfg.get("resume_from_checkpoint"),
                },
            },
            width=120,
            sort_dicts=False,
        )
    )
    print("-" * 80)
    print("Model architecture:")
    print(model)
    print("=" * 80 + "\n")


def mask_token_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int = 0,
    random_token_max_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT MLM masking to VQ token ids."""

    labels = input_ids.clone()
    valid_positions = attention_mask.bool()
    probability_matrix = torch.full(labels.shape, mask_probability, device=input_ids.device)
    masked_indices = torch.bernoulli(probability_matrix).bool() & valid_positions

    rows_need_mask = valid_positions.any(dim=1) & ~masked_indices.any(dim=1)
    if rows_need_mask.any():
        first_valid = valid_positions.float().argmax(dim=1)
        row_indices = torch.arange(input_ids.shape[0], device=input_ids.device)[rows_need_mask]
        masked_indices[row_indices, first_valid[rows_need_mask]] = True

    labels[~masked_indices] = -100

    corrupted = input_ids.clone()
    replace_with_mask = torch.bernoulli(
        torch.full(labels.shape, 0.8, device=input_ids.device)
    ).bool() & masked_indices
    corrupted[replace_with_mask] = mask_token_id

    replace_with_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
        & masked_indices
        & ~replace_with_mask
    )
    random_token_upper_bound = min(random_token_max_id or vocab_size, vocab_size)
    random_tokens = torch.randint(
        random_token_min_id,
        random_token_upper_bound,
        labels.shape,
        dtype=torch.long,
        device=input_ids.device,
    )
    corrupted[replace_with_random] = random_tokens[replace_with_random]

    return corrupted, labels


def original_token_len_from_item(item: dict[str, Any], sample_len: int) -> int:
    """Read real token length from preprocess_full_signal_tokenize_1600.py metadata."""

    meta = item.get("meta") or {}
    value = meta.get("original_token_len")
    if value is None:
        value = item.get("original_token_len")
    if value is None:
        value = sample_len

    try:
        original_token_len = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid original_token_len={value!r} in item id={item.get('id')!r}.") from exc

    return max(0, min(original_token_len, sample_len))


def parse_lb07_lb06_token_item(
    item: dict[str, Any],
    *,
    vocab: dict[str, int] | None = None,
    bwav_vocab_lookup: dict[int, int] | None = None,
    unk_token_id: int = 0,
    vocab_size: int | None = None,
) -> Stage2Batch:
    """Parse fixed-length token text and build attention_mask from meta.original_token_len."""

    if isinstance(item.get("input_ids"), list):
        input_ids = torch.tensor([int(token_id) for token_id in item["input_ids"]], dtype=torch.long)
    else:
        if "text" not in item:
            raise KeyError(f"Missing 'text' or 'input_ids' field in item id={item.get('id')!r}.")
        input_ids = parse_bwav_token_text(
            item["text"],
            vocab=vocab,
            bwav_vocab_lookup=bwav_vocab_lookup,
            unk_token_id=unk_token_id,
        )
    if vocab_size is not None and input_ids.numel() > 0 and int(input_ids.max()) >= vocab_size:
        raise ValueError(
            f"Token id out of range in item id={item.get('id')!r}: "
            f"max={int(input_ids.max())}, vocab_size={vocab_size}."
        )

    valid_len = original_token_len_from_item(item, sample_len=int(input_ids.shape[0]))
    if valid_len <= 0:
        raise ValueError(
            f"original_token_len must be > 0 for MLM training, got {valid_len} "
            f"in item id={item.get('id')!r}."
        )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
    attention_mask[:valid_len] = 1
    return Stage2Batch(input_ids=input_ids, attention_mask=attention_mask)


class LB07LB06TokenJsonlIterableDataset(IterableDataset):
    """Streaming JSONL/JSONL.GZ dataset using meta.original_token_len as attention mask."""

    def __init__(
        self,
        data_dir: str,
        pattern: str = "*.jsonl.gz",
        tokenizer_path: str | None = None,
        unk_token_id: int = 0,
        vocab_size: int | None = None,
        shuffle_files: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_dir = data_dir
        self.pattern = pattern
        self.tokenizer_path = tokenizer_path
        self.unk_token_id = unk_token_id
        self.vocab_size = vocab_size
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.vocab = load_tokenizer_vocab(tokenizer_path)
        self.bwav_vocab_lookup = build_bwav_vocab_lookup(self.vocab)
        self.files = sorted(Path(data_dir).glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} under {data_dir!r}.")

    @staticmethod
    def _open_text(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("r", encoding="utf-8")

    def _iter_file(self, path: Path):
        with self._open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                try:
                    yield parse_lb07_lb06_token_item(
                        item,
                        vocab=self.vocab,
                        bwav_vocab_lookup=self.bwav_vocab_lookup,
                        unk_token_id=self.unk_token_id,
                        vocab_size=self.vocab_size,
                    )
                except Exception as exc:
                    raise type(exc)(f"{path} line {line_number}: {exc}") from exc

    def __iter__(self):
        files = list(self.files)
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(files)

        rank_files = [
            path
            for file_index, path in enumerate(files)
            if file_index % max(1, world_size) == rank
        ]
        for file_index, path in enumerate(rank_files):
            if file_index % max(1, num_workers) != worker_id:
                continue
            yield from self._iter_file(path)


class LB07LB06TokenJsonlDataset(Dataset):
    """Cached JSONL/JSONL.GZ dataset using meta.original_token_len as attention mask."""

    def __init__(
        self,
        data_dir: str,
        pattern: str = "*.jsonl.gz",
        max_cache_files: int = 2,
        tokenizer_path: str | None = None,
        unk_token_id: int = 0,
        vocab_size: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.pattern = pattern
        self.max_cache_files = max_cache_files
        self.tokenizer_path = tokenizer_path
        self.unk_token_id = unk_token_id
        self.vocab_size = vocab_size
        self.vocab = load_tokenizer_vocab(tokenizer_path)
        self.bwav_vocab_lookup = build_bwav_vocab_lookup(self.vocab)
        self.files = sorted(Path(data_dir).glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r} under {data_dir!r}.")

        self.file_line_counts: list[int] = []
        self.offsets = [0]
        for path in self.files:
            line_count = self._count_lines(path)
            self.file_line_counts.append(line_count)
            self.offsets.append(self.offsets[-1] + line_count)

        self._cache: OrderedDict[Path, list[Stage2Batch]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    @staticmethod
    def _open_text(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return path.open("r", encoding="utf-8")

    def _count_lines(self, path: Path) -> int:
        with self._open_text(path) as handle:
            return sum(1 for line in handle if line.strip())

    def _load_file(self, path: Path) -> list[Stage2Batch]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        samples: list[Stage2Batch] = []
        with self._open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                try:
                    samples.append(
                        parse_lb07_lb06_token_item(
                            item,
                            vocab=self.vocab,
                            bwav_vocab_lookup=self.bwav_vocab_lookup,
                            unk_token_id=self.unk_token_id,
                            vocab_size=self.vocab_size,
                        )
                    )
                except Exception as exc:
                    raise type(exc)(f"{path} line {line_number}: {exc}") from exc

        if len(self._cache) >= self.max_cache_files:
            self._cache.popitem(last=False)
        self._cache[path] = samples
        return samples

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        for file_id in range(len(self.offsets) - 1):
            if self.offsets[file_id] <= index < self.offsets[file_id + 1]:
                return file_id, index - self.offsets[file_id]

        raise IndexError(f"Index {index} was not found in any jsonl file.")

    def __getitem__(self, index: int) -> Stage2Batch:
        file_id, local_index = self._locate(index)
        samples = self._load_file(self.files[file_id])
        return samples[local_index]


class LB07LB06Collator:
    """Collate fixed-length token samples while preserving per-sample attention masks."""

    def __init__(self, pad_token_id: int = 0, max_length: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, samples: list[Stage2Batch]) -> Stage2Batch:
        input_samples = [sample.input_ids for sample in samples]
        mask_samples = [sample.attention_mask for sample in samples]

        if self.max_length is not None:
            input_samples = [sample[: self.max_length] for sample in input_samples]
            mask_samples = [sample[: self.max_length] for sample in mask_samples]

        max_len = max(sample.shape[0] for sample in input_samples)
        input_ids = torch.full(
            (len(samples), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
        for i, (input_sample, mask_sample) in enumerate(zip(input_samples, mask_samples)):
            if input_sample.ndim != 1:
                raise ValueError(f"Token samples must have shape [seq_len]. Got {tuple(input_sample.shape)}.")
            if mask_sample.shape != input_sample.shape:
                raise ValueError(
                    f"attention_mask shape {tuple(mask_sample.shape)} does not match "
                    f"input_ids shape {tuple(input_sample.shape)}."
                )
            length = input_sample.shape[0]
            input_ids[i, :length] = input_sample.long()
            attention_mask[i, :length] = mask_sample.long()
        return Stage2Batch(input_ids=input_ids, attention_mask=attention_mask)


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    path_key = f"{split}_dir"
    streaming = bool(data_cfg.get("streaming", split == "train"))
    dataset_kwargs = {
        "data_dir": data_cfg[path_key],
        "pattern": str(data_cfg.get(f"{split}_pattern", data_cfg.get("file_pattern", "*.jsonl.gz"))),
        "tokenizer_path": model_cfg.get("tokenizer_path"),
        "unk_token_id": int(model_cfg.get("unk_token_id", 0)),
        "vocab_size": int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
    }
    if streaming:
        dataset = LB07LB06TokenJsonlIterableDataset(
            **dataset_kwargs,
            shuffle_files=bool(data_cfg.get(f"{split}_shuffle_files", split == "train")),
            seed=int(config.get("reproducibility", {}).get("seed", config.get("seed", 42))),
        )
    else:
        dataset = LB07LB06TokenJsonlDataset(
            **dataset_kwargs,
            max_cache_files=int(data_cfg.get("max_cache_files", 2)),
        )
    collator = LB07LB06Collator(
        pad_token_id=int(config.get("model", {}).get("pad_token_id", 0)),
        max_length=int(config.get("model", {}).get("max_position_embeddings", 4096)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("device_micro_batch_size", 8)),
        shuffle=(split == "train" and not streaming),
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
        drop_last=(split == "train"),
    )


def resolve_step_training_steps(training_cfg: dict[str, Any]) -> tuple[int, int]:
    """Resolve the explicit step-driven stop point and scheduler horizon."""

    max_train_steps = training_cfg.get("max_train_steps", training_cfg.get("max_steps"))
    if max_train_steps is None:
        raise ValueError("stage2_bert_encoder_train_v2.py requires training.max_train_steps.")
    max_train_steps = int(max_train_steps)
    if max_train_steps <= 0:
        raise ValueError("training.max_train_steps must be > 0.")

    scheduler_num_training_steps = training_cfg.get("scheduler_num_training_steps")
    if scheduler_num_training_steps is None:
        total_training_steps = max_train_steps
    else:
        total_training_steps = int(scheduler_num_training_steps)
        if total_training_steps <= 0:
            raise ValueError("training.scheduler_num_training_steps must be > 0 when provided.")

    return max_train_steps, total_training_steps


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_training_steps: int,
) -> LambdaLR:
    """Build a torch-only LR scheduler compatible with the old config names."""

    scheduler_name = scheduler_type.lower()
    warmup_steps = max(0, int(warmup_steps))
    total_training_steps = max(1, int(total_training_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if scheduler_name == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_name in {"constant", "constant_with_warmup"}:
            return 1.0
        raise ValueError(
            f"Unsupported lr_scheduler_type={scheduler_type!r}. "
            "Supported values: cosine, linear, constant, constant_with_warmup."
        )

    return LambdaLR(optimizer, lr_lambda)


def get_wandb_run_id(accelerator: Accelerator) -> str | None:
    """Return the active wandb run id when wandb tracking is enabled."""

    try:
        tracker = accelerator.get_tracker("wandb", unwrap=True)
    except Exception:
        return None
    return getattr(tracker, "id", None)


def load_trainer_state(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Load optimizer/scheduler training state if present."""

    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return state if "optimizer_state_dict" in state else None

    state_path = ckpt_path / "trainer_state.pt"
    if not state_path.exists():
        return None
    return torch.load(state_path, map_location="cpu", weights_only=False)


def load_model_from_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    checkpoint_dir: str | os.PathLike[str],
) -> None:
    """Load MaskedSignalLM weights from a v1 directory or v7-style .pt checkpoint."""

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    unwrapped = accelerator.unwrap_model(model)
    if ckpt_dir.is_file():
        checkpoint = torch.load(ckpt_dir, map_location="cpu", weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"Could not find model_state_dict in checkpoint file: {ckpt_dir}")
        missing, unexpected = unwrapped.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        weight_path = ckpt_dir / "model_state.pt"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Could not find model_state.pt in checkpoint directory: {ckpt_dir}. "
                "HF BertForMaskedLM checkpoints are not compatible with MaskedSignalLM."
            )
        missing, unexpected = unwrapped.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=False),
            strict=False,
        )

    if accelerator.is_main_process:
        print(
            f"Loaded model weights from {ckpt_dir} "
            f"(missing={list(missing)}, unexpected={list(unexpected)})",
            flush=True,
        )


def build_wandb_init_kwargs(
    config: dict[str, Any],
    resume_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build wandb init kwargs, including run resume settings when configured."""

    wandb_cfg = config.get("wandb", {})
    wandb_kwargs: dict[str, Any] = {"name": wandb_cfg.get("name")}
    run_id = wandb_cfg.get("id") or wandb_cfg.get("run_id")
    if run_id is None and resume_state is not None:
        run_id = resume_state.get("wandb_run_id")

    if run_id:
        wandb_kwargs["id"] = str(run_id)
        wandb_kwargs["resume"] = str(wandb_cfg.get("resume", "allow"))
    elif wandb_cfg.get("resume"):
        wandb_kwargs["resume"] = str(wandb_cfg["resume"])

    return {"wandb": wandb_kwargs}


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: str,
    step: int | str,
    global_step: int,
    epoch: int,
    best_eval_loss: float,
    config: dict[str, Any],
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    save_dir = Path(output_dir) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), save_dir / "model_state.pt")
    torch.save(
        {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "best_eval_loss": float(best_eval_loss),
            "model_class": "MaskedSignalLM",
            "model_state_path": "model_state.pt",
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "wandb_run_id": get_wandb_run_id(accelerator),
            "config": config,
        },
        save_dir / "trainer_state.pt",
    )


def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    valid_loader: DataLoader,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int = 0,
    random_token_max_id: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """Run MLM evaluation on the validation split."""

    model.eval()
    losses = []
    top1_correct = []
    top5_correct = []
    top10_correct = []
    top50_correct = []
    masked_counts = []

    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break

            input_ids = batch.input_ids.to(accelerator.device)
            attention_mask = batch.attention_mask.to(accelerator.device)
            corrupted, labels = mask_token_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                mask_probability=mask_probability,
                random_token_min_id=random_token_min_id,
                random_token_max_id=random_token_max_id,
            )
            logits = model(
                input_ids=corrupted,
                attention_mask=attention_mask,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            losses.append(accelerator.gather_for_metrics(loss.detach()))

            masked_positions = labels != -100
            masked_count = masked_positions.sum()
            masked_counts.append(accelerator.gather_for_metrics(masked_count.detach().reshape(1)))

            if masked_count.item() > 0:
                masked_logits = logits[masked_positions]
                masked_labels = labels[masked_positions]
                topk = torch.topk(masked_logits, k=min(50, masked_logits.shape[-1]), dim=-1).indices
                top1 = (topk[:, :1] == masked_labels[:, None]).any(dim=-1).sum()
                top5 = (topk[:, : min(5, topk.shape[-1])] == masked_labels[:, None]).any(dim=-1).sum()
                top10 = (topk[:, : min(10, topk.shape[-1])] == masked_labels[:, None]).any(dim=-1).sum()
                top50 = (topk == masked_labels[:, None]).any(dim=-1).sum()
            else:
                top1 = torch.zeros((), dtype=torch.long, device=labels.device)
                top5 = torch.zeros((), dtype=torch.long, device=labels.device)
                top10 = torch.zeros((), dtype=torch.long, device=labels.device)
                top50 = torch.zeros((), dtype=torch.long, device=labels.device)

            top1_correct.append(accelerator.gather_for_metrics(top1.detach().reshape(1)))
            top5_correct.append(accelerator.gather_for_metrics(top5.detach().reshape(1)))
            top10_correct.append(accelerator.gather_for_metrics(top10.detach().reshape(1)))
            top50_correct.append(accelerator.gather_for_metrics(top50.detach().reshape(1)))

    model.train()

    if not losses:
        return {
            "eval/loss": float("nan"),
            "eval/perplexity": float("nan"),
            "eval/top1_accuracy": float("nan"),
            "eval/top5_accuracy": float("nan"),
            "eval/top10_accuracy": float("nan"),
            "eval/top50_accuracy": float("nan"),
        }

    loss = torch.cat([loss.reshape(-1) for loss in losses]).mean().item()
    perplexity = math.exp(loss) if loss < 50 else float("inf")
    total_masked = torch.cat(masked_counts).sum().item()
    top1 = torch.cat(top1_correct).sum().item()
    top5 = torch.cat(top5_correct).sum().item()
    top10 = torch.cat(top10_correct).sum().item()
    top50 = torch.cat(top50_correct).sum().item()

    return {
        "eval/loss": loss,
        "eval/perplexity": perplexity,
        "eval/top1_accuracy": top1 / total_masked if total_masked else float("nan"),
        "eval/top5_accuracy": top5 / total_masked if total_masked else float("nan"),
        "eval/top10_accuracy": top10 / total_masked if total_masked else float("nan"),
        "eval/top50_accuracy": top50 / total_masked if total_masked else float("nan"),
        "eval/masked_tokens": total_masked,
    }


def train(config: dict[str, Any]) -> None:
    seed = int(config.get("reproducibility", {}).get("seed", config.get("seed", 42)))
    seed_everything(seed)

    training_cfg = config["training"]
    model_cfg = config["model"]
    mask_probability = float(model_cfg.get("mask_probability", 0.15))
    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint")
    resume_state = load_trainer_state(resume_from_checkpoint) if resume_from_checkpoint else None

    accelerator = Accelerator(
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(training_cfg.get("mixed_precision", "no")),
        log_with="wandb" if config.get("wandb", {}).get("use_wandb", False) else None,
        project_dir=str(training_cfg.get("log_dir", "log")),
    )

    train_loader = build_dataloader(config, "train")
    valid_loader = build_dataloader(config, "valid") if config["data"].get("valid_dir") else None

    model = build_masked_signal_lm(config)
    print_startup_summary(
        accelerator=accelerator,
        config=config,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seed=seed,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    max_train_steps, total_training_steps = resolve_step_training_steps(training_cfg)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=str(training_cfg.get("lr_scheduler_type", "cosine")),
        warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        total_training_steps=total_training_steps,
    )

    if config.get("wandb", {}).get("use_wandb", False):
        accelerator.init_trackers(
            project_name=str(config["wandb"].get("project", "poredlm-stage2-bert")),
            config=config,
            init_kwargs=build_wandb_init_kwargs(config, resume_state),
        )

    if valid_loader is not None:
        model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            valid_loader,
            scheduler,
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            scheduler,
        )

    resumed_step = resume_state.get("global_step", resume_state.get("step", 0)) if resume_state is not None else 0
    global_step = int(resumed_step) if isinstance(resumed_step, int) or str(resumed_step).isdigit() else 0
    start_epoch = int(resume_state.get("epoch", 0)) if resume_state is not None else 0
    if start_epoch < 0:
        raise ValueError(f"Invalid resumed completed pass count: {start_epoch}")
    if global_step > max_train_steps:
        raise ValueError(
            f"Checkpoint global_step={global_step} is beyond configured max_train_steps={max_train_steps}."
        )

    output_dir = str(training_cfg.get("output_dir", "outputs/stage2_BERT_Encoder"))
    save_every = int(training_cfg.get("save_every_steps", 1000))
    log_every = int(training_cfg.get("log_every_steps", 10))
    eval_every = int(training_cfg.get("eval_every_steps", 0))
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    vocab_size = int(model_cfg.get("vocab_size", 65537))
    mask_token_id = int(model_cfg.get("mask_token_id", vocab_size - 1))
    random_token_min_id = int(model_cfg.get("random_token_min_id", 0))
    random_token_max_id = int(model_cfg.get("random_token_max_id", vocab_size))
    best_eval_loss = (
        float(resume_state.get("best_eval_loss", resume_state.get("val_loss", float("inf"))))
        if resume_state is not None
        else float("inf")
    )

    if resume_from_checkpoint:
        load_model_from_checkpoint(accelerator, model, resume_from_checkpoint)
        if resume_state is not None:
            if resume_state.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if resume_state.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if resume_state.get("rng_state") is not None:
                torch.set_rng_state(resume_state["rng_state"])
            if torch.cuda.is_available() and resume_state.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])
            if resume_state.get("numpy_rng_state") is not None:
                np.random.set_state(resume_state["numpy_rng_state"])
            if resume_state.get("python_rng_state") is not None:
                random.setstate(resume_state["python_rng_state"])
            if accelerator.is_main_process:
                print(
                    "Resumed full training state from "
                    f"{resume_from_checkpoint}: global_step={global_step}, "
                    f"completed_passes={start_epoch}, best_eval_loss={best_eval_loss}",
                    flush=True,
                )
        elif accelerator.is_main_process:
            print(
                f"Loaded model-only checkpoint from {resume_from_checkpoint}; "
                "optimizer/scheduler/global_step start from scratch because trainer_state.pt was not found.",
                flush=True,
            )

    progress = tqdm(
        total=total_training_steps,
        initial=min(global_step, total_training_steps),
        disable=not accelerator.is_local_main_process,
    )
    model.train()

    completed_passes = start_epoch
    while global_step < max_train_steps:
        pass_start_step = global_step
        if accelerator.is_main_process:
            print(
                f"Starting streaming pass {completed_passes + 1} "
                f"(global_step={global_step}/{max_train_steps})",
                flush=True,
            )

        for batch in train_loader:
            if global_step >= max_train_steps:
                break

            with accelerator.accumulate(model):
                input_ids = batch.input_ids.to(accelerator.device)
                attention_mask = batch.attention_mask.to(accelerator.device)
                corrupted, labels = mask_token_ids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    mask_probability=mask_probability,
                    random_token_min_id=random_token_min_id,
                    random_token_max_id=random_token_max_id,
                )
                logits = model(
                    input_ids=corrupted,
                    attention_mask=attention_mask,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(training_cfg.get("gradient_clipping", 1.0)),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % log_every == 0:
                    loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
                    logs = {
                        "train/loss": loss_value,
                        "train/loss_log10": math.log10(loss_value + 1e-12),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/pass": completed_passes + 1,
                        "step": global_step,
                    }
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        print(logs)

                if save_every > 0 and global_step % save_every == 0:
                    save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        step=global_step,
                        global_step=global_step,
                        epoch=completed_passes,
                        best_eval_loss=best_eval_loss,
                        config=config,
                    )

                if valid_loader is not None and eval_every > 0 and global_step % eval_every == 0:
                    eval_logs = evaluate(
                        accelerator=accelerator,
                        model=model,
                        valid_loader=valid_loader,
                        vocab_size=vocab_size,
                        mask_token_id=mask_token_id,
                        mask_probability=mask_probability,
                        random_token_min_id=random_token_min_id,
                        random_token_max_id=random_token_max_id,
                        max_eval_batches=max_eval_batches,
                    )
                    eval_logs["eval/pass"] = completed_passes + 1
                    eval_logs["step"] = global_step
                    accelerator.log(eval_logs, step=global_step)
                    if accelerator.is_main_process:
                        print(eval_logs)

                    eval_loss = eval_logs["eval/loss"]
                    if eval_loss == eval_loss and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            output_dir=output_dir,
                            step="best",
                            global_step=global_step,
                            epoch=completed_passes,
                            best_eval_loss=best_eval_loss,
                            config=config,
                        )

        completed_passes += 1
        if global_step == pass_start_step:
            raise RuntimeError(
                "The training dataloader produced no optimizer steps in this streaming pass. "
                "For DDP streaming, make sure every rank/worker gets at least one file or "
                "try fewer processes/workers."
            )

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        step=global_step,
        global_step=global_step,
        epoch=completed_passes,
        best_eval_loss=best_eval_loss,
        config=config,
    )
    progress.close()
    accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 BERT Encoder")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(config)


if __name__ == "__main__":
    main()
