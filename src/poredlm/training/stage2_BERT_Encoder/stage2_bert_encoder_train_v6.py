"""Train a BERT encoder with a Stage1-initialized waveform decoder.

V6 keeps the V5 masking/loss behavior but drives training by epochs instead of
an outer max-step loop. It still supports optional step-based logging,
evaluation, checkpointing, and scheduler estimation.

V5 kept the V4 masking curriculum but replaced the codebook vector regression
head with a CNN waveform decoder initialized from the Stage1 tokenizer decoder.
The auxiliary target waveform is produced by a frozen copy of the original
Stage1 codebook + decoder using BERT-predicted codebook ids.

    Loss = CE_token + waveform_mse_weight * MSE(pred_waveform, target_waveform)

The replacement policy is still the standard BERT 80/10/10 rule.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from bert_encoder_model import build_bert_mlm
from dataset import Stage2Collator, Stage2TokenJsonlDataset, Stage2TokenJsonlIterableDataset


REPO_ROOT = Path(__file__).resolve().parents[4]
POREDLM_SRC = REPO_ROOT / "src" / "poredlm"
PROJECT_SRC = REPO_ROOT / "src"
for import_path in (POREDLM_SRC, PROJECT_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from models import SignalCNN


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


def build_accelerator(config: dict[str, Any]) -> Accelerator:
    """Build Accelerator with iterable datasets read independently on each rank."""

    training_cfg = config["training"]
    accelerator_kwargs: dict[str, Any] = {
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "mixed_precision": str(training_cfg.get("mixed_precision", "no")),
        "log_with": "wandb" if config.get("wandb", {}).get("use_wandb", False) else None,
        "project_dir": str(training_cfg.get("log_dir", "log")),
    }
    if bool(config.get("data", {}).get("streaming", False)):
        accelerator_params = inspect.signature(Accelerator).parameters
        if "dataloader_config" in accelerator_params:
            from accelerate import DataLoaderConfiguration

            accelerator_kwargs["dataloader_config"] = DataLoaderConfiguration(dispatch_batches=False)
        elif "dispatch_batches" in accelerator_params:
            accelerator_kwargs["dispatch_batches"] = False
    return Accelerator(**accelerator_kwargs)


def load_checkpoint_state_dict(checkpoint_dir: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a Stage 1 Accelerate checkpoint state dict and metadata."""

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Stage1 checkpoint directory not found: {ckpt_dir}")

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

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return state_dict, metadata


def load_stage1_codebook_vectors(
    checkpoint_dir: str,
    *,
    codebook_key: str | None = None,
) -> torch.Tensor:
    """Load Stage 1 VQ codebook vectors as [codebook_size, codebook_dim]."""

    state_dict, metadata = load_checkpoint_state_dict(checkpoint_dir)
    candidate_keys: list[str] = []
    if codebook_key:
        candidate_keys.append(codebook_key)
    candidate_keys.extend(
        key
        for key in state_dict
        if key.endswith("vq._codebook.embed") or key.endswith("_codebook.embed")
    )
    candidate_keys.extend(
        key
        for key in state_dict
        if "codebook" in key.lower() and key.endswith("embed")
    )

    seen_keys: set[str] = set()
    unique_candidate_keys = []
    for key in candidate_keys:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_candidate_keys.append(key)

    if not unique_candidate_keys:
        raise KeyError(
            "Could not find Stage1 codebook embedding in checkpoint. "
            "Set model.codebook_key in the config to the exact state_dict key."
        )

    selected_key = unique_candidate_keys[0]
    if selected_key not in state_dict:
        raise KeyError(f"Configured codebook_key={selected_key!r} was not found in the checkpoint.")

    codebook = state_dict[selected_key].detach().float()
    if codebook.ndim == 3:
        if codebook.shape[0] != 1:
            print(
                f"[WARN] Codebook tensor {selected_key!r} has shape {tuple(codebook.shape)}; using layer 0.",
                flush=True,
            )
        codebook = codebook[0]
    if codebook.ndim != 2:
        raise RuntimeError(
            f"Expected codebook tensor [K, D] or [1, K, D], got {tuple(codebook.shape)} "
            f"from key {selected_key!r}."
        )

    expected_size = metadata.get("codebook_size")
    expected_dim = metadata.get("codebook_dim")
    if expected_size is not None and int(expected_size) != int(codebook.shape[0]):
        raise RuntimeError(
            f"metadata codebook_size={expected_size} but loaded codebook has shape {tuple(codebook.shape)}."
        )
    if expected_dim is not None and int(expected_dim) != int(codebook.shape[1]):
        raise RuntimeError(
            f"metadata codebook_dim={expected_dim} but loaded codebook has shape {tuple(codebook.shape)}."
        )
    return codebook.contiguous()


def load_stage1_signal_cnn_decoder(
    checkpoint_dir: str,
    *,
    cnn_type: int | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    """Load two Stage1 CNN decoders: one trainable and one frozen target copy."""

    state_dict, metadata = load_checkpoint_state_dict(checkpoint_dir)
    resolved_cnn_type = int(metadata.get("cnn_type", 0 if cnn_type is None else cnn_type))
    if cnn_type is not None and int(cnn_type) != resolved_cnn_type:
        print(
            f"[WARN] Config cnn_type={cnn_type} differs from checkpoint metadata "
            f"cnn_type={resolved_cnn_type}; using config value.",
            flush=True,
        )
        resolved_cnn_type = int(cnn_type)

    def build_decoder() -> torch.nn.Module:
        signal_cnn = SignalCNN(cnn_type=resolved_cnn_type)
        cnn_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "cnn_model." not in key:
                continue
            cnn_key = key.split("cnn_model.", 1)[1]
            if cnn_key.startswith("decoder."):
                cnn_state[cnn_key] = value.detach()
        if not cnn_state:
            raise KeyError(
                "Could not find Stage1 decoder weights in checkpoint. "
                "Expected keys containing 'cnn_model.decoder.'."
            )
        missing, unexpected = signal_cnn.load_state_dict(cnn_state, strict=False)
        unexpected = [key for key in unexpected if key.startswith("decoder.")]
        missing_decoder = [key for key in missing if key.startswith("decoder.")]
        if missing_decoder or unexpected:
            raise RuntimeError(
                "Failed to load Stage1 decoder weights cleanly. "
                f"missing={missing_decoder}, unexpected={unexpected}"
            )
        return signal_cnn.decoder

    trainable_decoder = build_decoder()
    target_decoder = build_decoder()
    target_decoder.eval()
    for param in target_decoder.parameters():
        param.requires_grad = False

    return trainable_decoder, target_decoder, metadata


class BertMlmWithWaveformDecoder(torch.nn.Module):
    """BERT MLM plus a Stage1-initialized waveform decoder loss."""

    def __init__(
        self,
        bert_mlm: torch.nn.Module,
        *,
        codebook_vectors: torch.Tensor,
        wave_decoder: torch.nn.Module,
        target_decoder: torch.nn.Module,
        codebook_vocab_offset: int,
        waveform_mse_weight: float,
        cnn_type: int,
        cnn_stride: int,
    ) -> None:
        super().__init__()
        if codebook_vectors.ndim != 2:
            raise ValueError(f"codebook_vectors must be [K, D], got {tuple(codebook_vectors.shape)}")

        self.bert_mlm = bert_mlm
        self.wave_decoder = wave_decoder
        self.target_decoder = target_decoder
        self.codebook_vocab_offset = int(codebook_vocab_offset)
        self.waveform_mse_weight = float(waveform_mse_weight)
        self.cnn_type = int(cnn_type)
        self.cnn_stride = int(cnn_stride)
        self.codebook_size = int(codebook_vectors.shape[0])
        self.codebook_dim = int(codebook_vectors.shape[1])
        self.hidden_size = int(bert_mlm.config.hidden_size)
        if self.hidden_size != self.codebook_dim:
            raise ValueError(
                "V5 decodes BERT hidden_states directly with the Stage1 CNN decoder, "
                f"so hidden_size must equal codebook_dim. Got hidden_size={self.hidden_size}, "
                f"codebook_dim={self.codebook_dim}."
            )
        self.register_buffer("codebook_vectors", codebook_vectors.float(), persistent=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        outputs = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        pred_latents = hidden_states.permute(0, 2, 1).contiguous()
        predicted_waveform = self.wave_decoder(pred_latents)

        token_loss = outputs.loss
        waveform_mse_loss = predicted_waveform.new_zeros(())
        target_waveform = None

        if labels is not None:
            masked_positions = labels != -100
            if bool(masked_positions.any().item()):
                start = self.codebook_vocab_offset
                end = start + self.codebook_size
                codebook_logits = outputs.logits[..., start:end]
                predicted_codebook_ids = codebook_logits.detach().argmax(dim=-1)
                target_vectors = self.codebook_vectors[predicted_codebook_ids].to(
                    device=predicted_waveform.device,
                    dtype=predicted_waveform.dtype,
                )
                target_latents = target_vectors.permute(0, 2, 1).contiguous()
                with torch.no_grad():
                    target_waveform = self.target_decoder(target_latents)
                target_waveform = self._align_waveform_length(target_waveform, predicted_waveform.shape[-1])
                waveform_mse_loss = F.mse_loss(predicted_waveform, target_waveform)

        if token_loss is None:
            token_loss = predicted_waveform.new_zeros(())
        loss = token_loss + self.waveform_mse_weight * waveform_mse_loss
        return SimpleNamespace(
            loss=loss,
            logits=outputs.logits,
            token_loss=token_loss,
            waveform_mse_loss=waveform_mse_loss,
            predicted_waveform=predicted_waveform,
            target_waveform=target_waveform,
        )

    @staticmethod
    def _align_waveform_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = waveform.shape[-1]
        if current_len > target_len:
            return waveform[..., :target_len]
        if current_len < target_len:
            return F.pad(waveform, (0, target_len - current_len))
        return waveform

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save a HF-compatible BERT checkpoint plus the V5 auxiliary state."""

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.bert_mlm.save_pretrained(save_path)
        torch.save(
            {
                "wave_decoder": self.wave_decoder.state_dict(),
                "target_decoder": self.target_decoder.state_dict(),
                "codebook_vectors": self.codebook_vectors.detach().cpu(),
            },
            save_path / "poredlm_v5_auxiliary.bin",
        )
        metadata = {
            "model_type": self.__class__.__name__,
            "codebook_vocab_offset": self.codebook_vocab_offset,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "hidden_size": self.hidden_size,
            "cnn_type": self.cnn_type,
            "cnn_stride": self.cnn_stride,
            "waveform_mse_weight": self.waveform_mse_weight,
        }
        (save_path / "waveform_decoder_config.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


def build_bert_mlm_with_waveform_decoder(config: dict[str, Any]) -> BertMlmWithWaveformDecoder:
    """Build the V5 model with MLM and waveform decoder losses."""

    model_cfg = config["model"]
    stage1_ckpt = model_cfg.get("stage1_codebook_ckpt") or model_cfg.get("stage1_ckpt")
    if not stage1_ckpt:
        raise KeyError("V5 requires model.stage1_codebook_ckpt or model.stage1_ckpt in the config.")

    codebook_vectors = load_stage1_codebook_vectors(
        str(stage1_ckpt),
        codebook_key=model_cfg.get("codebook_key"),
    )
    wave_decoder, target_decoder, metadata = load_stage1_signal_cnn_decoder(
        str(stage1_ckpt),
        cnn_type=model_cfg.get("cnn_type"),
    )
    return BertMlmWithWaveformDecoder(
        build_bert_mlm(config),
        codebook_vectors=codebook_vectors,
        wave_decoder=wave_decoder,
        target_decoder=target_decoder,
        codebook_vocab_offset=int(model_cfg.get("codebook_vocab_offset", 129)),
        waveform_mse_weight=float(model_cfg.get("waveform_mse_weight", 1.0)),
        cnn_type=int(metadata.get("cnn_type", model_cfg.get("cnn_type", 0))),
        cnn_stride=int(metadata.get("cnn_stride", 5)),
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
    short_span_start_step = int(model_cfg.get("short_span_start_step", 20000))
    long_span_start_step = int(model_cfg.get("long_span_start_step", 60000))
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
    try:
        train_samples: int | str = len(train_dataset)
    except TypeError:
        train_samples = "streaming"
    try:
        valid_samples: int | str = len(valid_dataset) if valid_dataset is not None else 0
    except TypeError:
        valid_samples = "streaming"

    print("\n" + "=" * 80)
    print("Starting Stage 2 BERT Encoder Training")
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
                    "streaming": data_cfg.get("streaming", False),
                    "train_files": len(train_files),
                    "valid_files": len(valid_files),
                    "train_samples": train_samples,
                    "valid_samples": valid_samples,
                    "train_lines_per_file_head": train_line_counts[:5],
                    "valid_lines_per_file_head": valid_line_counts[:5],
                    "num_workers": data_cfg.get("num_workers", 8),
                    "prefetch_factor": data_cfg.get("prefetch_factor", 2),
                },
                "model": {
                    "type": model.__class__.__name__,
                    "vocab_size": model_cfg.get("vocab_size"),
                    "tokenizer_path": model_cfg.get("tokenizer_path"),
                    "stage1_codebook_ckpt": model_cfg.get("stage1_codebook_ckpt")
                    or model_cfg.get("stage1_ckpt"),
                    "codebook_key": model_cfg.get("codebook_key"),
                    "codebook_vocab_offset": model_cfg.get("codebook_vocab_offset", 129),
                    "waveform_mse_weight": model_cfg.get("waveform_mse_weight", 1.0),
                    "cnn_type": getattr(model, "cnn_type", model_cfg.get("cnn_type")),
                    "cnn_stride": getattr(model, "cnn_stride", None),
                    "loaded_codebook_shape": (
                        getattr(model, "codebook_size", None),
                        getattr(model, "codebook_dim", None),
                    ),
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
                    "mask_curriculum": {
                        f"0..{short_span_start_step - 1}": {
                            "single_token_sample_probability": 1.00,
                            "short_span_sample_probability": 0.00,
                            "long_span_sample_probability": 0.00,
                        },
                        f"{short_span_start_step}..{long_span_start_step - 1}": {
                            "single_token_sample_probability": 0.80,
                            "short_span_sample_probability": 0.20,
                            "long_span_sample_probability": 0.00,
                        },
                        f"{long_span_start_step}+": {
                            "single_token_sample_probability": 0.70,
                            "short_span_sample_probability": 0.20,
                            "long_span_sample_probability": 0.10,
                        },
                    },
                    "short_span_length": (
                        model_cfg.get("short_span_min_length", 2),
                        model_cfg.get("short_span_max_length", 5),
                    ),
                    "long_span_length": (
                        model_cfg.get("long_span_min_length", 6),
                        model_cfg.get("long_span_max_length", 20),
                    ),
                    "total_parameters": format_number(total_params),
                    "trainable_parameters": format_number(trainable_params),
                },
                "training": {
                    "num_train_epochs": training_cfg.get(
                        "num_train_epochs",
                        training_cfg.get("epochs", 1),
                    ),
                    "steps_per_epoch": training_cfg.get("steps_per_epoch"),
                    "scheduler_num_training_steps": training_cfg.get("scheduler_num_training_steps"),
                    "max_steps": training_cfg.get("max_steps"),
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


def _mask_random_single_positions(
    masked_indices: torch.Tensor,
    batch_index: int,
    valid_token_indices: torch.Tensor,
    target_mask_count: int,
) -> None:
    """Mask target_count independent token positions within one sample."""

    valid_count = int(valid_token_indices.numel())
    if target_mask_count >= valid_count:
        selected = valid_token_indices
    else:
        permutation = torch.randperm(valid_count, device=masked_indices.device)
        selected = valid_token_indices[permutation[:target_mask_count]]
    masked_indices[batch_index, selected] = True


def _mask_contiguous_spans(
    masked_indices: torch.Tensor,
    batch_index: int,
    valid_token_indices: torch.Tensor,
    target_mask_count: int,
    *,
    min_span_length: int,
    max_span_length: int,
) -> None:
    """Mask random contiguous spans until the sample reaches target_count tokens."""

    valid_count = int(valid_token_indices.numel())
    if target_mask_count >= valid_count:
        masked_indices[batch_index, valid_token_indices] = True
        return

    min_span_length = max(1, int(min_span_length))
    max_span_length = max(min_span_length, int(max_span_length))
    max_attempts = max(20, target_mask_count * 20)
    attempts = 0

    while attempts < max_attempts:
        current_count = int(masked_indices[batch_index, valid_token_indices].sum().item())
        remaining = target_mask_count - current_count
        if remaining <= 0:
            break
        if remaining < min_span_length:
            break

        span_upper = min(max_span_length, valid_count, remaining)
        if span_upper < min_span_length:
            break
        span_len = int(
            torch.randint(
                min_span_length,
                span_upper + 1,
                (1,),
                device=masked_indices.device,
            ).item()
        )
        start = int(torch.randint(0, valid_count - span_len + 1, (1,), device=masked_indices.device).item())
        selected = valid_token_indices[start : start + span_len]
        masked_indices[batch_index, selected] = True
        attempts += 1

    current_count = int(masked_indices[batch_index, valid_token_indices].sum().item())
    remaining = target_mask_count - current_count
    if remaining > 0:
        unmasked_valid = valid_token_indices[~masked_indices[batch_index, valid_token_indices]]
        if int(unmasked_valid.numel()) <= remaining:
            selected = unmasked_valid
        else:
            permutation = torch.randperm(int(unmasked_valid.numel()), device=masked_indices.device)
        selected = unmasked_valid[permutation[:remaining]]
    masked_indices[batch_index, selected] = True


def resolve_curriculum_mask_probabilities(
    global_step: int,
    *,
    short_span_start_step: int = 20000,
    long_span_start_step: int = 60000,
) -> tuple[int, float, float, float]:
    """Return the v3 mask curriculum phase and sample-level strategy probabilities."""

    short_span_start_step = int(short_span_start_step)
    long_span_start_step = int(long_span_start_step)
    if short_span_start_step < 0:
        raise ValueError("short_span_start_step must be >= 0.")
    if long_span_start_step < short_span_start_step:
        raise ValueError("long_span_start_step must be >= short_span_start_step.")

    step = int(global_step)
    if step < short_span_start_step:
        return 0, 1.00, 0.00, 0.00
    if step < long_span_start_step:
        return 1, 0.80, 0.20, 0.00
    return 2, 0.70, 0.20, 0.10


def build_mixed_mask_indices(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mask_probability: float,
    single_token_sample_probability: float,
    short_span_sample_probability: float,
    long_span_sample_probability: float,
    short_span_min_length: int,
    short_span_max_length: int,
    long_span_min_length: int,
    long_span_max_length: int,
) -> torch.Tensor:
    """Build sample-level 70/20/10 masks with a fixed per-sample mask budget."""

    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    valid_positions = attention_mask.bool()
    probability_total = (
        float(single_token_sample_probability)
        + float(short_span_sample_probability)
        + float(long_span_sample_probability)
    )
    if probability_total <= 0.0:
        raise ValueError("At least one sample mask strategy probability must be > 0.")

    single_threshold = float(single_token_sample_probability) / probability_total
    short_threshold = (
        float(single_token_sample_probability) + float(short_span_sample_probability)
    ) / probability_total

    for batch_index in range(input_ids.shape[0]):
        valid_token_indices = torch.nonzero(valid_positions[batch_index], as_tuple=False).flatten()
        valid_count = int(valid_token_indices.numel())
        if valid_count == 0:
            continue

        target_mask_count = int(round(valid_count * float(mask_probability)))
        if mask_probability > 0.0:
            target_mask_count = max(1, target_mask_count)
        target_mask_count = min(valid_count, target_mask_count)
        if target_mask_count <= 0:
            continue

        mode_sample = float(torch.rand((), device=input_ids.device).item())
        if mode_sample < single_threshold:
            _mask_random_single_positions(
                masked_indices,
                batch_index,
                valid_token_indices,
                target_mask_count,
            )
        elif mode_sample < short_threshold:
            _mask_contiguous_spans(
                masked_indices,
                batch_index,
                valid_token_indices,
                target_mask_count,
                min_span_length=short_span_min_length,
                max_span_length=short_span_max_length,
            )
        else:
            _mask_contiguous_spans(
                masked_indices,
                batch_index,
                valid_token_indices,
                target_mask_count,
                min_span_length=long_span_min_length,
                max_span_length=long_span_max_length,
            )

    if not masked_indices.any():
        valid_positions_flat = torch.nonzero(valid_positions, as_tuple=False)
        if int(valid_positions_flat.numel()) > 0:
            first_batch = int(valid_positions_flat[0, 0].item())
            first_position = int(valid_positions_flat[0, 1].item())
            masked_indices[first_batch, first_position] = True

    return masked_indices


def mask_token_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int = 0,
    random_token_max_id: int | None = None,
    single_token_sample_probability: float = 0.70,
    short_span_sample_probability: float = 0.20,
    long_span_sample_probability: float = 0.10,
    short_span_min_length: int = 2,
    short_span_max_length: int = 5,
    long_span_min_length: int = 6,
    long_span_max_length: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixed 70/20/10 sample-level MLM masking."""

    labels = input_ids.clone()
    masked_indices = build_mixed_mask_indices(
        input_ids,
        attention_mask,
        mask_probability=mask_probability,
        single_token_sample_probability=single_token_sample_probability,
        short_span_sample_probability=short_span_sample_probability,
        long_span_sample_probability=long_span_sample_probability,
        short_span_min_length=short_span_min_length,
        short_span_max_length=short_span_max_length,
        long_span_min_length=long_span_min_length,
        long_span_max_length=long_span_max_length,
    )

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


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    path_key = f"{split}_dir"
    pattern = str(data_cfg.get(f"{split}_pattern", data_cfg.get("file_pattern", "*.jsonl.gz")))
    use_streaming = bool(data_cfg.get("streaming", False))
    if use_streaming:
        dataset = Stage2TokenJsonlIterableDataset(
            data_dir=data_cfg[path_key],
            pattern=pattern,
            tokenizer_path=model_cfg.get("tokenizer_path"),
            unk_token_id=int(model_cfg.get("unk_token_id", 0)),
            vocab_size=int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
            shuffle_files=(split == "train"),
            seed=int(config.get("reproducibility", {}).get("seed", config.get("seed", 42))),
        )
    else:
        dataset = Stage2TokenJsonlDataset(
            data_dir=data_cfg[path_key],
            pattern=pattern,
            max_cache_files=int(data_cfg.get("max_cache_files", data_cfg.get("max_cache_size", 2))),
            tokenizer_path=model_cfg.get("tokenizer_path"),
            unk_token_id=int(model_cfg.get("unk_token_id", 0)),
            vocab_size=int(model_cfg.get("vocab_size")) if model_cfg.get("vocab_size") else None,
        )
    collator = Stage2Collator(
        pad_token_id=int(config.get("model", {}).get("pad_token_id", 0)),
        max_length=int(config.get("model", {}).get("max_position_embeddings", 4096)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("device_micro_batch_size", 8)),
        shuffle=(split == "train" and not use_streaming),
        num_workers=int(data_cfg.get("num_workers", 8)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        collate_fn=collator,
        drop_last=(split == "train"),
    )


def get_batch_tensor(batch: Any, name: str) -> torch.Tensor:
    """Read tensors from either the local Stage2Batch object or a plain mapping."""

    if isinstance(batch, dict):
        return batch[name]
    return getattr(batch, name)


def infer_update_steps_per_epoch(
    train_loader: DataLoader,
    gradient_accumulation_steps: int,
) -> int | None:
    """Infer optimizer update steps per epoch when the loader exposes length."""

    try:
        batches_per_epoch = len(train_loader)
    except TypeError:
        return None

    if batches_per_epoch <= 0:
        return None
    return math.ceil(batches_per_epoch / max(1, int(gradient_accumulation_steps)))


def resolve_epoch_training_steps(
    training_cfg: dict[str, Any],
    train_loader: DataLoader,
) -> tuple[int, int | None, int]:
    """Resolve epoch count, optional progress steps, and scheduler total steps."""

    num_train_epochs = int(training_cfg.get("num_train_epochs", training_cfg.get("epochs", 1)))
    if num_train_epochs <= 0:
        raise ValueError("training.num_train_epochs/training.epochs must be > 0.")

    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    inferred_steps_per_epoch = infer_update_steps_per_epoch(
        train_loader,
        gradient_accumulation_steps,
    )
    configured_steps_per_epoch = training_cfg.get("steps_per_epoch")
    if configured_steps_per_epoch is not None:
        steps_per_epoch = int(configured_steps_per_epoch)
        if steps_per_epoch <= 0:
            raise ValueError("training.steps_per_epoch must be > 0 when provided.")
    else:
        steps_per_epoch = inferred_steps_per_epoch

    if steps_per_epoch is not None:
        total_training_steps = steps_per_epoch * num_train_epochs
    else:
        total_training_steps = int(
            training_cfg.get(
                "scheduler_num_training_steps",
                training_cfg.get("max_steps", 100000),
            )
        )
        if total_training_steps <= 0:
            raise ValueError(
                "Could not infer scheduler training steps for an iterable dataloader. "
                "Set training.steps_per_epoch or training.scheduler_num_training_steps."
            )

    return num_train_epochs, steps_per_epoch, total_training_steps


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: str,
    step: int | str,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    save_dir = Path(output_dir) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(save_dir)
    else:
        torch.save(unwrapped.state_dict(), save_dir / "pytorch_model.bin")


def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    valid_loader: DataLoader,
    vocab_size: int,
    mask_token_id: int,
    mask_probability: float,
    random_token_min_id: int = 0,
    random_token_max_id: int | None = None,
    single_token_sample_probability: float = 0.70,
    short_span_sample_probability: float = 0.20,
    long_span_sample_probability: float = 0.10,
    short_span_min_length: int = 2,
    short_span_max_length: int = 5,
    long_span_min_length: int = 6,
    long_span_max_length: int = 20,
    max_eval_batches: int | None = None,
) -> dict[str, float]:
    """Run MLM evaluation on the validation split."""

    model.eval()
    losses = []
    token_losses = []
    waveform_mse_losses = []
    top1_correct = []
    top5_correct = []
    top10_correct = []
    top50_correct = []
    masked_counts = []

    with torch.no_grad():
        for batch_index, batch in enumerate(valid_loader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break

            input_ids = get_batch_tensor(batch, "input_ids").to(accelerator.device)
            attention_mask = get_batch_tensor(batch, "attention_mask").to(accelerator.device)
            corrupted, labels = mask_token_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vocab_size=vocab_size,
                mask_token_id=mask_token_id,
                mask_probability=mask_probability,
                random_token_min_id=random_token_min_id,
                random_token_max_id=random_token_max_id,
                single_token_sample_probability=single_token_sample_probability,
                short_span_sample_probability=short_span_sample_probability,
                long_span_sample_probability=long_span_sample_probability,
                short_span_min_length=short_span_min_length,
                short_span_max_length=short_span_max_length,
                long_span_min_length=long_span_min_length,
                long_span_max_length=long_span_max_length,
            )
            outputs = model(
                input_ids=corrupted,
                attention_mask=attention_mask,
                labels=labels,
            )
            losses.append(accelerator.gather_for_metrics(outputs.loss.detach()))
            token_losses.append(accelerator.gather_for_metrics(outputs.token_loss.detach()))
            waveform_mse_losses.append(accelerator.gather_for_metrics(outputs.waveform_mse_loss.detach()))

            masked_positions = labels != -100
            masked_count = masked_positions.sum()
            masked_counts.append(accelerator.gather_for_metrics(masked_count.detach().reshape(1)))

            if masked_count.item() > 0:
                masked_logits = outputs.logits[masked_positions]
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
            "eval/token_loss": float("nan"),
            "eval/waveform_mse_loss": float("nan"),
            "eval/perplexity": float("nan"),
            "eval/top1_accuracy": float("nan"),
            "eval/top5_accuracy": float("nan"),
            "eval/top10_accuracy": float("nan"),
            "eval/top50_accuracy": float("nan"),
        }

    loss = torch.cat([loss.reshape(-1) for loss in losses]).mean().item()
    token_loss = torch.cat([loss.reshape(-1) for loss in token_losses]).mean().item()
    waveform_mse_loss = torch.cat([loss.reshape(-1) for loss in waveform_mse_losses]).mean().item()
    perplexity = math.exp(loss) if loss < 50 else float("inf")
    total_masked = torch.cat(masked_counts).sum().item()
    top1 = torch.cat(top1_correct).sum().item()
    top5 = torch.cat(top5_correct).sum().item()
    top10 = torch.cat(top10_correct).sum().item()
    top50 = torch.cat(top50_correct).sum().item()

    return {
        "eval/loss": loss,
        "eval/token_loss": token_loss,
        "eval/waveform_mse_loss": waveform_mse_loss,
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
    short_span_start_step = int(model_cfg.get("short_span_start_step", 20000))
    long_span_start_step = int(model_cfg.get("long_span_start_step", 60000))
    short_span_min_length = int(model_cfg.get("short_span_min_length", 2))
    short_span_max_length = int(model_cfg.get("short_span_max_length", 5))
    long_span_min_length = int(model_cfg.get("long_span_min_length", 6))
    long_span_max_length = int(model_cfg.get("long_span_max_length", 20))
    resolve_curriculum_mask_probabilities(
        0,
        short_span_start_step=short_span_start_step,
        long_span_start_step=long_span_start_step,
    )

    accelerator = build_accelerator(config)

    train_loader = build_dataloader(config, "train")
    valid_loader = build_dataloader(config, "valid") if config["data"].get("valid_dir") else None

    model = build_bert_mlm_with_waveform_decoder(config)
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

    num_train_epochs, steps_per_epoch, total_training_steps = resolve_epoch_training_steps(
        training_cfg,
        train_loader,
    )
    scheduler = get_scheduler(
        name=str(training_cfg.get("lr_scheduler_type", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        num_training_steps=total_training_steps,
    )

    if config.get("wandb", {}).get("use_wandb", False):
        accelerator.init_trackers(
            project_name=str(config["wandb"].get("project", "poredlm-stage2-bert")),
            config=config,
            init_kwargs={"wandb": {"name": config["wandb"].get("name")}},
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

    global_step = 0
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
    best_eval_loss = float("inf")

    progress_total = total_training_steps if steps_per_epoch is not None else None
    progress = tqdm(total=progress_total, disable=not accelerator.is_local_main_process)
    model.train()

    for epoch in range(num_train_epochs):
        epoch_index = epoch + 1
        if accelerator.is_main_process:
            print(f"Starting epoch {epoch_index}/{num_train_epochs}", flush=True)

        for batch in train_loader:
            with accelerator.accumulate(model):
                input_ids = get_batch_tensor(batch, "input_ids").to(accelerator.device)
                attention_mask = get_batch_tensor(batch, "attention_mask").to(accelerator.device)
                (
                    mask_phase,
                    single_token_sample_probability,
                    short_span_sample_probability,
                    long_span_sample_probability,
                ) = resolve_curriculum_mask_probabilities(
                    global_step,
                    short_span_start_step=short_span_start_step,
                    long_span_start_step=long_span_start_step,
                )
                corrupted, labels = mask_token_ids(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vocab_size=vocab_size,
                    mask_token_id=mask_token_id,
                    mask_probability=mask_probability,
                    random_token_min_id=random_token_min_id,
                    random_token_max_id=random_token_max_id,
                    single_token_sample_probability=single_token_sample_probability,
                    short_span_sample_probability=short_span_sample_probability,
                    long_span_sample_probability=long_span_sample_probability,
                    short_span_min_length=short_span_min_length,
                    short_span_max_length=short_span_max_length,
                    long_span_min_length=long_span_min_length,
                    long_span_max_length=long_span_max_length,
                )
                outputs = model(
                    input_ids=corrupted,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

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
                    token_loss_value = (
                        accelerator.gather_for_metrics(outputs.token_loss.detach()).mean().item()
                    )
                    waveform_mse_loss_value = (
                        accelerator.gather_for_metrics(outputs.waveform_mse_loss.detach()).mean().item()
                    )
                    masked_positions = labels != -100
                    masked_count = (
                        accelerator.gather_for_metrics(
                            masked_positions.sum().detach().reshape(1)
                        )
                        .sum()
                        .item()
                    )
                    valid_count = (
                        accelerator.gather_for_metrics(attention_mask.sum().detach().reshape(1))
                        .sum()
                        .item()
                    )
                    logs = {
                        "train/loss": loss_value,
                        "train/token_loss": token_loss_value,
                        "train/waveform_mse_loss": waveform_mse_loss_value,
                        "train/loss_log10": math.log10(loss_value + 1e-12),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/masked_tokens": masked_count,
                        "train/mask_ratio": masked_count / valid_count if valid_count else float("nan"),
                        "train/mask_phase": mask_phase,
                        "train/single_token_sample_probability": single_token_sample_probability,
                        "train/short_span_sample_probability": short_span_sample_probability,
                        "train/long_span_sample_probability": long_span_sample_probability,
                        "train/epoch": epoch_index,
                        "step": global_step,
                    }
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        print(logs)

                if global_step % save_every == 0:
                    save_checkpoint(accelerator, model, output_dir, global_step)

                if valid_loader is not None and eval_every > 0 and global_step % eval_every == 0:
                    (
                        eval_mask_phase,
                        eval_single_token_sample_probability,
                        eval_short_span_sample_probability,
                        eval_long_span_sample_probability,
                    ) = resolve_curriculum_mask_probabilities(
                        global_step,
                        short_span_start_step=short_span_start_step,
                        long_span_start_step=long_span_start_step,
                    )
                    eval_logs = evaluate(
                        accelerator=accelerator,
                        model=model,
                        valid_loader=valid_loader,
                        vocab_size=vocab_size,
                        mask_token_id=mask_token_id,
                        mask_probability=mask_probability,
                        random_token_min_id=random_token_min_id,
                        random_token_max_id=random_token_max_id,
                        single_token_sample_probability=eval_single_token_sample_probability,
                        short_span_sample_probability=eval_short_span_sample_probability,
                        long_span_sample_probability=eval_long_span_sample_probability,
                        short_span_min_length=short_span_min_length,
                        short_span_max_length=short_span_max_length,
                        long_span_min_length=long_span_min_length,
                        long_span_max_length=long_span_max_length,
                        max_eval_batches=max_eval_batches,
                    )
                    eval_logs["eval/mask_phase"] = eval_mask_phase
                    eval_logs["eval/single_token_sample_probability"] = eval_single_token_sample_probability
                    eval_logs["eval/short_span_sample_probability"] = eval_short_span_sample_probability
                    eval_logs["eval/long_span_sample_probability"] = eval_long_span_sample_probability
                    eval_logs["eval/epoch"] = epoch_index
                    eval_logs["step"] = global_step
                    accelerator.log(eval_logs, step=global_step)
                    if accelerator.is_main_process:
                        print(eval_logs)

                    eval_loss = eval_logs["eval/loss"]
                    if eval_loss == eval_loss and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(accelerator, model, output_dir, "best")

        save_checkpoint(accelerator, model, output_dir, f"epoch_{epoch_index}")

    save_checkpoint(accelerator, model, output_dir, global_step)
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
