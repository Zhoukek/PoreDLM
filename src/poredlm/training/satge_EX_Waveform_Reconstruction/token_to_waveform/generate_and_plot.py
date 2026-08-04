"""Conditionally generate pore tokens and decode them with the Stage-1 codec."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoModel


HERE = Path(__file__).resolve().parent
TOKEN_DATASET_DIR = HERE.parent / "training_waveform_decoder"
if str(TOKEN_DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(TOKEN_DATASET_DIR))

from token_dataset import TokenSequenceDataset  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def parse_indices(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        result = [int(item) for item in value]
    else:
        result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or min(result) < 0:
        raise ValueError("sample_indices must contain non-negative integers.")
    return result


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def register_custom_tokenizer(tokenizer_path: Path) -> None:
    modeling_file = tokenizer_path / "modeling_pore_vq_codec.py"
    if not modeling_file.is_file():
        raise FileNotFoundError(f"Missing tokenizer model code: {modeling_file}")
    if str(tokenizer_path) not in sys.path:
        sys.path.insert(0, str(tokenizer_path))
    importlib.import_module("modeling_pore_vq_codec")


def freeze(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def select_samples(config: dict[str, Any], indices: list[int]) -> list[dict[str, object]]:
    data = config["data"]
    dataset = TokenSequenceDataset(
        [str(data["eval_path"])],
        pattern=str(data.get("file_pattern", "*.npy")),
        dtype=str(data.get("token_dtype", "uint32")),
        shuffle_files=False,
        repeat=False,
    )
    wanted = set(indices)
    selected: dict[int, dict[str, object]] = {}
    for index, sample in enumerate(dataset):
        if index in wanted:
            selected[index] = sample
            if len(selected) == len(wanted):
                break
    missing = sorted(wanted - selected.keys())
    if missing:
        raise IndexError(f"Sample indices not found in eval dataset: {missing}")
    return [selected[index] for index in indices]


def stage1_decode(tokenizer: torch.nn.Module, codec_ids: torch.Tensor) -> torch.Tensor:
    vq = getattr(tokenizer, "vq", None)
    decoder = getattr(getattr(tokenizer, "cnn_model", None), "decoder", None)
    if vq is None or not hasattr(vq, "get_output_from_indices"):
        raise ValueError("Tokenizer does not expose vq.get_output_from_indices().")
    if decoder is None:
        raise ValueError("Tokenizer does not expose cnn_model.decoder.")
    embeddings = vq.get_output_from_indices(codec_ids)
    if embeddings.ndim != 3:
        raise ValueError(f"Expected codebook embeddings [B,T,D], got {embeddings.shape}.")
    return decoder(embeddings.transpose(1, 2)).squeeze(1)


def normalize_raw_tokens(
    raw: torch.Tensor,
    bos_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    raw = torch.as_tensor(raw, dtype=torch.long).flatten()
    if raw.numel() < 3:
        raise ValueError("A sample must contain BOS, at least one content token, and EOS.")
    if int(raw[0]) != bos_token_id or int(raw[-1]) != eos_token_id:
        raise ValueError(
            f"Expected BOS/EOS={bos_token_id}/{eos_token_id}, got {int(raw[0])}/{int(raw[-1])}."
        )
    return raw


def validate_or_fix_generated_ids(
    token_ids: torch.Tensor,
    token_offset: int,
    codebook_size: int,
    policy: str,
) -> tuple[torch.Tensor, int]:
    if policy not in {"error", "clip"}:
        raise ValueError("invalid_token_policy must be 'error' or 'clip'.")
    low, high = token_offset, token_offset + codebook_size - 1
    invalid = (token_ids < low) | (token_ids > high)
    count = int(invalid.sum().item())
    if count and policy == "error":
        values = torch.unique(token_ids[invalid]).detach().cpu().tolist()[:20]
        raise ValueError(
            f"DLM generated {count} non-codebook tokens; valid DLM IDs are [{low}, {high}]. "
            f"Examples: {values}. Use invalid_token_policy=clip only for diagnostic plotting."
        )
    if count:
        token_ids = token_ids.clamp(low, high)
    return token_ids, count


def waveform_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    length = min(a.size, b.size)
    a, b = a[:length], b[:length]
    mse = float(np.mean((a - b) ** 2))
    if np.std(a) == 0 or np.std(b) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return {"mse": mse, "pearson_r": correlation}


def aligned_mse(
    first: np.ndarray,
    second: np.ndarray,
    start: int = 0,
    end: int | None = None,
) -> float:
    length = min(first.size, second.size)
    stop = length if end is None else min(int(end), length)
    if start >= stop:
        return float("nan")
    return float(np.mean((first[start:stop] - second[start:stop]) ** 2))


def bert_repair(
    dlm: torch.nn.Module,
    masked_input_ids: torch.Tensor,
    masked_positions: torch.Tensor,
    token_offset: int,
    codebook_size: int,
) -> torch.Tensor:
    """Repair masks with the embedded Stage-2 BERT and its own MLM head only."""
    adapter = getattr(dlm, "context_encoder", None)
    bert_mlm = getattr(adapter, "model", None)
    if bert_mlm is None:
        raise ValueError("DLM does not expose a context_encoder for BERT reconstruction.")

    # The converted conditional DLM stores the complete Stage-2 MLM at
    # context_encoder.model.  Never use elf_denoiser (the DLM token decoder)
    # for this BERT baseline.
    bert_lm_head = getattr(bert_mlm, "lm_head", None)
    if bert_lm_head is None:
        raise ValueError(
            "context_encoder.model has no Stage-2 BERT lm_head; refusing to "
            "fall back to the DLM decoder head. Re-convert from a Stage-2 MLM checkpoint."
        )

    outputs = bert_mlm(
        input_ids=masked_input_ids,
        attention_mask=torch.ones_like(masked_input_ids),
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError(
                "The embedded Stage-2 BERT returned neither MLM logits nor hidden states."
            )
        logits = bert_lm_head(hidden)
    vocab_end = token_offset + codebook_size
    if logits.shape[-1] < vocab_end:
        raise ValueError(
            f"BERT vocabulary size {logits.shape[-1]} is smaller than codebook end {vocab_end}."
        )
    predicted = torch.argmax(logits[..., token_offset:vocab_end], dim=-1) + token_offset
    repaired = masked_input_ids.clone()
    repaired[masked_positions] = predicted[masked_positions]
    return repaired


def plot_waveforms(
    stage1_waveform: np.ndarray,
    bert_waveform: np.ndarray,
    reference_waveform: np.ndarray,
    mask_start_token: int,
    mask_length_tokens: int,
    total_tokens: int,
    title: str,
    output_path: Path,
) -> None:
    common = min(stage1_waveform.size, bert_waveform.size, reference_waveform.size)
    stage1_waveform = stage1_waveform[:common]
    bert_waveform = bert_waveform[:common]
    reference_waveform = reference_waveform[:common]
    region_start = min(int(round(common * mask_start_token / total_tokens)), common - 1)
    region_end = min(
        int(round(common * (mask_start_token + mask_length_tokens) / total_tokens)),
        common,
    )
    dlm_region_mse = aligned_mse(reference_waveform, stage1_waveform, region_start, region_end)
    bert_region_mse = aligned_mse(reference_waveform, bert_waveform, region_start, region_end)
    figure, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)
    x = np.arange(common)
    black_label = "Reference tokens → Stage-1 decoder"
    blue_label = "DLM repaired tokens → Stage-1 decoder"
    red_label = "BERT repaired tokens → Stage-1 decoder"
    axes[0].plot(x, reference_waveform, color="black", alpha=0.6, linewidth=0.8, label=black_label)
    axes[0].plot(x, stage1_waveform, color="#0072B2", linewidth=0.8, label=blue_label)
    axes[0].set_title(f"Full sequence: Reference vs DLM\nmasked-region MSE={dlm_region_mse:.6g}")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(x[region_start:region_end], reference_waveform[region_start:region_end], color="black", alpha=0.65, linewidth=0.9, label=black_label)
    axes[1].plot(x[region_start:region_end], stage1_waveform[region_start:region_end], color="#0072B2", linewidth=0.9, label=blue_label)
    axes[1].set_title(f"Masked region: Reference vs DLM\nMSE={dlm_region_mse:.6g}")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(x[region_start:region_end], reference_waveform[region_start:region_end], color="black", alpha=0.65, linewidth=0.9, label=black_label)
    axes[2].plot(x[region_start:region_end], bert_waveform[region_start:region_end], color="#D62728", linewidth=0.9, label=red_label)
    axes[2].set_title(f"Masked region: Reference vs BERT\nMSE={bert_region_mse:.6g}")
    axes[2].legend(loc="upper right", fontsize=8)

    for axis in axes:
        axis.axvline(region_start, color="#009E73", linestyle="--", linewidth=1.0)
        axis.axvline(region_end, color="#009E73", linestyle="--", linewidth=1.0)
        axis.set_xlabel("Waveform sample")
        axis.set_ylabel("Normalized current")
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def process_sample(
    sample: dict[str, object],
    sample_index: int,
    dlm: torch.nn.Module,
    tokenizer: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    generation = config["generation"]
    data = config["data"]
    total_content_length = int(generation["total_length"])
    mask_start = int(generation["mask_start"])
    mask_length = int(generation["mask_length"])
    mask_end = mask_start + mask_length
    if mask_start < 0 or mask_length < 1 or mask_end > total_content_length:
        raise ValueError("Require 0 <= mask_start < mask_start + mask_length <= total_length.")
    bos = int(data.get("bos_token_id", 2))
    eos = int(data.get("eos_token_id", 3))
    offset = int(data.get("token_offset", 128))
    codebook_size = int(data.get("codebook_size", 65536))
    raw = normalize_raw_tokens(torch.as_tensor(sample["tokens"]), bos, eos)
    reference_content = raw[1:-1]
    if reference_content.numel() < total_content_length:
        raise ValueError(
            f"Sample {sample_index} contains {reference_content.numel()} content tokens, "
            f"but {total_content_length} are required."
        )
    reference_content = reference_content[:total_content_length].to(device)
    full_ids = torch.cat(
        [torch.tensor([bos], device=device), reference_content]
    ).unsqueeze(0)
    condition_token_mask = torch.ones_like(full_ids, dtype=torch.bool)
    masked_positions = torch.zeros_like(full_ids, dtype=torch.bool)
    masked_positions[:, 1 + mask_start : 1 + mask_end] = True
    condition_token_mask[masked_positions] = False
    context_core = getattr(getattr(dlm, "context_encoder", None), "model", getattr(dlm, "context_encoder", None))
    mask_token_id = int(
        generation.get("mask_token_id")
        if generation.get("mask_token_id") is not None
        else getattr(getattr(context_core, "config", None), "mask_token_id", 1)
    )
    masked_input_ids = full_ids.clone()
    masked_input_ids[masked_positions] = mask_token_id
    generation_output = dlm.generate(
        condition_input_ids=masked_input_ids,
        condition_attention_mask=torch.ones_like(masked_input_ids),
        condition_token_mask=condition_token_mask,
        max_length=1 + total_content_length,
        num_steps=int(generation.get("num_steps", 50)),
        sampling_method=str(generation.get("sampling_method", "ode")),
        sde_gamma=float(generation.get("sde_gamma", 0.1)),
        cfg_scale=float(generation.get("cfg_scale", 1.0)),
        self_cond_cfg_scale=float(generation.get("self_cond_cfg_scale", 1.0)),
        seed=int(generation.get("seed", 6198)) + sample_index,
        return_dict=True,
        return_latents=False,
    )
    if not isinstance(generation_output, dict):
        raise TypeError("DLM generate() must return a dict when return_dict=True.")
    if "sequences" not in generation_output:
        raise KeyError(
            f"DLM generation output must contain sequences; keys={list(generation_output)}"
        )
    generated = generation_output["sequences"]
    if bool(generation.get("restrict_to_codebook", True)):
        logits = generation_output.get("logits")
        if logits is None:
            raise KeyError("restrict_to_codebook=True requires logits in DLM generation output.")
        vocab_end = offset + codebook_size
        if logits.shape[-1] < vocab_end:
            raise ValueError(
                f"DLM vocabulary size {logits.shape[-1]} is smaller than required codebook end {vocab_end}."
            )
        generated = generated.clone()
        constrained_ids = torch.argmax(logits[..., offset:vocab_end], dim=-1) + offset
        generated[masked_positions] = constrained_ids[masked_positions]
    generated_content = generated[0, 1 : 1 + total_content_length]
    generated_content, invalid_count = validate_or_fix_generated_ids(
        generated_content,
        offset,
        codebook_size,
        str(generation.get("invalid_token_policy", "error")),
    )
    generated_codec = (generated_content - offset).unsqueeze(0)
    bert_ids = bert_repair(
        dlm, masked_input_ids, masked_positions, offset, codebook_size
    )
    bert_content = bert_ids[0, 1 : 1 + total_content_length]
    bert_content, bert_invalid_count = validate_or_fix_generated_ids(
        bert_content, offset, codebook_size, "error"
    )
    bert_codec = (bert_content - offset).unsqueeze(0)
    reference_codec = (reference_content - offset).unsqueeze(0)
    if int(reference_codec.min()) < 0 or int(reference_codec.max()) >= codebook_size:
        raise ValueError(f"Reference sample {sample_index} contains invalid codebook IDs.")

    stage1_waveform = stage1_decode(tokenizer, generated_codec).float()
    bert_reconstructed_waveform = stage1_decode(tokenizer, bert_codec).float()
    reference_waveform = stage1_decode(tokenizer, reference_codec).float()

    stage1_np = stage1_waveform[0].detach().cpu().numpy()
    bert_np = bert_reconstructed_waveform[0].detach().cpu().numpy()
    reference_np = reference_waveform[0].detach().cpu().numpy()
    output_dir = Path(config["output"]["directory"]).expanduser()
    stem = f"sample_{sample_index:06d}_mask{mask_start}-{mask_end}"
    plot_waveforms(
        stage1_np,
        bert_np,
        reference_np,
        mask_start,
        mask_length,
        total_content_length,
        f"{stem} | id={sample.get('id', '')}",
        output_dir / f"{stem}.png",
    )
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        generated_dlm_ids=generated_content.detach().cpu().numpy(),
        generated_codec_ids=generated_codec[0].detach().cpu().numpy(),
        bert_repaired_dlm_ids=bert_content.detach().cpu().numpy(),
        bert_repaired_codec_ids=bert_codec[0].detach().cpu().numpy(),
        reference_dlm_ids=reference_content.detach().cpu().numpy(),
        stage1_waveform=stage1_np,
        bert_waveform=bert_np,
        reference_waveform=reference_np,
    )
    masked_region = slice(mask_start, mask_end)
    dlm_token_accuracy = float(
        (generated_content[masked_region] == reference_content[masked_region])
        .float()
        .mean()
        .item()
    )
    bert_token_accuracy = float(
        (bert_content[masked_region] == reference_content[masked_region])
        .float()
        .mean()
        .item()
    )
    waveform_length = min(reference_np.size, stage1_np.size, bert_np.size)
    waveform_start = int(waveform_length * mask_start / total_content_length)
    waveform_end = int(waveform_length * mask_end / total_content_length)
    return {
        "sample_index": sample_index,
        "sample_id": str(sample.get("id", "")),
        "total_length": total_content_length,
        "mask_start": mask_start,
        "mask_length": mask_length,
        "invalid_generated_tokens": invalid_count,
        "invalid_bert_tokens": bert_invalid_count,
        "dlm_masked_region_token_accuracy": dlm_token_accuracy,
        "bert_masked_region_token_accuracy": bert_token_accuracy,
        "stage1_generated_vs_reference_full_metrics": waveform_metrics(stage1_np, reference_np),
        "bert_vs_reference_full_metrics": waveform_metrics(bert_np, reference_np),
        "mse": {
            "reference_vs_dlm_masked_region": aligned_mse(reference_np, stage1_np, waveform_start, waveform_end),
            "reference_vs_bert_masked_region": aligned_mse(reference_np, bert_np, waveform_start, waveform_end),
        },
        "figure": str(output_dir / f"{stem}.png"),
        "arrays": str(output_dir / f"{stem}.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--total-length", type=int)
    parser.add_argument("--mask-start", type=int)
    parser.add_argument("--mask-length", type=int)
    parser.add_argument("--sample-indices")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.total_length is not None:
        config["generation"]["total_length"] = args.total_length
    if args.mask_start is not None:
        config["generation"]["mask_start"] = args.mask_start
    if args.mask_length is not None:
        config["generation"]["mask_length"] = args.mask_length
    if args.sample_indices is not None:
        config["data"]["sample_indices"] = args.sample_indices
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)

    device = resolve_device(str(config.get("device", "auto")))
    models = config["models"]
    dlm_path = Path(models["dlm_path"]).expanduser().resolve()
    tokenizer_path = Path(models["tokenizer_path"]).expanduser().resolve()
    register_custom_tokenizer(tokenizer_path)
    dlm = freeze(
        AutoModel.from_pretrained(
            str(dlm_path), trust_remote_code=True, local_files_only=True
        ),
        device,
    )
    if not hasattr(dlm, "generate"):
        raise AttributeError(f"DLM at {dlm_path} does not provide conditional generate().")
    tokenizer = freeze(
        AutoModel.from_pretrained(
            str(tokenizer_path), trust_remote_code=True, local_files_only=True
        ),
        device,
    )
    output_dir = Path(config["output"]["directory"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = parse_indices(config["data"].get("sample_indices", [0]))
    samples = select_samples(config, indices)
    results = [
        process_sample(
            sample,
            index,
            dlm,
            tokenizer,
            config,
            device,
        )
        for index, sample in zip(indices, samples)
    ]
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
