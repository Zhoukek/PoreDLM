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
from transformers import AutoModel, AutoModelForCausalLM, __version__ as transformers_version


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
    """Register either the VQ codec or the RSQ/FSQ codec shipped with a HF model."""
    config_path = tokenizer_path / "config.json"
    module_names: list[str] = []
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            hf_config = json.load(handle)
        auto_map = hf_config.get("auto_map", {})
        auto_model = auto_map.get("AutoModel") if isinstance(auto_map, dict) else None
        if isinstance(auto_model, str) and "." in auto_model:
            module_names.append(auto_model.split(".", 1)[0])
    module_names.extend(("modeling_pore_codec", "modeling_pore_vq_codec"))
    module_names = list(dict.fromkeys(module_names))
    modeling_file = next(
        (tokenizer_path / f"{name}.py" for name in module_names if (tokenizer_path / f"{name}.py").is_file()),
        None,
    )
    if modeling_file is None:
        expected = ", ".join(f"{name}.py" for name in module_names)
        raise FileNotFoundError(f"Missing codec model code under {tokenizer_path}; tried {expected}.")
    if str(tokenizer_path) not in sys.path:
        sys.path.insert(0, str(tokenizer_path))
    module_name = modeling_file.stem
    if module_name in sys.modules:
        loaded_path = Path(getattr(sys.modules[module_name], "__file__", "")).resolve()
        if loaded_path != modeling_file.resolve():
            raise RuntimeError(
                f"Python module {module_name!r} was already loaded from {loaded_path}, "
                f"not the requested codec {modeling_file}."
            )
    else:
        importlib.import_module(module_name)


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


def stage1_decode(
    tokenizer: torch.nn.Module,
    codec_ids: torch.Tensor,
    codec_layer: int = 0,
) -> torch.Tensor:
    # PoreRSQCodec stores a packed token per time step. Its public decoder first
    # expands that token into residual-FSQ level indices, then reconstructs signal.
    decode_token = getattr(tokenizer, "decode_token", None)
    if callable(decode_token):
        if getattr(tokenizer, "fsq_levels", None) is not None:
            waveform = decode_token(codec_ids, layer=codec_layer)
        else:
            waveform = decode_token(codec_ids)
        return waveform.squeeze(1) if waveform.ndim == 3 and waveform.shape[1] == 1 else waveform

    # PoreVQCodec uses one ordinary codebook index per time step.
    vq = getattr(tokenizer, "vq", None)
    decoder = getattr(getattr(tokenizer, "cnn_model", None), "decoder", None)
    if vq is None or not hasattr(vq, "get_output_from_indices"):
        raise ValueError(
            "Codec exposes neither decode_token() nor vq.get_output_from_indices()."
        )
    if decoder is None:
        raise ValueError("Tokenizer does not expose cnn_model.decoder.")
    embeddings = vq.get_output_from_indices(codec_ids)
    if embeddings.ndim != 3:
        raise ValueError(f"Expected codebook embeddings [B,T,D], got {embeddings.shape}.")
    return decoder(embeddings.transpose(1, 2)).squeeze(1)


def infer_codec_vocabulary_size(
    codec: torch.nn.Module,
    codec_layer: int = 0,
) -> int | None:
    """Return the number of packed codec tokens when the codec exposes it."""
    fsq_levels = getattr(codec, "fsq_levels", None)
    num_quantizers = getattr(codec, "num_quantizers", None)
    if fsq_levels is not None and num_quantizers is not None:
        base_size = int(np.prod([int(level) for level in fsq_levels]))
        active_layers = int(num_quantizers) if codec_layer == 0 else int(codec_layer)
        if active_layers < 1 or active_layers > int(num_quantizers):
            raise ValueError(
                f"codec_layer must be 0 or in [1, {int(num_quantizers)}], got {codec_layer}."
            )
        return base_size ** active_layers
    codebook_size = getattr(codec, "codebook_size", None)
    if codebook_size is None:
        codebook_size = getattr(getattr(codec, "config", None), "codebook_size", None)
    return int(codebook_size) if codebook_size is not None else None


def infer_codec_downsample_rate(codec: torch.nn.Module) -> int:
    """Return the number of waveform samples represented by one codec token."""
    candidates = (
        getattr(codec, "cnn_stride", None),
        getattr(codec, "downsample_rate", None),
        getattr(getattr(codec, "cnn_model", None), "stride", None),
        getattr(getattr(codec, "config", None), "cnn_stride", None),
        getattr(getattr(codec, "config", None), "downsample_rate", None),
    )
    for value in candidates:
        if value is None:
            continue
        rate = int(value)
        if rate > 0:
            return rate
    raise ValueError("Could not infer a positive downsample rate from the selected codec.")


def token_span_to_waveform_span(
    token_start: int,
    token_end: int,
    downsample_rate: int,
    waveform_length: int,
) -> tuple[int, int]:
    """Map a half-open content-token span to its logical waveform span."""
    if downsample_rate <= 0:
        raise ValueError(f"downsample_rate must be positive, got {downsample_rate}.")
    start = min(max(0, int(token_start) * downsample_rate), waveform_length)
    end = min(max(start, int(token_end) * downsample_rate), waveform_length)
    return start, end


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
            f"Generator produced {count} non-codebook tokens; valid IDs are [{low}, {high}]. "
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
    bert_source: torch.nn.Module,
    masked_input_ids: torch.Tensor,
    masked_positions: torch.Tensor,
    token_offset: int,
    codebook_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repair masks with a standalone or DLM-embedded Stage-2 BERT MLM."""
    if getattr(bert_source, "lm_head", None) is not None:
        bert_mlm = bert_source
    else:
        adapter = getattr(bert_source, "context_encoder", None)
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
    raw_predicted = torch.argmax(logits, dim=-1)
    legal_predicted = torch.argmax(logits[..., token_offset:vocab_end], dim=-1) + token_offset
    predicted = torch.where(
        (raw_predicted >= token_offset) & (raw_predicted < vocab_end),
        raw_predicted,
        legal_predicted,
    )
    repaired = masked_input_ids.clone()
    repaired[masked_positions] = predicted[masked_positions]
    return repaired, raw_predicted[masked_positions]


def gpt_repair(
    gpt: torch.nn.Module,
    full_ids: torch.Tensor,
    mask_start: int,
    mask_length: int,
    token_offset: int,
    codebook_size: int,
) -> torch.Tensor:
    """Autoregressively replace a content-token interval with a causal LM."""
    # full_ids includes BOS, while mask_start is in content-token coordinates.
    prefix_length = 1 + mask_start
    prefix = full_ids[:, :prefix_length]
    if prefix.shape[1] < 1:
        raise ValueError("GPT generation requires at least one prefix token.")
    max_positions = getattr(gpt.config, "max_position_embeddings", None)
    if max_positions is not None and prefix_length + mask_length > int(max_positions):
        raise ValueError(
            f"GPT needs {prefix_length + mask_length} positions, but "
            f"max_position_embeddings={max_positions}."
        )

    outputs = gpt(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
    past_key_values = getattr(outputs, "past_key_values", None)
    vocab_end = token_offset + codebook_size
    logits = outputs.logits[:, -1, :]
    if logits.shape[-1] < vocab_end:
        raise ValueError(
            f"GPT vocabulary size {logits.shape[-1]} is smaller than codebook end {vocab_end}."
        )

    predicted: list[torch.Tensor] = []
    for step in range(mask_length):
        next_token = torch.argmax(logits[:, token_offset:vocab_end], dim=-1) + token_offset
        predicted.append(next_token)
        if step + 1 == mask_length:
            break
        step_input = next_token.unsqueeze(1)
        if past_key_values is None:
            # Compatibility fallback for causal models that do not expose KV cache.
            running = torch.cat([prefix, torch.stack(predicted, dim=1)], dim=1)
            outputs = gpt(
                input_ids=running,
                attention_mask=torch.ones_like(running),
                use_cache=False,
            )
        else:
            outputs = gpt(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)
        logits = outputs.logits[:, -1, :]

    repaired = full_ids.clone()
    repaired[:, prefix_length : prefix_length + mask_length] = torch.stack(predicted, dim=1)
    return repaired


def plot_waveforms(
    stage1_waveform: np.ndarray,
    reference_waveform: np.ndarray,
    mask_start_token: int,
    mask_length_tokens: int,
    downsample_rate: int,
    title: str,
    output_path: Path,
    generator_name: str,
    bert_waveform: np.ndarray | None = None,
) -> None:
    sizes = [stage1_waveform.size, reference_waveform.size]
    if bert_waveform is not None:
        sizes.append(bert_waveform.size)
    common = min(sizes)
    stage1_waveform = stage1_waveform[:common]
    if bert_waveform is not None:
        bert_waveform = bert_waveform[:common]
    reference_waveform = reference_waveform[:common]
    region_start, region_end = token_span_to_waveform_span(
        mask_start_token,
        mask_start_token + mask_length_tokens,
        downsample_rate,
        common,
    )
    generated_region_mse = aligned_mse(reference_waveform, stage1_waveform, region_start, region_end)
    panel_count = 3 if bert_waveform is not None else 2
    figure, axes = plt.subplots(1, panel_count, figsize=(8 * panel_count, 6), constrained_layout=True)
    x = np.arange(common)
    black_label = "Reference tokens → Stage-1 decoder"
    blue_label = f"{generator_name} repaired tokens → Stage-1 decoder"
    axes[0].plot(x, reference_waveform, color="black", alpha=0.6, linewidth=0.8, label=black_label)
    axes[0].plot(x, stage1_waveform, color="#0072B2", linewidth=0.8, label=blue_label)
    axes[0].set_title(f"Full sequence: Reference vs {generator_name}\nmasked-region MSE={generated_region_mse:.6g}")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(x[region_start:region_end], reference_waveform[region_start:region_end], color="black", alpha=0.65, linewidth=0.9, label=black_label)
    axes[1].plot(x[region_start:region_end], stage1_waveform[region_start:region_end], color="#0072B2", linewidth=0.9, label=blue_label)
    axes[1].set_title(f"Masked region: Reference vs {generator_name}\nMSE={generated_region_mse:.6g}")
    axes[1].legend(loc="upper right", fontsize=8)

    if bert_waveform is not None:
        bert_region_mse = aligned_mse(reference_waveform, bert_waveform, region_start, region_end)
        red_label = "BERT repaired tokens → Stage-1 decoder"
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
    generator_model: torch.nn.Module,
    generator_type: str,
    bert_model: torch.nn.Module | None,
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
    codec_layer = int(data.get("codec_layer", 0))
    downsample_rate = infer_codec_downsample_rate(tokenizer)
    raw = normalize_raw_tokens(torch.as_tensor(sample["tokens"]), bos, eos)
    available_reference_content = raw[1:-1].to(device)
    if available_reference_content.numel() < total_content_length:
        raise ValueError(
            f"Sample {sample_index} contains {available_reference_content.numel()} content tokens, "
            f"but {total_content_length} are required."
        )
    reference_content = available_reference_content[:total_content_length]
    full_ids = torch.cat(
        [
            torch.tensor([bos], device=device),
            reference_content,
            torch.tensor([eos], device=device),
        ]
    ).unsqueeze(0)
    masked_positions = torch.zeros_like(full_ids, dtype=torch.bool)
    masked_positions[:, 1 + mask_start : 1 + mask_end] = True
    bert_content: torch.Tensor | None = None
    bert_raw_region: torch.Tensor | None = None
    if generator_type == "dlm":
        condition_token_mask = ~masked_positions
        bert_source = bert_model if bert_model is not None else generator_model
        context_core = (
            bert_source if getattr(bert_source, "lm_head", None) is not None
            else getattr(getattr(bert_source, "context_encoder", None), "model", None)
        )
        mask_token_id = int(
            generation.get("mask_token_id")
            if generation.get("mask_token_id") is not None
            else getattr(getattr(context_core, "config", None), "mask_token_id", 1)
        )
        masked_input_ids = full_ids.clone()
        masked_input_ids[masked_positions] = mask_token_id
        generation_output = generator_model.generate(
            condition_input_ids=masked_input_ids,
            condition_attention_mask=torch.ones_like(masked_input_ids),
            condition_token_mask=condition_token_mask,
            max_length=2 + total_content_length,
            num_steps=int(generation.get("num_steps", 50)),
            sampling_method=str(generation.get("sampling_method", "ode")),
            sde_gamma=float(generation.get("sde_gamma", 0.1)),
            cfg_scale=float(generation.get("cfg_scale", 1.0)),
            self_cond_cfg_scale=float(generation.get("self_cond_cfg_scale", 1.0)),
            seed=int(generation.get("seed", 6198)) + sample_index,
            return_dict=True,
            return_latents=False,
        )
        if not isinstance(generation_output, dict) or "sequences" not in generation_output:
            raise TypeError("DLM generate() must return a dict containing 'sequences'.")
        generated = generation_output["sequences"]
        if bool(generation.get("restrict_to_codebook", True)):
            logits = generation_output.get("logits")
            if logits is None:
                raise KeyError("restrict_to_codebook=True requires DLM logits.")
            vocab_end = offset + codebook_size
            if logits.shape[-1] < vocab_end:
                raise ValueError(
                    f"DLM vocabulary size {logits.shape[-1]} is smaller than codebook end {vocab_end}."
                )
            generated = generated.clone()
            constrained_ids = torch.argmax(logits[..., offset:vocab_end], dim=-1) + offset
            generated[masked_positions] = constrained_ids[masked_positions]
        # Mirror stage2_BERT_trian/eval: BERT sees [BOS, content, EOS] and keeps
        # as much right context as its own maximum position length permits.
        bert_max_positions = int(
            getattr(getattr(context_core, "config", None), "max_position_embeddings", total_content_length + 2)
        )
        bert_reference_content = available_reference_content[: bert_max_positions - 2]
        if mask_end > bert_reference_content.numel():
            raise ValueError("BERT masked interval exceeds its available content context.")
        bert_input_ids = torch.cat(
            [
                torch.tensor([bos], dtype=full_ids.dtype, device=device),
                bert_reference_content,
                torch.tensor([eos], dtype=full_ids.dtype, device=device),
            ]
        ).unsqueeze(0)
        bert_masked_positions = torch.zeros_like(bert_input_ids, dtype=torch.bool)
        bert_masked_positions[:, 1 + mask_start : 1 + mask_end] = True
        bert_input_ids[bert_masked_positions] = mask_token_id
        bert_ids, bert_raw_region = bert_repair(
            bert_source, bert_input_ids, bert_masked_positions, offset, codebook_size
        )
        bert_content = bert_ids[0, 1 : 1 + total_content_length]
        bert_content, _ = validate_or_fix_generated_ids(
            bert_content, offset, codebook_size, "error"
        )
    elif generator_type == "gpt":
        generated = gpt_repair(
            generator_model, full_ids, mask_start, mask_length, offset, codebook_size
        )
    else:
        raise ValueError("generator_type must be 'dlm' or 'gpt'.")
    generated_content = generated[0, 1 : 1 + total_content_length]
    generated_content, invalid_count = validate_or_fix_generated_ids(
        generated_content,
        offset,
        codebook_size,
        str(generation.get("invalid_token_policy", "error")),
    )
    generated_codec = (generated_content - offset).unsqueeze(0)
    reference_codec = (reference_content - offset).unsqueeze(0)
    if int(reference_codec.min()) < 0 or int(reference_codec.max()) >= codebook_size:
        raise ValueError(f"Reference sample {sample_index} contains invalid codebook IDs.")

    stage1_waveform = stage1_decode(tokenizer, generated_codec, codec_layer).float()
    reference_waveform = stage1_decode(tokenizer, reference_codec, codec_layer).float()
    bert_reconstructed_waveform = None
    if bert_content is not None:
        bert_codec = (bert_content - offset).unsqueeze(0)
        bert_reconstructed_waveform = stage1_decode(tokenizer, bert_codec, codec_layer).float()

    stage1_np = stage1_waveform[0].detach().cpu().numpy()
    bert_np = (
        bert_reconstructed_waveform[0].detach().cpu().numpy()
        if bert_reconstructed_waveform is not None else None
    )
    reference_np = reference_waveform[0].detach().cpu().numpy()
    output_dir = Path(config["output"]["directory"]).expanduser()
    stem = f"{generator_type}_sample_{sample_index:06d}_mask{mask_start}-{mask_end}"
    plot_waveforms(
        stage1_np,
        reference_np,
        mask_start,
        mask_length,
        downsample_rate,
        f"{stem} | id={sample.get('id', '')}",
        output_dir / f"{stem}.png",
        generator_type.upper(),
        bert_waveform=bert_np,
    )
    arrays = {
        "generated_token_ids": generated_content.detach().cpu().numpy(),
        "generated_codec_ids": generated_codec[0].detach().cpu().numpy(),
        "reference_token_ids": reference_content.detach().cpu().numpy(),
        "generated_waveform": stage1_np,
        "reference_waveform": reference_np,
    }
    if bert_content is not None and bert_np is not None:
        arrays["bert_repaired_token_ids"] = bert_content.detach().cpu().numpy()
        arrays["bert_waveform"] = bert_np
        if bert_raw_region is not None:
            arrays["bert_raw_predicted_region_token_ids"] = bert_raw_region.detach().cpu().numpy()
    np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
    masked_region = slice(mask_start, mask_end)
    generated_token_accuracy = float(
        (generated_content[masked_region] == reference_content[masked_region])
        .float()
        .mean()
        .item()
    )
    waveform_length = min(reference_np.size, stage1_np.size)
    waveform_start, waveform_end = token_span_to_waveform_span(
        mask_start,
        mask_end,
        downsample_rate,
        waveform_length,
    )
    result = {
        "generator_type": generator_type,
        "sample_index": sample_index,
        "sample_id": str(sample.get("id", "")),
        "total_length": total_content_length,
        "mask_start": mask_start,
        "mask_length": mask_length,
        "token_offset": offset,
        "codebook_size": codebook_size,
        "codec_layer": codec_layer,
        "downsample_rate": downsample_rate,
        "invalid_generated_tokens": invalid_count,
        "masked_region_token_accuracy": generated_token_accuracy,
        "reference_region_token_ids": reference_content[masked_region].detach().cpu().tolist(),
        "predicted_region_token_ids": generated_content[masked_region].detach().cpu().tolist(),
        "generated_vs_reference_full_metrics": waveform_metrics(stage1_np, reference_np),
        "mse": {
            "reference_vs_generated_masked_region": aligned_mse(reference_np, stage1_np, waveform_start, waveform_end),
        },
        "figure": str(output_dir / f"{stem}.png"),
        "arrays": str(output_dir / f"{stem}.npz"),
    }
    if bert_content is not None and bert_np is not None:
        result["bert_predicted_region_token_ids"] = bert_content[masked_region].detach().cpu().tolist()
        if bert_raw_region is not None:
            result["bert_raw_predicted_region_token_ids"] = bert_raw_region.detach().cpu().tolist()
            result["bert_raw_masked_region_token_accuracy"] = float(
                (bert_raw_region == reference_content[masked_region]).float().mean().item()
            )
        result["bert_masked_region_token_accuracy"] = float(
            (bert_content[masked_region] == reference_content[masked_region]).float().mean().item()
        )
        result["bert_vs_reference_full_metrics"] = waveform_metrics(bert_np, reference_np)
        result["mse"]["reference_vs_bert_masked_region"] = aligned_mse(
            reference_np, bert_np, waveform_start, waveform_end
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--total-length", type=int)
    parser.add_argument("--mask-start", type=int)
    parser.add_argument("--mask-length", type=int)
    parser.add_argument("--sample-indices")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generator-type", choices=("dlm", "gpt"))
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
    if args.generator_type is not None:
        config["generation"]["generator_type"] = args.generator_type

    device = resolve_device(str(config.get("device", "auto")))
    models = config["models"]
    generator_type = str(config["generation"].get("generator_type", "dlm")).lower()
    if generator_type not in {"dlm", "gpt"}:
        raise ValueError("generation.generator_type must be 'dlm' or 'gpt'.")
    model_path = Path(models[f"{generator_type}_path"]).expanduser().resolve()
    tokenizer_path_value = models.get(f"{generator_type}_tokenizer_path", models.get("tokenizer_path"))
    if tokenizer_path_value is None:
        raise KeyError(
            f"models must define {generator_type}_tokenizer_path or tokenizer_path."
        )
    tokenizer_path = Path(tokenizer_path_value).expanduser().resolve()
    register_custom_tokenizer(tokenizer_path)
    model_loader = AutoModel if generator_type == "dlm" else AutoModelForCausalLM
    try:
        loaded_generator = model_loader.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True
        )
    except ValueError as error:
        if generator_type == "gpt" and "model type `olmo2`" in str(error):
            raise RuntimeError(
                f"GPT checkpoint {model_path} is OLMo2, but installed Transformers "
                f"{transformers_version} does not support it. Install Transformers >=4.48,<5 "
                f"in the active environment, then rerun."
            ) from error
        raise
    generator_model = freeze(loaded_generator, device)
    if generator_type == "dlm" and not hasattr(generator_model, "generate"):
        raise AttributeError(f"DLM at {model_path} does not provide conditional generate().")
    tokenizer = freeze(
        AutoModel.from_pretrained(
            str(tokenizer_path), trust_remote_code=True, local_files_only=True
        ),
        device,
    )
    bert_model = None
    if generator_type == "dlm" and models.get("bert_path"):
        bert_model = freeze(
            AutoModel.from_pretrained(
                str(Path(models["bert_path"]).expanduser().resolve()),
                trust_remote_code=True,
                local_files_only=True,
            ),
            device,
        )
    data = config["data"]
    generator_eval_path = data.get(f"{generator_type}_eval_path")
    if generator_eval_path is not None:
        data["eval_path"] = generator_eval_path
    generator_token_dtype = data.get(f"{generator_type}_token_dtype")
    if generator_token_dtype is not None:
        data["token_dtype"] = str(generator_token_dtype)
    generator_offset = data.get(f"{generator_type}_token_offset")
    if generator_offset is not None:
        data["token_offset"] = int(generator_offset)
    data["codec_layer"] = int(data.get(f"{generator_type}_codec_layer", 0))
    generator_codebook_size = data.get(f"{generator_type}_codebook_size")
    if generator_codebook_size is not None:
        data["codebook_size"] = int(generator_codebook_size)
    inferred_size = infer_codec_vocabulary_size(tokenizer, int(data["codec_layer"]))
    if generator_type == "gpt":
        if generator_codebook_size is None:
            if inferred_size is None:
                raise ValueError(
                    "Could not infer the GPT codec vocabulary size; set data.gpt_codebook_size."
                )
            data["codebook_size"] = inferred_size
        elif inferred_size is not None and int(generator_codebook_size) != inferred_size:
            raise ValueError(
                f"data.gpt_codebook_size={generator_codebook_size} does not match the "
                f"RSQ codec packed vocabulary size {inferred_size}. Use null to infer it."
            )
    elif generator_type == "dlm":
        if generator_codebook_size is None:
            # Backward compatibility for old configs that only define codebook_size.
            if "codebook_size" not in data:
                if inferred_size is None:
                    raise ValueError("Set data.dlm_codebook_size for the selected DLM codec.")
                data["codebook_size"] = inferred_size
        elif inferred_size is not None and int(generator_codebook_size) != inferred_size:
            raise ValueError(
                f"data.dlm_codebook_size={generator_codebook_size} does not match the "
                f"DLM codec vocabulary size {inferred_size}."
            )
    output_dir = Path(config["output"]["directory"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = parse_indices(config["data"].get("sample_indices", [0]))
    samples = select_samples(config, indices)
    results = [
        process_sample(
            sample,
            index,
            generator_model,
            generator_type,
            bert_model,
            tokenizer,
            config,
            device,
        )
        for index, sample in zip(indices, samples)
    ]
    summary_path = output_dir / f"summary_{generator_type}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
