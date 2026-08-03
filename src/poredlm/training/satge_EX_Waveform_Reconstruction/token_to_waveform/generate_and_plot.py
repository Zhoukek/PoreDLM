"""Conditionally generate pore tokens and compare two waveform decoders."""

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
TRAINING_DECODER_DIR = HERE.parent / "training_waveform_decoder"
if str(TRAINING_DECODER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DECODER_DIR))

from modeling_waveform_decoder import WaveformDecoder  # noqa: E402
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


def load_waveform_decoder(checkpoint_path: Path, device: torch.device) -> tuple[WaveformDecoder, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config: dict[str, Any] = {}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        checkpoint_config = payload.get("config") or {}
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported waveform decoder checkpoint: {type(payload)!r}")
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    hidden_size = int(
        checkpoint_config.get("model", {}).get("hidden_size", 768)
    )
    decoder = WaveformDecoder(hidden_size=hidden_size)
    decoder.load_state_dict(state_dict, strict=True)
    return freeze(decoder, device), checkpoint_config


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


def dlm_hidden(
    dlm: torch.nn.Module,
    input_ids: torch.Tensor,
    hidden_key: str,
    config: dict[str, Any],
) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    kwargs: dict[str, Any] = {}
    sampling = config.get("hidden_sampling", {})
    if hidden_key == "context_hidden_state":
        kwargs["return_context"] = True
    elif hidden_key == "ode_hidden_state":
        kwargs.update(
            return_ode_hidden=True,
            ode_steps=int(sampling.get("ode_steps", 4)),
            ode_start_t=float(sampling.get("ode_start_t", 0.85)),
            ode_self_cond_cfg_scale=float(sampling.get("self_cond_cfg_scale", 1.0)),
        )
    elif hidden_key == "sde_hidden_state":
        kwargs.update(
            return_sde_hidden=True,
            sde_steps=int(sampling.get("sde_steps", 4)),
            sde_start_t=float(sampling.get("sde_start_t", 0.85)),
            sde_gamma=float(sampling.get("sde_gamma", 0.1)),
            sde_self_cond_cfg_scale=float(sampling.get("self_cond_cfg_scale", 1.0)),
            sde_seed=sampling.get("seed"),
        )
    elif hidden_key != "last_hidden_state":
        raise ValueError(f"Unsupported hidden_state_key={hidden_key!r}.")
    outputs = dlm(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    if hidden_key not in outputs:
        raise KeyError(f"DLM output does not contain {hidden_key!r}; keys={list(outputs)}")
    return outputs[hidden_key].float()


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


def plot_waveforms(
    stage1_waveform: np.ndarray,
    learned_waveform: np.ndarray,
    reference_waveform: np.ndarray,
    condition_tokens: int,
    total_tokens: int,
    title: str,
    output_path: Path,
) -> None:
    common = min(stage1_waveform.size, learned_waveform.size, reference_waveform.size)
    stage1_waveform = stage1_waveform[:common]
    learned_waveform = learned_waveform[:common]
    reference_waveform = reference_waveform[:common]
    boundary = int(round(common * condition_tokens / total_tokens))
    zoom_start = max(0, boundary - min(300, boundary))

    figure, axes = plt.subplots(2, 1, figsize=(18, 9), constrained_layout=True)
    x = np.arange(common)
    for axis, start, subtitle in (
        (axes[0], 0, "Full reconstructed waveform"),
        (axes[1], zoom_start, "Condition / generated boundary"),
    ):
        axis.plot(x[start:], reference_waveform[start:], color="black", alpha=0.45, linewidth=0.8, label="Reference tokens → Stage-1 decoder")
        axis.plot(x[start:], stage1_waveform[start:], color="#0072B2", linewidth=0.9, label="Generated tokens → codebook → Stage-1 decoder")
        axis.plot(x[start:], learned_waveform[start:], color="#D55E00", linewidth=0.9, label="Generated tokens → DLM hidden → waveform decoder")
        axis.axvline(boundary, color="#009E73", linestyle="--", linewidth=1.4, label=f"generation starts ({condition_tokens} tokens)")
        axis.set_title(subtitle)
        axis.set_xlabel("Waveform sample")
        axis.set_ylabel("Normalized current")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].set_xlim(zoom_start, common)
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
    waveform_decoder: WaveformDecoder,
    config: dict[str, Any],
    device: torch.device,
    hidden_key: str,
) -> dict[str, Any]:
    generation = config["generation"]
    data = config["data"]
    condition_length = int(generation["condition_length"])
    generation_length = int(generation["generation_length"])
    total_content_length = condition_length + generation_length
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
    condition_ids = torch.cat(
        [torch.tensor([bos], device=device), reference_content[:condition_length]]
    ).unsqueeze(0)
    generated = dlm.generate(
        condition_input_ids=condition_ids,
        condition_attention_mask=torch.ones_like(condition_ids),
        max_length=1 + total_content_length,
        num_steps=int(generation.get("num_steps", 50)),
        sampling_method=str(generation.get("sampling_method", "ode")),
        sde_gamma=float(generation.get("sde_gamma", 0.1)),
        cfg_scale=float(generation.get("cfg_scale", 1.0)),
        self_cond_cfg_scale=float(generation.get("self_cond_cfg_scale", 1.0)),
        seed=int(generation.get("seed", 6198)) + sample_index,
    )
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

    stage1_waveform = stage1_decode(tokenizer, generated_codec).float()
    reference_waveform = stage1_decode(tokenizer, reference_codec).float()
    generated_with_boundaries = torch.cat(
        [
            torch.tensor([[bos]], device=device),
            generated_content.unsqueeze(0),
            torch.tensor([[eos]], device=device),
        ],
        dim=1,
    )
    hidden = dlm_hidden(dlm, generated_with_boundaries, hidden_key, config)
    learned_waveform = waveform_decoder(hidden[:, 1 : 1 + total_content_length]).float()

    stage1_np = stage1_waveform[0].detach().cpu().numpy()
    learned_np = learned_waveform[0, 0].detach().cpu().numpy()
    reference_np = reference_waveform[0].detach().cpu().numpy()
    output_dir = Path(config["output"]["directory"]).expanduser()
    stem = f"sample_{sample_index:06d}_cond{condition_length}_gen{generation_length}"
    plot_waveforms(
        stage1_np,
        learned_np,
        reference_np,
        condition_length,
        total_content_length,
        f"{stem} | id={sample.get('id', '')}",
        output_dir / f"{stem}.png",
    )
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        generated_dlm_ids=generated_content.detach().cpu().numpy(),
        generated_codec_ids=generated_codec[0].detach().cpu().numpy(),
        reference_dlm_ids=reference_content.detach().cpu().numpy(),
        stage1_waveform=stage1_np,
        waveform_decoder_waveform=learned_np,
        reference_waveform=reference_np,
    )
    generated_region = slice(condition_length, total_content_length)
    token_accuracy = float(
        (generated_content[generated_region] == reference_content[generated_region])
        .float()
        .mean()
        .item()
    )
    return {
        "sample_index": sample_index,
        "sample_id": str(sample.get("id", "")),
        "condition_length": condition_length,
        "generation_length": generation_length,
        "invalid_generated_tokens": invalid_count,
        "generated_region_token_accuracy": token_accuracy,
        "waveform_decoder_vs_stage1": waveform_metrics(learned_np, stage1_np),
        "stage1_generated_vs_reference": waveform_metrics(stage1_np, reference_np),
        "figure": str(output_dir / f"{stem}.png"),
        "arrays": str(output_dir / f"{stem}.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--condition-length", type=int)
    parser.add_argument("--generation-length", type=int)
    parser.add_argument("--sample-indices")
    parser.add_argument("--waveform-decoder-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.condition_length is not None:
        config["generation"]["condition_length"] = args.condition_length
    if args.generation_length is not None:
        config["generation"]["generation_length"] = args.generation_length
    if args.sample_indices is not None:
        config["data"]["sample_indices"] = args.sample_indices
    if args.waveform_decoder_checkpoint is not None:
        config["models"]["waveform_decoder_checkpoint"] = str(args.waveform_decoder_checkpoint)
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)
    if int(config["generation"]["condition_length"]) < 1 or int(config["generation"]["generation_length"]) < 1:
        raise ValueError("condition_length and generation_length must both be positive.")

    device = resolve_device(str(config.get("device", "auto")))
    models = config["models"]
    dlm_path = Path(models["dlm_path"]).expanduser().resolve()
    tokenizer_path = Path(models["tokenizer_path"]).expanduser().resolve()
    checkpoint_path = Path(models["waveform_decoder_checkpoint"]).expanduser().resolve()
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
    waveform_decoder, checkpoint_config = load_waveform_decoder(checkpoint_path, device)
    hidden_key = str(
        models.get("hidden_state_key")
        or checkpoint_config.get("model", {}).get("hidden_state_key")
        or "last_hidden_state"
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
            waveform_decoder,
            config,
            device,
            hidden_key,
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
