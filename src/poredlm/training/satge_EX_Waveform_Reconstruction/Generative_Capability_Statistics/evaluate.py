"""Measure conditional DLM generation quality versus predicted span length."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
TOKEN_TO_WAVEFORM_DIR = HERE.parent / "token_to_waveform"
if str(TOKEN_TO_WAVEFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TOKEN_TO_WAVEFORM_DIR))

from generate_and_plot import (  # noqa: E402
    TokenSequenceDataset,
    freeze,
    infer_codec_downsample_rate,
    infer_codec_vocabulary_size,
    normalize_raw_tokens,
    register_custom_tokenizer,
    resolve_device,
    stage1_decode,
    token_span_to_waveform_span,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return config


def parse_int_list(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        result = [int(item) for item in value]
    else:
        result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise ValueError("Expected a non-empty list of non-negative integers.")
    return result


def choose_indices(data: dict[str, Any]) -> list[int]:
    explicit = data.get("sample_indices")
    if explicit is not None:
        return parse_int_list(explicit)
    count = int(data.get("num_chunks", 20))
    start = int(data.get("start_index", 0))
    if count < 1 or start < 0:
        raise ValueError("Require data.num_chunks >= 1 and data.start_index >= 0.")
    return list(range(start, start + count))


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
        raise IndexError(f"Sample indices not found: {missing}")
    return [selected[index] for index in indices]


def mask_start_for_position(total_length: int, mask_length: int, center_fraction: float) -> int:
    if mask_length < 1 or mask_length > total_length:
        raise ValueError(
            f"mask_length must be in [1, {total_length}], got {mask_length}."
        )
    if not 0.0 <= center_fraction <= 1.0:
        raise ValueError(f"Position fraction must be in [0, 1], got {center_fraction}.")
    center = center_fraction * total_length
    return min(max(0, int(round(center - mask_length / 2.0))), total_length - mask_length)


def prepare_batch(
    samples: list[dict[str, object]],
    indices: list[int],
    total_length: int,
    bos: int,
    eos: int,
    token_offset: int,
    codebook_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    rows: list[torch.Tensor] = []
    sample_ids: list[str] = []
    for sample_index, sample in zip(indices, samples):
        raw = normalize_raw_tokens(torch.as_tensor(sample["tokens"]), bos, eos)
        content = raw[1:-1]
        if content.numel() < total_length:
            raise ValueError(
                f"Sample {sample_index} contains {content.numel()} content tokens; "
                f"evaluation.total_length={total_length}."
            )
        content = content[:total_length].long()
        codec_ids = content - token_offset
        if int(codec_ids.min()) < 0 or int(codec_ids.max()) >= codebook_size:
            raise ValueError(f"Sample {sample_index} contains token IDs outside the codec vocabulary.")
        rows.append(content)
        sample_ids.append(str(sample.get("id", "")))
    return torch.stack(rows).to(device), sample_ids


@torch.inference_mode()
def evaluate_case(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    reference_content: torch.Tensor,
    mask_start: int,
    mask_length: int,
    config: dict[str, Any],
    seed: int,
    token_offset: int,
    codebook_size: int,
    codec_layer: int,
    downsample_rate: int,
    bos: int,
    eos: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, total_length = reference_content.shape
    full_ids = torch.cat(
        [
            torch.full((batch_size, 1), bos, dtype=torch.long, device=reference_content.device),
            reference_content,
            torch.full((batch_size, 1), eos, dtype=torch.long, device=reference_content.device),
        ],
        dim=1,
    )
    mask_end = mask_start + mask_length
    masked_positions = torch.zeros_like(full_ids, dtype=torch.bool)
    masked_positions[:, 1 + mask_start : 1 + mask_end] = True
    condition_token_mask = ~masked_positions
    masked_input_ids = full_ids.clone()
    configured_mask = config.get("mask_token_id")
    mask_token_id = int(
        configured_mask
        if configured_mask is not None
        else getattr(model.config, "mask_token_id", 1)
    )
    masked_input_ids[masked_positions] = mask_token_id

    result = model.generate(
        condition_input_ids=masked_input_ids,
        condition_attention_mask=torch.ones_like(masked_input_ids),
        condition_token_mask=condition_token_mask,
        max_length=total_length + 2,
        num_steps=int(config.get("num_steps", 64)),
        sampling_method=str(config.get("sampling_method", "ode")),
        sde_gamma=float(config.get("sde_gamma", 0.1)),
        cfg_scale=float(config.get("cfg_scale", 1.0)),
        self_cond_cfg_scale=float(config.get("self_cond_cfg_scale", 1.0)),
        seed=seed,
        return_dict=True,
    )
    if not isinstance(result, dict) or "sequences" not in result:
        raise TypeError("DLM generate() must return a dict containing sequences.")
    generated = result["sequences"]
    if bool(config.get("restrict_to_codebook", True)):
        logits = result.get("logits")
        if logits is None or logits.shape[-1] < token_offset + codebook_size:
            raise ValueError("DLM logits do not cover the configured codec vocabulary.")
        constrained = logits[..., token_offset : token_offset + codebook_size].argmax(dim=-1)
        constrained = constrained + token_offset
        generated = generated.clone()
        generated[masked_positions] = constrained[masked_positions]

    generated_content = generated[:, 1 : 1 + total_length]
    generated_region = generated_content[:, mask_start:mask_end]
    reference_region = reference_content[:, mask_start:mask_end]
    token_accuracy = (generated_region == reference_region).float().mean(dim=1).cpu().numpy()

    generated_codec = generated_content - token_offset
    if int(generated_codec.min()) < 0 or int(generated_codec.max()) >= codebook_size:
        raise ValueError("DLM generated token IDs outside the configured codec vocabulary.")
    reference_codec = reference_content - token_offset
    generated_waveform = stage1_decode(codec, generated_codec, codec_layer).float()
    reference_waveform = stage1_decode(codec, reference_codec, codec_layer).float()
    waveform_length = min(generated_waveform.shape[-1], reference_waveform.shape[-1])
    waveform_start, waveform_end = token_span_to_waveform_span(
        mask_start, mask_end, downsample_rate, waveform_length
    )
    difference = (
        generated_waveform[:, waveform_start:waveform_end]
        - reference_waveform[:, waveform_start:waveform_end]
    )
    mse = difference.float().square().mean(dim=1).cpu().numpy()
    return mse, token_accuracy


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["position"]), int(row["mask_length"]))].append(float(row["mse"]))
        groups[("overall", int(row["mask_length"]))].append(float(row["mse"]))
    output: list[dict[str, Any]] = []
    for (position, mask_length), values_list in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        values = np.asarray(values_list, dtype=np.float64)
        count = int(values.size)
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        output.append(
            {
                "position": position,
                "mask_length": mask_length,
                "count": count,
                "mean_mse": float(values.mean()),
                "median_mse": float(np.median(values)),
                "std_mse": std,
                "q25_mse": float(np.quantile(values, 0.25)),
                "q75_mse": float(np.quantile(values, 0.75)),
                "ci95_low": max(0.0, float(values.mean() - 1.96 * std / math.sqrt(count))),
                "ci95_high": float(values.mean() + 1.96 * std / math.sqrt(count)),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_statistics(
    details: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    positions: list[str],
    output_dir: Path,
) -> None:
    color_values = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, max(1, len(positions))))
    colors = {name: color for name, color in zip(positions, color_values)}
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for position in positions:
        selected = sorted(
            (row for row in aggregate if row["position"] == position),
            key=lambda row: int(row["mask_length"]),
        )
        x = np.asarray([row["mask_length"] for row in selected])
        y = np.asarray([row["mean_mse"] for row in selected])
        low = np.asarray([row["ci95_low"] for row in selected])
        high = np.asarray([row["ci95_high"] for row in selected])
        axis.plot(x, y, marker="o", linewidth=1.8, label=position, color=colors[position])
        axis.fill_between(x, low, high, alpha=0.16, color=colors[position])
    overall = sorted(
        (row for row in aggregate if row["position"] == "overall"),
        key=lambda row: int(row["mask_length"]),
    )
    axis.plot(
        [row["mask_length"] for row in overall],
        [row["mean_mse"] for row in overall],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="overall",
    )
    axis.set_xlabel("Predicted span length (tokens)")
    axis.set_ylabel("Waveform MSE")
    axis.set_title("Conditional DLM generative capability by predicted span length")
    axis.grid(alpha=0.25)
    axis.legend(title="Chunk position")
    figure.savefig(output_dir / "mse_vs_mask_length.png", dpi=200)
    plt.close(figure)

    lengths = sorted({int(row["mask_length"]) for row in details})
    matrix = np.full((len(positions), len(lengths)), np.nan, dtype=np.float64)
    for row_index, position in enumerate(positions):
        for col_index, length in enumerate(lengths):
            values = [float(row["mse"]) for row in details if row["position"] == position and int(row["mask_length"]) == length]
            if values:
                matrix[row_index, col_index] = float(np.mean(values))
    figure, axis = plt.subplots(figsize=(max(9, len(lengths) * 0.75), 4.5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axis.set_xticks(np.arange(len(lengths)), labels=lengths)
    axis.set_yticks(np.arange(len(positions)), labels=positions)
    axis.set_xlabel("Predicted span length (tokens)")
    axis.set_ylabel("Chunk position")
    axis.set_title("Mean waveform MSE heatmap")
    figure.colorbar(image, ax=axis, label="Mean MSE")
    figure.savefig(output_dir / "mean_mse_heatmap.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(len(positions), 1, figsize=(max(10, len(lengths) * 0.8), 4 * len(positions)), squeeze=False, constrained_layout=True)
    for axis, position in zip(axes[:, 0], positions):
        distributions = [
            [float(row["mse"]) for row in details if row["position"] == position and int(row["mask_length"]) == length]
            for length in lengths
        ]
        axis.boxplot(distributions, labels=[str(length) for length in lengths], showfliers=True)
        axis.set_title(f"MSE distribution: {position}")
        axis.set_xlabel("Predicted span length (tokens)")
        axis.set_ylabel("Waveform MSE")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "mse_distributions.png", dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--sample-indices")
    parser.add_argument("--mask-lengths")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.num_chunks is not None:
        config["data"]["num_chunks"] = args.num_chunks
        config["data"]["sample_indices"] = None
    if args.sample_indices is not None:
        config["data"]["sample_indices"] = parse_int_list(args.sample_indices)
    if args.mask_lengths is not None:
        config["evaluation"]["mask_lengths"] = parse_int_list(args.mask_lengths)
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)

    data = config["data"]
    evaluation = config["evaluation"]
    models = config["models"]
    output_dir = Path(config["output"]["directory"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.get("device", "auto")))

    model_path = Path(models["dlm_path"]).expanduser().resolve()
    tokenizer_path = Path(models["tokenizer_path"]).expanduser().resolve()
    register_custom_tokenizer(tokenizer_path)
    model = freeze(
        AutoModel.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True),
        device,
    )
    if not hasattr(model, "generate"):
        raise AttributeError(f"DLM at {model_path} does not provide conditional generate().")
    codec = freeze(
        AutoModel.from_pretrained(str(tokenizer_path), trust_remote_code=True, local_files_only=True),
        device,
    )

    token_offset = int(data.get("token_offset", 128))
    codebook_size = int(data.get("codebook_size", 65536))
    codec_layer = int(data.get("codec_layer", 0))
    inferred_size = infer_codec_vocabulary_size(codec, codec_layer)
    if inferred_size is not None and inferred_size != codebook_size:
        raise ValueError(
            f"Configured codebook_size={codebook_size}, but codec reports {inferred_size}."
        )
    downsample_rate = infer_codec_downsample_rate(codec)
    total_length = int(evaluation["total_length"])
    mask_lengths = sorted(set(parse_int_list(evaluation["mask_lengths"])))
    positions = {str(key): float(value) for key, value in evaluation["positions"].items()}
    batch_size = int(evaluation.get("batch_size", 2))
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be >= 1.")
    if total_length + 2 > int(model.config.dlm_config.get("max_length", total_length + 2)):
        raise ValueError("evaluation.total_length plus BOS/EOS exceeds the DLM maximum length.")

    indices = choose_indices(data)
    samples = select_samples(config, indices)
    reference_content, sample_ids = prepare_batch(
        samples,
        indices,
        total_length,
        int(data.get("bos_token_id", 2)),
        int(data.get("eos_token_id", 3)),
        token_offset,
        codebook_size,
        device,
    )

    details: list[dict[str, Any]] = []
    total_cases = len(mask_lengths) * len(positions)
    case_number = 0
    base_seed = int(evaluation.get("seed", 6198))
    for mask_length in mask_lengths:
        for position, center_fraction in positions.items():
            case_number += 1
            mask_start = mask_start_for_position(total_length, mask_length, center_fraction)
            print(
                f"[{case_number}/{total_cases}] position={position} "
                f"tokens=[{mask_start}, {mask_start + mask_length}) seed={base_seed}",
                flush=True,
            )
            for batch_start in range(0, len(indices), batch_size):
                batch_end = min(batch_start + batch_size, len(indices))
                # Reuse the same sample-level random streams across every mask
                # length and position, so the comparison is not confounded by
                # drawing an unrelated Gaussian initialization for each case.
                batch_seed = base_seed + batch_start
                mse, token_accuracy = evaluate_case(
                    model,
                    codec,
                    reference_content[batch_start:batch_end],
                    mask_start,
                    mask_length,
                    evaluation,
                    batch_seed,
                    token_offset,
                    codebook_size,
                    codec_layer,
                    downsample_rate,
                    int(data.get("bos_token_id", 2)),
                    int(data.get("eos_token_id", 3)),
                )
                for local_index, global_index in enumerate(range(batch_start, batch_end)):
                    details.append(
                        {
                            "sample_index": indices[global_index],
                            "sample_id": sample_ids[global_index],
                            "position": position,
                            "center_fraction": center_fraction,
                            "mask_start": mask_start,
                            "mask_end": mask_start + mask_length,
                            "mask_length": mask_length,
                            "waveform_start": mask_start * downsample_rate,
                            "waveform_end": min((mask_start + mask_length) * downsample_rate, total_length * downsample_rate),
                            "mse": float(mse[local_index]),
                            "token_accuracy": float(token_accuracy[local_index]),
                            "seed": batch_seed,
                        }
                    )

    aggregate = aggregate_rows(details)
    write_csv(output_dir / "per_sample_metrics.csv", details)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": str(model_path),
                "tokenizer_path": str(tokenizer_path),
                "sample_indices": indices,
                "total_length": total_length,
                "mask_lengths": mask_lengths,
                "positions": positions,
                "downsample_rate": downsample_rate,
                "num_steps": int(evaluation.get("num_steps", 64)),
                "batch_size": batch_size,
                "sampling_method": str(evaluation.get("sampling_method", "ode")),
                "aggregate": aggregate,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    plot_statistics(details, aggregate, list(positions), output_dir)
    print(f"Saved statistics to {output_dir}")


if __name__ == "__main__":
    main()
