"""Inference for the continuous CNN -> BERT -> basecalling route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from modeling_continuous_basecall import ContinuousBasecallModel
from poredlm.training_public.stage4_basecall.Basecalling.basecaller_v8_0420.ctc_crf import decode as ctc_crf_decode
from poredlm.training_public.stage4_basecall.Basecalling.basecaller_v8_0420.metrics import batch_bonito_accuracy, ctc_viterbi_decode


ID2BASE = {0: "N", 1: "A", 2: "C", 3: "G", 4: "T"}


def find_chunk_files(paths: str | list[str]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(path.glob("**/*_chunks.npy"))
        elif path.is_file() and path.name.endswith("_chunks.npy"):
            files.append(path)
        elif path.is_file():
            raise ValueError(f"Input file must end with _chunks.npy: {path}")
        else:
            raise FileNotFoundError(f"Input path not found: {item}")
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No *_chunks.npy files found.")
    return files


class SignalInferenceDataset(Dataset):
    def __init__(self, paths: str | list[str]):
        self.chunks_files = find_chunk_files(paths)
        self.entries: list[tuple[int, int]] = []
        self._chunks: dict[int, np.ndarray] = {}
        self._references: dict[int, np.ndarray | None] = {}
        for file_index, chunks_path in enumerate(self.chunks_files):
            chunks = np.load(chunks_path, allow_pickle=True)
            if chunks.ndim != 2:
                raise ValueError(f"Expected 2D chunks array, got {chunks_path}={chunks.shape}")
            reference_path = chunks_path.with_name(
                chunks_path.name[: -len("_chunks.npy")] + "_references.npy"
            )
            references = None
            if reference_path.exists():
                references = np.load(reference_path, allow_pickle=True)
                if references.ndim != 2 or references.shape[0] != chunks.shape[0]:
                    raise ValueError(
                        f"Invalid reference shape: {reference_path}={references.shape}; chunks={chunks.shape}"
                    )
            self._references[file_index] = references
            self.entries.extend((file_index, row) for row in range(chunks.shape[0]))
        print(f"[EvalDataset] files={len(self.chunks_files)} reads={len(self.entries)}", flush=True)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        file_index, row = self.entries[index]
        if file_index not in self._chunks:
            self._chunks[file_index] = np.load(self.chunks_files[file_index], allow_pickle=True)
        signal = torch.from_numpy(np.asarray(self._chunks[file_index][row], dtype=np.float32).copy())
        references = self._references[file_index]
        labels = None
        if references is not None:
            reference = np.asarray(references[row]).reshape(-1)
            labels = torch.from_numpy(reference[reference > 0].astype(np.int64, copy=False))
        return {"signal": signal, "labels": labels, "file_index": file_index, "row": row}


def collate_inference(batch: list[dict[str, object]]) -> dict[str, object]:
    signals = [item["signal"] for item in batch]
    lengths = torch.tensor([int(signal.numel()) for signal in signals], dtype=torch.long)
    padded = torch.zeros((len(signals), int(lengths.max().item())), dtype=torch.float32)
    for index, signal in enumerate(signals):
        padded[index, : signal.numel()] = signal
    labels = [item["labels"] for item in batch]
    has_references = all(label is not None for label in labels)
    return {
        "signal": padded,
        "signal_lengths": lengths,
        "target_seqs": [label.tolist() for label in labels] if has_references else None,
        "ids": [(int(item["file_index"]), int(item["row"])) for item in batch],
    }


def decode_predictions(logits: torch.Tensor, lengths: torch.Tensor, head_type: str) -> list[list[int]]:
    logits_tbc = logits.detach().float().transpose(0, 1)
    if head_type == "ctc_crf":
        return [
            ctc_crf_decode(logits_tbc[: int(length), index:index + 1, :])[0]
            for index, length in enumerate(lengths.cpu().tolist())
        ]
    return ctc_viterbi_decode(logits_tbc, input_lengths=lengths, blank_idx=0)


def ids_to_sequence(ids: list[int]) -> str:
    return "".join(ID2BASE.get(int(item), "N") for item in ids if int(item) != 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Directory or *_chunks.npy file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    backbone_cfg = config["backbone"]
    head_type = model_cfg.get("head_type", "ctc")
    if head_type == "ctc_crf":
        os.environ["CTC_CRF_STATE_LEN"] = str(model_cfg.get("ctc_crf_state_len", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SignalInferenceDataset(args.input)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate_inference,
    )
    model = ContinuousBasecallModel(
        cnn_checkpoint=backbone_cfg["cnn_checkpoint"],
        bert_checkpoint=backbone_cfg["bert_checkpoint"],
        num_classes=int(model_cfg.get("num_classes", 5)),
        freeze_cnn=True,
        freeze_bert=True,
        head_type=head_type,
        ctc_crf_state_len=int(model_cfg.get("ctc_crf_state_len", 5)),
        ctc_crf_blank_score=float(model_cfg.get("ctc_crf_blank_score", 2.0)),
        pre_head_type=model_cfg.get("pre_head_type", "none"),
        pre_head_transformer_nhead=int(model_cfg.get("pre_head_transformer_nhead", 8)),
        head_output_activation=model_cfg.get("head_output_activation"),
        head_output_scale=model_cfg.get("head_output_scale"),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[list[int]] = []
    all_references: list[list[int]] = []
    jsonl_path = output_dir / "predictions.jsonl"
    fastq_path = output_dir / "predictions.fastq"
    with jsonl_path.open("w", encoding="utf-8") as jsonl, fastq_path.open("w", encoding="utf-8") as fastq:
        with torch.no_grad():
            for batch in tqdm(loader, desc="basecalling"):
                signal = batch["signal"].to(device, non_blocking=True)
                lengths = batch["signal_lengths"].to(device)
                logits, input_lengths = model(signal, lengths)
                predictions = decode_predictions(logits, input_lengths, head_type)
                references = batch["target_seqs"]
                for index, prediction in enumerate(predictions):
                    file_index, row = batch["ids"][index]
                    read_id = f"{dataset.chunks_files[file_index].stem}:{row}"
                    sequence = ids_to_sequence(prediction)
                    record = {"id": read_id, "sequence": sequence, "predicted_ids": prediction}
                    if references is not None:
                        record["reference_ids"] = references[index]
                        all_references.append(references[index])
                    all_predictions.append(prediction)
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fastq.write(f"@{read_id}\n{sequence}\n+\n{'I' * len(sequence)}\n")

    metrics = {"num_reads": len(all_predictions)}
    if len(all_references) == len(all_predictions) and all_references:
        metrics["bonito_accuracy"] = batch_bonito_accuracy(all_predictions, all_references)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[Eval] wrote {jsonl_path}", flush=True)
    print(f"[Eval] wrote {fastq_path}", flush=True)
    print(f"[Eval] wrote {output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
