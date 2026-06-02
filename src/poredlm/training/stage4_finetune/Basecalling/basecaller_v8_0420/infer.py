# infer_jsonl_fastq.py
# 用法示例：
# python infer_jsonl_fastq.py \
#   --ckpt your.ckpt \
#   --model_name_or_path your_hf_model_dir_or_name \
#   --jsonl_gz reads.jsonl.gz \
#   --out out.fastq \
#   --amp

import argparse
import math
import json
import os
import gzip
import re
from typing import List, Tuple, Iterable

import torch
from tqdm import tqdm

from .ctc_crf import decode as ctc_crf_decode
from .model import BasecallModel
from .utils import (
    ID2BASE,
    BLANK_IDX,
    seed_everything,
    resolve_input_lengths,
    infer_head_config_from_state_dict,
    infer_pre_head_type_from_state_dict,
)
from .metrics import ctc_viterbi_decode, koi_beam_search_decode


def _print_model_structure(model: torch.nn.Module, *, prefix: str = "[Model]") -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{prefix} class={model.__class__.__name__}")
    print(f"{prefix} pre_head={model.pre_head.__class__.__name__} head={model.base_head.__class__.__name__}")
    print(f"{prefix} params total={total_params:,} trainable={trainable_params:,}")
    print(f"{prefix} structure:")
    print(model)


def _phred_to_char(q: int) -> str:
    # standard Sanger FASTQ (Phred+33)
    q = min(max(q, 0), 93)
    return chr(q + 33)


def _constant_qstring(length: int, q: int) -> str:
    return _phred_to_char(q) * max(length, 0)


def write_fastq(fp, read_id: str, seq: str, q: str):
    fp.write(f"@{read_id}\n")
    fp.write(seq + "\n")
    fp.write("+\n")
    fp.write(q + "\n")


def iter_jsonl_reads(path: str) -> Iterable[Tuple[str, str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            read_id = str(obj.get("read_id", "") or obj.get("id", ""))
            text = obj.get("text", "")
            if not read_id or not text:
                continue
            yield read_id, text


def split_bwav_tokens(text: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(text):
        start = text.find("<|bwav:", i)
        if start < 0:
            break
        end = text.find("|>", start)
        if end < 0:
            break
        tokens.append(text[start : end + 2])
        i = end + 2
    return tokens


_BWAV_TOKEN_RE = re.compile(r"<\|bwav:(\d+)\|>")


def apply_token_offset_to_signal_str(signal_str: str, token_offset: int) -> str:
    if token_offset <= 0 or not signal_str:
        return signal_str
    return _BWAV_TOKEN_RE.sub(lambda m: f"<|bwav:{int(m.group(1)) + token_offset}|>", signal_str)


def chunk_tokens(tokens: List[str], max_tokens: int, overlap: int) -> List[List[str]]:
    if max_tokens <= 0:
        return [tokens]
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens.")
    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + max_tokens]
        if chunk:
            chunks.append(chunk)
    return chunks


def _token_slice_to_base_idx(token_idx: int, token_len: int, base_len: int) -> int:
    if token_len <= 0 or base_len <= 0:
        return 0
    return int(round(token_idx * base_len / token_len))


def _infer_crf_state_len(num_classes: int, n_base: int) -> int:
    if n_base <= 1:
        raise ValueError("Cannot infer CTC-CRF state_len with n_base <= 1.")
    candidates = []
    if num_classes % (n_base + 1) == 0:
        base = num_classes / (n_base + 1)
        state_len = math.log(base, n_base)
        if math.isclose(state_len, round(state_len)):
            candidates.append(int(round(state_len)))
    base = math.log(num_classes, n_base) - 1
    if math.isclose(base, round(base)):
        candidates.append(int(round(base)))
    if candidates:
        return candidates[0]
    raise ValueError(
        "Unable to infer CTC-CRF state_len from num_classes and n_base. "
        "Please pass --ctc_crf_state_len or set CTC_CRF_STATE_LEN."
    )


def _ctc_crf_decode_batch(
    logits_tbc: torch.Tensor,
    input_lengths: torch.Tensor,
) -> List[List[int]]:
    logits_tbc = logits_tbc.float()
    decoded: List[List[int]] = []
    for idx, step_len in enumerate(input_lengths.tolist()):
        if step_len <= 0:
            decoded.append([])
            continue
        sample_logits = logits_tbc[:step_len, idx : idx + 1, :]
        decoded_ids = ctc_crf_decode(sample_logits, blank_idx=BLANK_IDX)[0]
        decoded_len = min(len(decoded_ids), step_len)
        decoded.append(decoded_ids[:decoded_len])
    return decoded


def stitch_sequences(
    chunk_seqs: List[str],
    chunk_qs: List[str],
    chunk_token_lengths: List[int],
    total_tokens: int,
    chunksize: int,
    overlap: int,
) -> Tuple[str, str]:
    if not chunk_seqs:
        return "", ""
    if len(chunk_seqs) == 1:
        return chunk_seqs[0], chunk_qs[0]
    if overlap <= 0 or chunksize <= 0:
        return "".join(chunk_seqs), "".join(chunk_qs)

    semi_overlap = overlap // 2
    start_tok = semi_overlap
    end_tok = chunksize - semi_overlap
    stub = (total_tokens - overlap) % (chunksize - overlap)
    first_chunk_end_tok = (stub + semi_overlap) if stub > 0 else end_tok

    stitched_seq: List[str] = []
    stitched_q: List[str] = []

    for idx, (seq, q, token_len) in enumerate(zip(chunk_seqs, chunk_qs, chunk_token_lengths)):
        if idx == 0:
            end_idx = _token_slice_to_base_idx(first_chunk_end_tok, token_len, len(seq))
            stitched_seq.append(seq[:end_idx])
            stitched_q.append(q[:end_idx])
        elif idx == len(chunk_seqs) - 1:
            start_idx = _token_slice_to_base_idx(start_tok, token_len, len(seq))
            stitched_seq.append(seq[start_idx:])
            stitched_q.append(q[start_idx:])
        else:
            start_idx = _token_slice_to_base_idx(start_tok, token_len, len(seq))
            end_idx = _token_slice_to_base_idx(end_tok, token_len, len(seq))
            stitched_seq.append(seq[start_idx:end_idx])
            stitched_q.append(q[start_idx:end_idx])

    return "".join(stitched_seq), "".join(stitched_q)


# --------------------------
# main
# --------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--model_name_or_path", required=True, type=str)
    ap.add_argument("--jsonl_gz", type=str, required=True)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--beam_width", type=int, default=32)
    ap.add_argument("--ctc_crf_blank_score", type=float, default=2.0,
                    help="Blank score used by CTC-CRF head logits (keep consistent with training).")
    ap.add_argument("--decoder", choices=["auto", "ctc_viterbi", "koi", "ctc_crf"], default="auto",
                    help="Decoder to use. auto picks ctc_viterbi for CTC head and ctc_crf for CTC-CRF head.")
    ap.add_argument("--head_type", choices=["ctc", "ctc_crf"], default=None,
                    help="Override head type (default: infer from checkpoint).")
    ap.add_argument("--ctc_crf_state_len", type=int, default=None,
                    help="Override CTC-CRF state_len (default: infer from head or CTC_CRF_STATE_LEN env).")
    ap.add_argument("--beam_q", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--hidden_layer", type=int, default=-1,
                    help="Which backbone hidden layer to use when --feature_source hidden.")
    ap.add_argument("--learnable_fuse_last_n_layers", type=int, default=0,
                    help="If >0, learn a softmax-weighted fusion over the last N hidden layers (overrides --hidden_layer).")
    ap.add_argument("--feature_source", "--feature-source", choices=["hidden", "embedding"], default="hidden",
                    help="Use transformer hidden states or input embeddings as head input features.")
    ap.add_argument("--pre_head_type", choices=["auto", "none", "bilstm", "transformer", "tcn"], default="auto",
                    help="Optional module before CTC-CRF head. Default auto-infers from checkpoint.")
    ap.add_argument("--pre_head_transformer_nhead", type=int, default=8,
                    help="Attention heads for --pre_head_type transformer.")
    ap.add_argument("--token_offset", type=int, default=0,
                    help="Add this offset to each <|bwav:ID|> token in input signal_str (e.g. 0->128).")
    args = ap.parse_args()
    if args.token_offset < 0:
        raise ValueError("--token_offset must be >= 0")

    seed_everything(42)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    # 兼容 ckpt 格式：{"model": ...} / {"model_state_dict": ...} / {"state_dict": ...} / 直接 state_dict
    if isinstance(state, dict):
        if "model" in state:
            sd = state["model"]
        elif "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
    else:
        sd = state
    head_config = infer_head_config_from_state_dict(sd)
    inferred_pre_head_type = infer_pre_head_type_from_state_dict(sd)
    head_type = args.head_type or head_config.get("head_type", "ctc")
    pre_head_type = args.pre_head_type
    if pre_head_type == "auto":
        pre_head_type = inferred_pre_head_type
        print(f"[Model] pre_head_type auto -> {pre_head_type}")
    elif pre_head_type != inferred_pre_head_type:
        print(
            "[Model][Warning] --pre_head_type overrides checkpoint inference: "
            f"arg={pre_head_type}, inferred={inferred_pre_head_type}"
        )
    # load model
    n_base = len(ID2BASE) - 1
    state_len = args.ctc_crf_state_len
    decoder_mode = args.decoder
    if decoder_mode == "auto":
        decoder_mode = "ctc_viterbi" if head_type == "ctc" else "ctc_crf"

    if head_type == "ctc" and decoder_mode not in {"ctc_viterbi", "koi"}:
        raise ValueError("CTC head supports --decoder ctc_viterbi, koi, or auto.")
    if head_type == "ctc_crf" and decoder_mode not in {"ctc_crf", "koi"}:
        raise ValueError("CTC-CRF head supports --decoder ctc_crf, koi, or auto.")
    if decoder_mode == "ctc_crf":
        if head_type != "ctc_crf":
            raise ValueError("--decoder ctc_crf requires checkpoint/model head_type=ctc_crf.")
        if state_len is None:
            env_state_len = os.environ.get("CTC_CRF_STATE_LEN")
            if env_state_len is not None:
                state_len = int(env_state_len)
        if state_len is None:
            state_len = _infer_crf_state_len(head_config["num_classes"], n_base)
        os.environ["CTC_CRF_STATE_LEN"] = str(state_len)
    model = BasecallModel(
        model_path=args.model_name_or_path,
        num_classes=head_config["num_classes"],
        hidden_layer=args.hidden_layer,
        learnable_fuse_last_n_layers=args.learnable_fuse_last_n_layers,
        feature_source=args.feature_source,
        pre_head_type=pre_head_type,
        pre_head_transformer_nhead=args.pre_head_transformer_nhead,
        head_type=head_type,
        head_crf_blank_score=float(args.ctc_crf_blank_score),
        head_crf_n_base=n_base,
        head_crf_state_len=state_len,
        head_crf_expand_blanks=True,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    _print_model_structure(model)

    tokenizer = model.tokenizer  # 你的 BasecallModel 里应有

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as w:
        for read_id, signal_str in tqdm(iter_jsonl_reads(args.jsonl_gz), desc="jsonl->fastq"):
            signal_str = apply_token_offset_to_signal_str(signal_str, args.token_offset)
            tokens = split_bwav_tokens(signal_str)
            if not tokens:
                continue
            chunks = chunk_tokens(tokens, args.max_tokens, args.overlap)
            chunk_seqs: List[str] = []
            chunk_qs: List[str] = []

            chunk_token_lengths = [len(chunk) for chunk in chunks]
            for start in range(0, len(chunks), args.batch_size):
                batch_chunks = chunks[start:start + args.batch_size]
                batch_strs = ["".join(chunk) for chunk in batch_chunks]

                enc = tokenizer(batch_strs, return_tensors="pt", padding=True, truncation=False)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                input_lengths = resolve_input_lengths(
                    input_ids,
                    attention_mask=attention_mask,
                )

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits_btc = model(input_ids, attention_mask=attention_mask)  # [B,T,C]

                logits_tbc = logits_btc.transpose(0, 1)
                if decoder_mode == "ctc_crf":
                    pred_ids = _ctc_crf_decode_batch(logits_tbc, input_lengths)
                elif decoder_mode == "ctc_viterbi":
                    pred_ids = ctc_viterbi_decode(logits_tbc, input_lengths=input_lengths, blank_idx=BLANK_IDX)
                else:
                    pred_ids = koi_beam_search_decode(
                        logits_tbc,
                        beam_width=args.beam_width,
                        beam_cut=100.0,
                        scale=1.0,
                        offset=0.0,
                        blank_score=float(args.ctc_crf_blank_score),
                        reverse=False,
                        input_lengths=input_lengths,
                    )
                for ids in pred_ids:
                    seq = "".join(ID2BASE.get(i, "N") for i in ids)
                    qstring = _constant_qstring(len(seq), args.beam_q)
                    chunk_seqs.append(seq)
                    chunk_qs.append(qstring)

            if args.overlap > 0 and chunk_seqs:
                full_seq, full_q = stitch_sequences(
                    chunk_seqs,
                    chunk_qs,
                    chunk_token_lengths,
                    total_tokens=len(tokens),
                    chunksize=args.max_tokens,
                    overlap=args.overlap,
                )
            else:
                full_seq = "".join(chunk_seqs)
                full_q = "".join(chunk_qs)
            write_fastq(w, read_id, full_seq, full_q)

    print(f"[OK] wrote FASTQ: {args.out}")


if __name__ == "__main__":
    main()
