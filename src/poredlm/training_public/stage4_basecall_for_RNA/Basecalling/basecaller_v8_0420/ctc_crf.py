# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Tuple

import torch

from .utils import ID2BASE


def _load_koi():
    try:
        import koi  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Bonito CTC-CRF requires the ont-koi package. Install it and ensure it "
            "matches your PyTorch/CUDA version."
        ) from exc
    return koi


def _alphabet() -> List[str]:
    max_id = max(ID2BASE.keys())
    return [ID2BASE[i] for i in range(max_id + 1)]

def crf_num_classes(state_len: int, alphabet: List[str] | None = None) -> int:
    alphabet = alphabet or _alphabet()
    n_base = len(alphabet[1:])
    return len(alphabet) * (n_base ** state_len)


def crf_num_classes_no_blank(state_len: int, alphabet: List[str] | None = None) -> int:
    alphabet = alphabet or _alphabet()
    n_base = len(alphabet[1:])
    return (n_base ** state_len) * n_base


def _prepare_targets(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    max_len = int(target_lengths.max().item()) if batch_size > 0 else 0
    padded = torch.zeros((batch_size, max_len), dtype=targets.dtype, device=device)
    offset = 0
    for i in range(batch_size):
        length = int(target_lengths[i].item())
        if length > 0:
            padded[i, :length] = targets[offset: offset + length]
        offset += length
    return padded


class CTC_CRF:
    def __init__(self, state_len: int, alphabet: List[str]):
        koi = _load_koi()
        from koi.ctc import SequenceDist, Max, Log, semiring
        from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse

        class _Dist(SequenceDist):
            def __init__(self, state_len: int, alphabet: List[str]):
                super().__init__()
                self.alphabet = alphabet
                self.state_len = state_len
                self.n_base = len(alphabet[1:])
                self.idx = torch.cat(
                    [
                        torch.arange(self.n_base ** (self.state_len))[:, None],
                        torch.arange(self.n_base ** (self.state_len))
                        .repeat_interleave(self.n_base)
                        .reshape(self.n_base, -1)
                        .T,
                    ],
                    dim=1,
                ).to(torch.int32)

            def n_score(self):
                return len(self.alphabet) * self.n_base ** (self.state_len)

            def logZ(self, scores, S: semiring = Log):
                T, N, _ = scores.shape
                Ms = scores.reshape(T, N, -1, len(self.alphabet))
                alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
                beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
                return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

            def normalise(self, scores):
                return (scores - self.logZ(scores)[:, None] / len(scores))

            def forward_scores(self, scores, S: semiring = Log):
                T, N, _ = scores.shape
                Ms = scores.reshape(T, N, -1, self.n_base + 1)
                alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
                return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

            def backward_scores(self, scores, S: semiring = Log):
                T, N, _ = scores.shape
                Ms = scores.reshape(T, N, -1, self.n_base + 1)
                beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
                return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

            def compute_transition_probs(self, scores, betas):
                T, N, _ = scores.shape
                log_trans_probs = scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None]
                log_trans_probs = torch.cat(
                    [
                        log_trans_probs[:, :, :, [0]],
                        log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base),
                    ],
                    dim=-1,
                )
                trans_probs = torch.softmax(log_trans_probs, dim=-1)
                init_state_probs = torch.softmax(betas[0], dim=-1)
                return trans_probs, init_state_probs

            def viterbi(self, scores):
                traceback = self.posteriors(scores, Max)
                a_traceback = traceback.argmax(2)
                moves = (a_traceback % len(self.alphabet)) != 0
                paths = 1 + (torch.div(a_traceback, len(self.alphabet), rounding_mode="floor") % self.n_base)
                return torch.where(moves, paths, 0)

            def prepare_ctc_scores(self, scores, targets):
                targets = torch.clamp(targets - 1, 0)
                T, N, _ = scores.shape
                scores = scores.to(torch.float32)
                n = targets.size(1) - (self.state_len - 1)
                stay_indices = sum(
                    targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
                    for i in range(self.state_len)
                ) * len(self.alphabet)
                move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
                stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
                move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
                return stay_scores, move_scores

            def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction="mean", normalise_scores=True):
                scores = scores.to(torch.float32)
                if normalise_scores:
                    scores = self.normalise(scores)
                stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
                logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
                if normalise_scores:
                    logz = torch.minimum(logz, logz.new_zeros(logz.shape))
                loss = -(logz / target_lengths)
                if loss_clip:
                    loss = torch.clamp(loss, 0.0, loss_clip)
                if reduction == "mean":
                    return loss.mean()
                if reduction in ("none", None):
                    return loss
                raise ValueError(f"Unknown reduction type {reduction}")

            def ctc_viterbi_alignments(self, scores, targets, target_lengths):
                stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
                return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)

        self._dist = _Dist(state_len=state_len, alphabet=alphabet)

    @property
    def n_score(self) -> int:
        return self._dist.n_score()

    def ctc_loss(self, scores, targets, target_lengths) -> torch.Tensor:
        return self._dist.ctc_loss(scores, targets, target_lengths)

    def viterbi(self, scores) -> torch.Tensor:
        return self._dist.viterbi(scores)


_MODEL_CACHE: Tuple[int, List[str], CTC_CRF] | None = None


def _get_model() -> CTC_CRF:
    global _MODEL_CACHE
    state_len = int(os.environ.get("CTC_CRF_STATE_LEN", "5"))
    alphabet = _alphabet()
    if _MODEL_CACHE is None or _MODEL_CACHE[0] != state_len or _MODEL_CACHE[1] != alphabet:
        _MODEL_CACHE = (state_len, alphabet, CTC_CRF(state_len=state_len, alphabet=alphabet))
    return _MODEL_CACHE[2]


def ctc_crf_loss(
    logits_tbc: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = 0,
) -> torch.Tensor:
    model = _get_model()
    if blank_idx != 0:
        raise ValueError("Bonito CTC-CRF expects blank_idx=0.")
    if logits_tbc.size(-1) != model.n_score:
        raise ValueError(
            f"Bonito CTC-CRF expects scores with dim={model.n_score}, got {logits_tbc.size(-1)}. "
            "Update the model head to output CRF scores."
        )
    device = logits_tbc.device
    batch_size = logits_tbc.size(1)
    targets = targets.to(device=device)
    target_lengths = target_lengths.to(device=device)
    input_lengths = input_lengths.to(device=device)
    padded_targets = _prepare_targets(targets, target_lengths, batch_size, device)
    losses = []
    for i in range(batch_size):
        time_len = int(input_lengths[i].item())
        if time_len <= 0:
            continue
        target_len = int(target_lengths[i].item())
        if target_len <= 0:
            continue
        sample_targets = padded_targets[i : i + 1, :target_len]
        sample_target_lengths = target_lengths[i : i + 1]
        sample_scores = logits_tbc[:time_len, i : i + 1, :]
        losses.append(model.ctc_loss(sample_scores, sample_targets, sample_target_lengths))
    if not losses:
        return logits_tbc.new_tensor(0.0)
    return torch.stack(losses).mean()


def _collapse_paths(paths: torch.Tensor) -> List[List[int]]:
    if paths.dim() != 2:
        raise ValueError("Expected viterbi paths with shape [T, B].")
    paths = paths.transpose(0, 1).cpu().numpy()
    collapsed: List[List[int]] = []
    for seq in paths:
        out: List[int] = []
        prev = None
        for token in seq:
            token = int(token)
            if token == 0:
                prev = token
                continue
            if token != prev:
                out.append(token)
            prev = token
        collapsed.append(out)
    return collapsed


def decode(logits_tbc: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    model = _get_model()
    if blank_idx != 0:
        raise ValueError("Bonito CTC-CRF expects blank_idx=0.")
    if logits_tbc.size(-1) != model.n_score:
        raise ValueError(
            f"Bonito CTC-CRF expects scores with dim={model.n_score}, got {logits_tbc.size(-1)}. "
            "Update the model head to output CRF scores."
        )
    paths = model.viterbi(logits_tbc)
    return _collapse_paths(paths)
