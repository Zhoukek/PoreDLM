import math
import statistics
from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import sacrebleu
import transformers
from flax.jax_utils import replicate
from tqdm import tqdm

from utils.logging_utils import log_for_0


# ============================================
# Text-similarity metrics (BLEU / ROUGE)
# ============================================
def _mean_std_sem(values):
    n = len(values)
    mean = sum(values) / n
    std = statistics.pstdev(values) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, sem


def compute_bleu(hypotheses, references):
    return sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True, use_effective_order=True).score


def compute_rouge(hypotheses, references, return_std=False):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        r1.append(s["rouge1"].fmeasure * 100)
        r2.append(s["rouge2"].fmeasure * 100)
        rL.append(s["rougeL"].fmeasure * 100)
    m1, s1, e1 = _mean_std_sem(r1)
    m2, s2, e2 = _mean_std_sem(r2)
    mL, sL, eL = _mean_std_sem(rL)
    means = {"rouge1": m1, "rouge2": m2, "rougeL": mL}
    if not return_std:
        return means
    stds = {
        "rouge1_std": s1, "rouge2_std": s2, "rougeL_std": sL,
        "rouge1_sem": e1, "rouge2_sem": e2, "rougeL_sem": eL,
    }
    return means, stds


# ============================================
# JAX perplexity / entropy metrics
# ============================================
class NLL:
    """JAX implementation of NLL metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean_value = jnp.array(0.0, dtype=jnp.float32)
        self.weight = jnp.array(0.0, dtype=jnp.float32)

    def update(self, value: Union[float, jnp.ndarray], weight: Union[float, jnp.ndarray] = 1.0):
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value, dtype=jnp.float32)
        if weight is not None and not isinstance(weight, jnp.ndarray):
            weight = jnp.array(weight, dtype=jnp.float32)
        weight = jnp.broadcast_to(weight, value.shape)
        if value.size == 0:
            return
        self.mean_value = self.mean_value + jnp.sum(value)
        self.weight = self.weight + jnp.sum(weight)


class Perplexity(NLL):
    def compute(self) -> jnp.ndarray:
        return jnp.exp(self.mean_value / self.weight)


class MeanMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_value = jnp.array(0.0, dtype=jnp.float32)
        self.count = jnp.array(0.0, dtype=jnp.float32)

    def update(self, value: Union[float, jnp.ndarray]):
        if not isinstance(value, jnp.ndarray):
            value = jnp.array(value, dtype=jnp.float32)
        self.sum_value = self.sum_value + jnp.sum(value)
        self.count = self.count + value.size

    def compute(self) -> jnp.ndarray:
        return self.sum_value / self.count


class Metrics:
    def __init__(
        self,
        gen_ppl_eval_model_name_or_path=None,
        eval_ppl_batch_size=None,
        eval_context_size=1024,
    ) -> None:
        self.gen_ppl = Perplexity()
        self.sample_entropy = MeanMetric()
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.gen_ppl_eval_model_name_or_path = gen_ppl_eval_model_name_or_path
        self.eval_context_size = eval_context_size
        self._ppl_params = None
        self._ppl_compute_batch_nlls = None

        # mT5 needs use_fast=False to avoid Tiktoken/SentencePiece conversion issues.
        use_fast = "mt5" not in gen_ppl_eval_model_name_or_path.lower()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            gen_ppl_eval_model_name_or_path, use_fast=use_fast,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def reset(self):
        self.gen_ppl.reset()
        self.sample_entropy.reset()

    def _eval_retokenize(self, text_samples, max_length):
        """Retokenize samples for the eval model. Returns (samples, attn_mask, eval_context_size)."""
        out = self.tokenizer(
            text_samples,
            return_tensors="np",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        return out["input_ids"], out["attention_mask"], self.eval_context_size

    def record_generative_perplexity(
        self,
        text_samples: List[str],
        max_length: int,
        retokenize: bool = True,
    ) -> Dict:
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        n_devices = jax.local_device_count()

        # Load model and compile pmap once; reuse on subsequent calls
        if self._ppl_params is None:
            from transformers import FlaxAutoModelForCausalLM
            log_for_0(f"Loading JAX/Flax model: {self.gen_ppl_eval_model_name_or_path}")
            eval_model = FlaxAutoModelForCausalLM.from_pretrained(self.gen_ppl_eval_model_name_or_path)
            log_for_0(f"Replicating model parameters across {n_devices} devices...")
            params = replicate(eval_model.params)

            @jax.pmap
            def compute_batch_nlls(params, input_ids, attention_mask, eos_token_id):
                logits = eval_model(input_ids, attention_mask=attention_mask, params=params).logits
                targets = input_ids[:, 1:]
                logits_pred = logits[:, :-1, :]
                batch_indices = jnp.arange(targets.shape[0])[:, None]
                seq_indices = jnp.arange(targets.shape[1])[None, :]
                target_logits = logits_pred[batch_indices, seq_indices, targets]
                log_normalizers = jax.nn.logsumexp(logits_pred, axis=-1)
                nlls = log_normalizers - target_logits
                is_eos = input_ids == eos_token_id
                first_eos = jnp.cumsum(is_eos, axis=-1) == 1
                token_mask = input_ids != eos_token_id
                valid_tokens = first_eos[:, 1:] + token_mask[:, 1:]
                return nlls, valid_tokens

            self._ppl_params = params
            self._ppl_compute_batch_nlls = compute_batch_nlls
            log_for_0("PPL model cached for reuse")

        params = self._ppl_params
        compute_batch_nlls = self._ppl_compute_batch_nlls

        if retokenize:
            samples, attn_mask, eval_context_size = self._eval_retokenize(text_samples, max_length=max_length)
        else:
            samples = text_samples
            attn_mask = np.ones(samples.shape)
            eval_context_size = samples.shape[-1]

        # Round batch size down to a multiple of n_devices (>=1).
        batch_size = self.eval_ppl_batch_size or samples.shape[0]
        batch_size = min(batch_size, samples.shape[0])
        batch_size = (batch_size // n_devices) * n_devices or n_devices

        num_batches = (samples.shape[0] + batch_size - 1) // batch_size
        log_for_0(f"PPL: batch_size={batch_size} ({batch_size // n_devices}/device), {num_batches} batches")

        per_sample_nll_sum = np.zeros(samples.shape[0], dtype=np.float64)
        per_sample_token_count = np.zeros(samples.shape[0], dtype=np.float64)

        for i in tqdm(range(num_batches), desc="Evaluating perplexity"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, samples.shape[0])
            actual_batch_size = batch_end - batch_start

            batch_samples = samples[batch_start:batch_end]
            batch_attn_mask = attn_mask[batch_start:batch_end]

            # Pad the last batch to full batch_size for pmap
            if actual_batch_size < batch_size:
                pad_size = batch_size - actual_batch_size
                batch_samples = np.concatenate([
                    batch_samples,
                    np.zeros((pad_size, batch_samples.shape[1]), dtype=batch_samples.dtype),
                ], axis=0)
                batch_attn_mask = np.concatenate([
                    batch_attn_mask,
                    np.zeros((pad_size, batch_attn_mask.shape[1]), dtype=batch_attn_mask.dtype),
                ], axis=0)

            for chunk_start in range(0, batch_samples.shape[1], eval_context_size):
                chunk_end = min(chunk_start + eval_context_size, batch_samples.shape[1])
                sample_chunk = batch_samples[:, chunk_start:chunk_end]
                attn_mask_chunk = batch_attn_mask[:, chunk_start:chunk_end]

                # [n_devices, batch_per_device, seq_len]
                sample_chunk_sharded = sample_chunk.reshape(n_devices, batch_size // n_devices, sample_chunk.shape[1])
                attn_mask_chunk_sharded = attn_mask_chunk.reshape(n_devices, batch_size // n_devices, attn_mask_chunk.shape[1])
                eos_token_id_replicated = jnp.array([self.tokenizer.eos_token_id] * n_devices)

                nlls_sharded, valid_tokens_sharded = compute_batch_nlls(
                    params, sample_chunk_sharded, attn_mask_chunk_sharded, eos_token_id_replicated,
                )

                nlls = nlls_sharded.reshape(batch_size, nlls_sharded.shape[2])
                valid_tokens = valid_tokens_sharded.reshape(batch_size, valid_tokens_sharded.shape[2])
                if actual_batch_size < batch_size:
                    nlls = nlls[:actual_batch_size]
                    valid_tokens = valid_tokens[:actual_batch_size]

                # Device-to-host transfer for accumulation
                nlls_np = np.asarray(nlls)
                valid_tokens_np = np.asarray(valid_tokens)
                weighted_nlls = nlls_np * valid_tokens_np

                self.gen_ppl.update(jnp.array(weighted_nlls), jnp.array(valid_tokens_np))

                per_sample_nll_sum[batch_start:batch_end] += weighted_nlls.sum(axis=-1)
                per_sample_token_count[batch_start:batch_end] += valid_tokens_np.sum(axis=-1)

                del nlls_sharded, valid_tokens_sharded, nlls, valid_tokens
                del nlls_np, valid_tokens_np, weighted_nlls

        # Per-sample perplexity (NaN for zero-token samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_sample_ppl = np.exp(per_sample_nll_sum / per_sample_token_count)
        per_sample_ppl = np.where(per_sample_token_count > 0, per_sample_ppl, np.nan).tolist()

        # Per-sample entropy (only on valid tokens, excluding padding)
        per_sample_entropy = []
        for i in range(samples.shape[0]):
            valid_len = int(attn_mask[i].sum())
            valid_tokens = samples[i, :valid_len]
            _, counts = np.unique(valid_tokens, return_counts=True)
            probs = counts.astype(np.float32) / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            per_sample_entropy.append(entropy)
            self.sample_entropy.update(entropy)

        return {
            "ppl": float(self.gen_ppl.compute()),
            "per_sample_ppl": per_sample_ppl,
            "mean_entropy": sum(per_sample_entropy) / len(per_sample_entropy),
        }
