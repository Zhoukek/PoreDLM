import time
import os
import json
import itertools
import wandb
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Dict
from jax.random import PRNGKey

from utils.logging_utils import log_for_0
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.train_utils import TrainState
from configs.config import Config, SamplingConfig

from utils.metrics_utils import Metrics as PPLMetrics, compute_bleu, compute_rouge
from jax.experimental.multihost_utils import sync_global_devices
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    mask_after_eos, shift_left,
    _setup_generation, _make_pmap_pair, _shard_timesteps, _shard_noise,
    _build_run_name,
)


# ============================================
# Generation Helper
# ============================================
def run_generation(
    state: Any,
    encoder_params: Dict,
    encoder_apply_fn,
    eval_dataset,
    tokenizer,
    config,
    rng: PRNGKey,
    local_batch_size: int,
) -> PRNGKey:
    """Run test generation."""
    for sc_idx, sc in enumerate(config.sampling_configs):
        if len(config.sampling_configs) > 1:
            log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
        rng, sample_rng = jax.random.split(rng)
        common_kwargs = dict(
            state=state,
            tokenizer=tokenizer,
            rng=sample_rng,
            config=config,
            sampling_config=sc,
            batch_size=local_batch_size,
            num_samples=config.num_samples,
        )
        if eval_dataset is None:
            test_generation_uncond(**common_kwargs)
        else:
            test_generation_cond(
                **common_kwargs,
                encoder_params=encoder_params,
                encoder_apply_fn=encoder_apply_fn,
                dataset=eval_dataset,
            )

    return rng


# ============================================
# Evaluation / Testing Utilities
# ============================================
def test_generation_uncond(
    state: TrainState,
    tokenizer,
    rng: PRNGKey,
    config: Config,
    sampling_config: SamplingConfig,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Test unconditional generation (multi-device pmap)."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    (state_unreplicated, model_apply_fn, model_params_replicated,
     d_model, num_local_devices, effective_batch_size) = _setup_generation(
        state, config, batch_size, "              UNCONDITIONAL GENERATION EXAMPLES",
    )

    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    cfg_list = [1]
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    wandb_tables = {}
    ppl_metrics = None
    if config.online_eval:
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Method: {sampling_method}, Steps: {num_sampling_steps}, "
                  f"CFG Scale: {cfg_scale}, SC-CFG: {self_cond_cfg_scale} ---")

        p_generate, p_decode_ids = _make_pmap_pair(
            model_apply_fn, config, sampling_config, cfg_scale, self_cond_cfg_scale,
        )

        all_generated = []
        generation_time = 0.0
        decode_time = 0.0
        num_batches = (num_samples + effective_batch_size - 1) // effective_batch_size
        samples_processed = 0

        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            if samples_processed >= num_samples:
                break

            current_total_batch = min(effective_batch_size, num_samples - samples_processed)
            current_total_batch = (
                (current_total_batch + num_local_devices - 1) // num_local_devices
            ) * num_local_devices
            current_per_device = current_total_batch // num_local_devices

            batch_rng = jax.random.fold_in(rng, batch_idx * jax.process_count() + jax.process_index())
            noise_rng, t_rng = jax.random.split(batch_rng)
            device_rngs = jax.random.split(noise_rng, num_local_devices)

            t_steps_sharded = _shard_timesteps(
                t_rng, num_local_devices, num_sampling_steps, time_schedule, config,
            )
            z_sharded = _shard_noise(
                device_rngs, num_local_devices, current_per_device,
                config.max_length, d_model, config.denoiser_noise_scale,
            )

            gen_start = time.time()
            latent_sharded = p_generate(
                model_params=model_params_replicated, rng=device_rngs,
                z=z_sharded, t_steps=t_steps_sharded,
                cond_seq=None, cond_seq_mask=None,
            )
            latent_sharded.block_until_ready()
            generation_time += time.time() - gen_start

            dec_start = time.time()
            t_final_sharded = t_steps_sharded[:, -1]
            predicted_ids_sharded = p_decode_ids(
                z=latent_sharded, model_params=model_params_replicated, t_final_val=t_final_sharded,
            )
            predicted_ids_sharded.block_until_ready()
            decode_time += time.time() - dec_start

            predicted_ids = predicted_ids_sharded.reshape(-1, predicted_ids_sharded.shape[-1])
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

            for i in range(predicted_ids.shape[0]):
                if samples_processed >= num_samples:
                    break
                text = tokenizer.decode(np.array(predicted_ids[i]), skip_special_tokens=True)
                all_generated.append((samples_processed, text))
                samples_processed += 1

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state_unreplicated.epoch)
        step_val = int(state_unreplicated.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="uncond",
        )

        # Dummy op so all processes stay in sync before rank-0-only file write / PPL
        jax.random.normal(jax.random.PRNGKey(0))

        # Rank 0 writes one file to the same folder (no shards).
        out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
        if jax.process_index() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, gen in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

        sync_global_devices("after_save")
        ppl_results = None
        if config.online_eval:
            log_for_0("\n" + "=" * 70)
            log_for_0("              PPL EVALUATION (all hosts, logged on rank 0)")
            log_for_0("=" * 70)
            ppl_metrics.reset()

            # All hosts read the exact same samples so JAX launches stay aligned.
            with open(out_path, "r", encoding="utf-8") as f:
                text_samples = [json.loads(line)["generated"] for line in f]

            nonempty_samples = [s for s in text_samples if isinstance(s, str) and s.strip()]
            skipped = len(text_samples) - len(nonempty_samples)
            if skipped > 0:
                log_for_0(f"PPL eval: skipped {skipped} empty samples")
            if not nonempty_samples:
                log_for_0("PPL eval: all samples empty; skipping perplexity computation")
            else:
                ppl_results = ppl_metrics.record_generative_perplexity(
                    text_samples=nonempty_samples,
                    max_length=config.eval_ppl_max_length,
                    retokenize=True,
                )
                mean_ppl = ppl_results["ppl"]
                log_for_0(f"Perplexity: {mean_ppl:.4f}")
                log_for_0(f"Mean Entropy: {ppl_results['mean_entropy']:.4f}")
            log_for_0("=" * 70 + "\n")

        sync_global_devices("after_ppl")

        if jax.process_index() == 0:
            if ppl_results is not None:
                mean_ppl = ppl_results["ppl"]
                metrics_line = {
                    "epoch": epoch_val, "step": step_val,
                    "ppl": mean_ppl, "mean_entropy": ppl_results["mean_entropy"],
                }
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation metrics")

            if config.use_wandb:
                table = wandb.Table(columns=["sample_id", "text"])
                for tid, gen in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, gen)
                wandb_tables[f"generated_samples_uncond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if ppl_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/ppl": mean_ppl,
                        f"generation/{name}/mean_entropy": ppl_results["mean_entropy"],
                    })

    if jax.process_index() == 0 and config.use_wandb and wandb_tables:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")

    log_for_0("=" * 70 + "\n")


def test_generation_cond(
    state: TrainState,
    encoder_params: Dict,
    encoder_apply_fn,
    tokenizer,
    rng: PRNGKey,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Test conditional generation (multi-device pmap)."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    (state_unreplicated, model_apply_fn, model_params_replicated,
     d_model, num_local_devices, effective_batch_size) = _setup_generation(
        state, config, batch_size, "              CONDITIONAL GENERATION EXAMPLES",
    )

    encode_latent_mean, encode_latent_std = config.latent_mean, config.latent_std

    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    dataloader = get_dataloader(
        dataset, batch_size=effective_batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    p_encode = jax.pmap(
        partial(
            encode_text, encoder_apply_fn=encoder_apply_fn,
            latent_mean=encode_latent_mean, latent_std=encode_latent_std,
        ),
        axis_name="batch",
    )

    wandb_tables = {}
    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Steps: {num_sampling_steps}, CFG Scale: {cfg_scale}, "
                  f"SC-CFG: {self_cond_cfg_scale} ---")

        p_generate, p_decode_ids = _make_pmap_pair(
            model_apply_fn, config, sampling_config, cfg_scale, self_cond_cfg_scale,
        )

        all_generated = []  # (id, original_text, generated_text, context_text)
        generation_time = 0.0
        decode_time = 0.0
        samples_processed = 0

        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= num_samples:
                break

            batch_size_current = batch["input_ids"].shape[0]

            if batch_size_current % num_local_devices != 0:
                pad_size = num_local_devices - (batch_size_current % num_local_devices)
                for key in batch:
                    if isinstance(batch[key], np.ndarray):
                        pad_arr = np.zeros((pad_size,) + batch[key].shape[1:], dtype=batch[key].dtype)
                        batch[key] = np.concatenate([batch[key], pad_arr], axis=0)
                    elif isinstance(batch[key], list):
                        batch[key] = batch[key] + [""] * pad_size

            actual_batch_size = batch["input_ids"].shape[0]
            current_per_device = actual_batch_size // num_local_devices

            batch_rng = jax.random.fold_in(rng, batch_idx * jax.process_count() + jax.process_index())
            noise_rng, t_rng = jax.random.split(batch_rng)
            device_rngs = jax.random.split(noise_rng, num_local_devices)

            t_steps_sharded = _shard_timesteps(
                t_rng, num_local_devices, num_sampling_steps, time_schedule, config,
            )

            # Encode full input sequence using encoder_attention_mask so cond tokens
            # only attend to cond tokens
            max_length = config.max_length
            input_ids = jnp.array(batch["input_ids"])
            encoder_attention_mask = jnp.array(batch["encoder_attention_mask"])
            cond_seq_mask_arr = jnp.array(batch["cond_seq_mask"])

            input_ids_sharded = input_ids.reshape(num_local_devices, current_per_device, -1)
            encoder_attention_mask_sharded = encoder_attention_mask.reshape(
                num_local_devices, current_per_device, max_length, max_length,
            )

            cond_seq_sharded = p_encode(
                input_ids=input_ids_sharded,
                attention_mask=encoder_attention_mask_sharded,
                encoder_params=encoder_params,
            )
            cond_seq_mask_sharded = cond_seq_mask_arr.reshape(num_local_devices, current_per_device, -1)

            z_sharded = _shard_noise(
                device_rngs, num_local_devices, current_per_device,
                max_length, d_model, config.denoiser_noise_scale,
            )

            gen_start = time.time()
            latent_sharded = p_generate(
                model_params=model_params_replicated, rng=device_rngs,
                z=z_sharded, t_steps=t_steps_sharded,
                cond_seq=cond_seq_sharded, cond_seq_mask=cond_seq_mask_sharded,
            )
            latent_sharded.block_until_ready()
            generation_time += time.time() - gen_start

            gen_length = config.max_length - config.max_input_length
            cond_len_per_sample = cond_seq_mask_arr.astype(jnp.int32).sum(axis=1)

            dec_start = time.time()
            t_final_sharded = t_steps_sharded[:, -1]
            predicted_ids_sharded = p_decode_ids(
                z=latent_sharded, model_params=model_params_replicated, t_final_val=t_final_sharded,
            )
            predicted_ids_sharded.block_until_ready()
            predicted_ids = predicted_ids_sharded.reshape(-1, predicted_ids_sharded.shape[-1])
            # Strip cond prefix; prefix tokens may contain EOS so mask only after stripping.
            predicted_ids = shift_left(predicted_ids, cond_len_per_sample, 0)[:, :gen_length]
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            decode_time += time.time() - dec_start

            original_texts = [batch["target"][i] for i in range(actual_batch_size)]
            context_texts = [batch["input"][i] for i in range(actual_batch_size)]

            for i in range(min(batch_size_current, actual_batch_size)):
                if samples_processed >= num_samples:
                    break
                text = tokenizer.decode(np.array(predicted_ids[i]), skip_special_tokens=True)
                all_generated.append((samples_processed, original_texts[i], text, context_texts[i]))
                samples_processed += 1

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state_unreplicated.epoch)
        step_val = int(state_unreplicated.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="cond",
        )

        if jax.process_index() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, orig, gen, ctx in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

            cond_eval_results = None
            if config.online_eval and all_generated:
                hypotheses = [gen for _, _, gen, _ in all_generated]
                references = [orig for _, orig, _, _ in all_generated]
                bleu_score = compute_bleu(hypotheses, references)
                rouge_scores = compute_rouge(hypotheses, references)
                cond_eval_results = {"bleu": bleu_score, **rouge_scores}
                log_for_0(
                    f"BLEU: {bleu_score:.2f}  ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                    f"ROUGE-2: {rouge_scores['rouge2']:.2f}  ROUGE-L: {rouge_scores['rougeL']:.2f}"
                )

            if config.use_wandb:
                table = wandb.Table(columns=["sample_id", "context", "original", "generated"])
                for tid, orig, gen, ctx in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, ctx, orig, gen)
                wandb_tables[f"generated_samples_cond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if cond_eval_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/bleu": cond_eval_results["bleu"],
                        f"generation/{name}/rouge1": cond_eval_results["rouge1"],
                        f"generation/{name}/rouge2": cond_eval_results["rouge2"],
                        f"generation/{name}/rougeL": cond_eval_results["rougeL"],
                    })
            if cond_eval_results is not None:
                metrics_line = {"epoch": epoch_val, "step": step_val, **cond_eval_results}
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation metrics")

    if jax.process_index() == 0 and config.use_wandb and wandb_tables:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")

    log_for_0("=" * 70 + "\n")
