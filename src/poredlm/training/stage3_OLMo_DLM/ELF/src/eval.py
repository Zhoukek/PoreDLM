#!/usr/bin/env python
"""Evaluation script for trained ELF models: loads a checkpoint and generates text samples."""

import argparse
import contextlib
import copy
import logging
import os
import sys

# Initialize JAX distributed BEFORE importing other JAX modules
import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass  # Single-host run, or already initialized.

# Ensure repo root on sys.path so imports work when run as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import optax
from flax import jax_utils
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.model import ELF_models
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState
from utils.data_utils import load_jsonl_dataset, load_dataset_split, get_pad_token_id
from generation import test_generation_uncond, test_generation_cond
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained ELF model by generating text samples")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used when --seeds is not specified)")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated list of seeds to evaluate (e.g. '42,123,456'). Overrides --seed.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to checkpoint file (e.g. outputs/elf_b-owt/checkpoint_19000) or HF repo id.",
    )
    parser.add_argument(
        "--use_cpu", action="store_true",
        help="Host model init, train state template, and encoder/state replication on CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_for_0("Loading configuration...")
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")

    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()
    num_hosts = jax.process_count()
    cpu_device = jax.local_devices(backend="cpu")[0] if args.use_cpu else None

    def cpu_ctx():
        return jax.default_device(cpu_device) if args.use_cpu else contextlib.nullcontext()

    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size for evaluation: {config.global_batch_size}")
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // num_hosts
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * num_devices
        local_batch_size = config.batch_size * num_local_devices
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")

    log_for_0(f"Config loaded from {args.config}")
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Max length: {config.max_length}")
    log_for_0(f"Max input length: {config.max_input_length}")
    log_for_0(f"Num samples: {config.num_samples}")
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")

    seed_list = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds is not None else [args.seed]
    log_for_0(f"Seeds to evaluate: {seed_list}")

    rng = jax.random.PRNGKey(config.seed)

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    eval_dataset = None
    if config.eval_data_path is not None:
        log_for_0("Loading dataset for conditional generation...")
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(
                config.eval_data_path, tokenizer,
                input_key="input",
                output_key="output",
            )
        else:
            eval_dataset = load_dataset_split(config.eval_data_path)
        log_for_0(f"Eval dataset size: {len(eval_dataset)}")

    # ============================================
    # Load Encoder (frozen)
    # ============================================
    log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
    encoder_config, encoder_model, _ = get_encoder(config.encoder_model_name, jnp.float32)
    encoder_params = load_encoder_checkpoint(config.encoder_checkpoint)
    log_for_0("encoder weights loaded.")

    # Multi-device eval passes encoder params directly into pmap, so replicate them
    # across local accelerator devices.
    encoder_params = jax_utils.replicate(encoder_params)
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    # ============================================
    # Create ELF Model
    # ============================================
    log_for_0(f"Creating {config.model} model...")
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    max_length = config.max_length

    with cpu_ctx():
        # 2x dim if self_cond_prob > 0 to initialize self_cond_proj layer
        _text_enc_dim = encoder_config.d_model
        input_dim = 2 * _text_enc_dim if config.self_cond_prob > 0 else _text_enc_dim
        dummy_x = jnp.ones((1, max_length, input_dim))
        dummy_t = jnp.ones((1,))
        dummy_self_cond_cfg_scale = jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None
        log_for_0(f"Dummy x shape: {dummy_x.shape}")
        log_for_0(f"Dummy t shape: {dummy_t.shape}")

    vocab_size = tokenizer.vocab_size
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )

    log_for_0("Initializing ELF model...")
    init_args = dict(
        x=dummy_x, t=dummy_t, deterministic=True,
        self_cond_cfg_scale=dummy_self_cond_cfg_scale,
    )
    with cpu_ctx():
        elf_params = model.init(init_rng, **init_args)
        log_for_0("\n" + model.tabulate(init_rng, **init_args))
        log_for_0("ELF initialization complete")

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(elf_params))
    log_for_0(f"ELF parameters: {total_params:,}")

    # ============================================
    # Create Train State Template
    # ============================================
    optimizer = optax.adamw(learning_rate=1e-4)
    with cpu_ctx():
        state = TrainState.create(
            apply_fn=model.apply,
            params=elf_params["params"],
            tx=optimizer,
            dropout_rng=dropout_rng,
            ema_params1=copy.deepcopy(elf_params["params"]),
        )

    # ============================================
    # Determine checkpoints to evaluate
    # ============================================
    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    log_for_0(f"Loading checkpoint from: {args.checkpoint_path}")
    state, _ = load_checkpoint(args.checkpoint_path, state)
    state_replicated = jax_utils.replicate(state)

    for seed_idx, seed_val in enumerate(seed_list):
        if len(seed_list) > 1:
            log_for_0(f"\n{'#' * 70}")
            log_for_0(f"Seed {seed_idx + 1}/{len(seed_list)}: {seed_val}")
            log_for_0(f"{'#' * 70}")

        seed_rng = jax.random.PRNGKey(seed_val)

        original_output_dir = config.output_dir
        if len(seed_list) > 1:
            config.output_dir = os.path.join(original_output_dir, f"seed_{seed_val}")

        for sc_idx, sc in enumerate(config.sampling_configs):
            if len(config.sampling_configs) > 1:
                log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
            seed_rng, sample_rng = jax.random.split(seed_rng)
            common_kwargs = dict(
                state=state_replicated,
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
                    encoder_apply_fn=encoder_model.apply,
                    dataset=eval_dataset,
                )

        config.output_dir = original_output_dir

    log_for_0("\nEvaluation complete!")


if __name__ == "__main__":
    main()
