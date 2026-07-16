from functools import partial

import jax
import jax.numpy as jnp
from flax import jax_utils
from jax import Array

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.sampling_utils import (
    restore_cond, _ode_step, _sde_step, get_sampling_steps,
)
from modules.t5_encoder import get_encoder

PRNGKey = jax.random.PRNGKey


# ============================================
# Generation utilities
# ============================================


def mask_after_eos(predicted_ids, eos_token_id, pad_token_id):
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = predicted_ids == eos_token_id
    keep_mask = jnp.cumsum(eos_mask, axis=1) == 0
    return jnp.where(keep_mask, predicted_ids, pad_token_id)


def shift_left(x, shift_per_sample, pad_value=0, axis=1):
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.ndim < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    axis = axis if axis >= 0 else x.ndim + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.astype(jnp.int32)
    if axis != 1:
        x = jnp.moveaxis(x, axis, 1)
    seq_len = x.shape[1]
    base_idx = jnp.arange(seq_len)[None, :]
    gather_idx = shift_per_sample[:, None] + base_idx
    valid = gather_idx < seq_len
    gather_idx = jnp.clip(gather_idx, 0, seq_len - 1)
    if x.ndim == 2:
        shifted = jnp.take_along_axis(x, gather_idx, axis=1)
        shifted = jnp.where(valid, shifted, pad_value)
    else:
        expand_axes = tuple(range(2, x.ndim))
        shifted = jnp.take_along_axis(x, jnp.expand_dims(gather_idx, expand_axes), axis=1)
        shifted = jnp.where(jnp.expand_dims(valid, expand_axes), shifted, pad_value)
    if axis != 1:
        shifted = jnp.moveaxis(shifted, 1, axis)
    return shifted


# ============================================
# Multi-device helpers (pmap)
# ============================================

def _sample_step_for_scan(
    model_apply_fn, model_params, config, sampling_config: SamplingConfig,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, rng=None,
):
    """Create a scan-compatible step function.

    For method == "sde", `rng` must be provided and the scan carry must include a step index
    (z, x_pred, step_idx); fold_in is done per step. Other methods use a (z, x_pred) carry.
    """
    method = sampling_config.sampling_method
    base_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    if method == "sde":
        assert rng is not None, "SDE method requires rng to be passed to _sample_step_for_scan"
        sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

        def step_fn(carry, t_pair):
            z, x_pred, step_idx = carry
            t, t_next = t_pair
            step_rng = jax.random.fold_in(rng, step_idx)
            z_new, x_pred_new = _sde_step(
                z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                gamma=sde_gamma, rng=step_rng, **base_kwargs,
            )
            return (z_new, x_pred_new, step_idx + 1), None
        return step_fn

    if method == "ode":
        base_step_fn = _ode_step
    else:
        raise ValueError(f"Invalid sampling method: {method}")

    def step_fn(carry, t_pair):
        z, x_pred = carry
        t, t_next = t_pair
        z_new, x_pred_new = base_step_fn(
            z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **base_kwargs,
        )
        return (z_new, x_pred_new), None
    return step_fn


def _generate_samples_single_batch(
    model_params, model_apply_fn, rng: PRNGKey, z: Array, t_steps: Array,
    cond_seq: Array, cond_seq_mask: Array, config: Config, sampling_config: SamplingConfig,
    cfg_scale: float, self_cond_cfg_scale: float,
) -> Array:
    """Generate samples for a single batch (pmap-compatible, uses lax.scan)."""
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = jnp.zeros((batch_size, max_length, d_model))
        cond_seq_mask = jnp.zeros((batch_size, max_length))
    step_kwargs = dict(
        model_apply_fn=model_apply_fn, model_params=model_params,
        config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(jnp.zeros_like(z), cond_seq, cond_seq_mask)

    t_pairs = jnp.stack([t_steps[:-2], t_steps[1:-1]], axis=1)
    if method == "sde":
        step_fn = _sample_step_for_scan(sampling_config=sampling_config, rng=rng, **step_kwargs)
        (z, x_pred, _), _ = jax.lax.scan(step_fn, (z, x_pred, jnp.int32(0)), t_pairs)
    else:
        step_fn = _sample_step_for_scan(sampling_config=sampling_config, **step_kwargs)
        (z, x_pred), _ = jax.lax.scan(step_fn, (z, x_pred), t_pairs)

    # Last step always with ode
    z, x_pred = _ode_step(
        z=z, t=t_steps[-2], t_next=t_steps[-1], x_pred_prev=x_pred, **step_kwargs,
    )
    return z


def _dlm_decode_batch(z, model_params, model_apply_fn, t_final_val, config, self_cond_cfg_scale):
    """Decode z→tokens with the DLM decoder head."""
    batch_size = z.shape[0]
    t_final = jnp.full((batch_size,), t_final_val, dtype=z.dtype)
    self_cond_cfg_scale_batch = (
        jnp.full((batch_size,), self_cond_cfg_scale, dtype=z.dtype)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = jnp.concatenate([z, jnp.zeros_like(z)], axis=-1) if config.self_cond_prob > 0 else z
    _, decoder_logits = model_apply_fn(
        {"params": model_params}, z_input, t_final,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale_batch,
        decoder_step_active=jnp.array(True),
    )
    return jnp.argmax(decoder_logits, axis=-1)


# ============================================
# Shared generation scaffolding
# ============================================
def _make_pmap_pair(model_apply_fn, config, sampling_config, cfg_scale, self_cond_cfg_scale):
    """Build pmapped (generate, decode) pair for a (cfg, sccfg) combo."""
    p_generate = jax.pmap(
        partial(
            _generate_samples_single_batch,
            model_apply_fn=model_apply_fn, config=config, sampling_config=sampling_config,
            cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        ),
        axis_name="batch",
    )
    p_decode_ids = jax.pmap(
        partial(
            _dlm_decode_batch, model_apply_fn=model_apply_fn, config=config,
            self_cond_cfg_scale=self_cond_cfg_scale,
        )
    )
    return p_generate, p_decode_ids


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}-{suffix}"


def _shard_timesteps(t_rng, num_local_devices, num_sampling_steps, time_schedule, config):
    t_device_rngs = jax.random.split(t_rng, num_local_devices)
    return jnp.stack([
        get_sampling_steps(
            t_device_rngs[i], n_steps=num_sampling_steps,
            time_schedule=time_schedule, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        )
        for i in range(num_local_devices)
    ])


def _shard_noise(device_rngs, num_local_devices, per_device, max_length, d_model, noise_scale):
    return jnp.stack([
        jax.random.normal(device_rngs[i], (per_device, max_length, d_model)) * noise_scale
        for i in range(num_local_devices)
    ])


def _setup_generation(state, config, batch_size, header):
    """Shared setup: log header, unreplicate state, build replicated model_params, compute batch sizes."""
    log_for_0("\n" + "=" * 70)
    log_for_0(header)
    log_for_0("=" * 70)

    num_local_devices = jax.local_device_count()
    log_for_0(f"Using {num_local_devices} local devices for generation")

    state_unreplicated = jax_utils.unreplicate(state)
    model_apply_fn = state_unreplicated.apply_fn

    encoder_config, _, _ = get_encoder(config.encoder_model_name, None)
    d_model = encoder_config.d_model

    model_params_replicated = jax_utils.replicate(state_unreplicated.ema_params1)

    per_device_batch = max(1, batch_size // num_local_devices)
    effective_batch_size = per_device_batch * num_local_devices
    log_for_0(f"Per-device batch size: {per_device_batch}, effective batch size: {effective_batch_size}")

    return state_unreplicated, model_apply_fn, model_params_replicated, d_model, num_local_devices, effective_batch_size


