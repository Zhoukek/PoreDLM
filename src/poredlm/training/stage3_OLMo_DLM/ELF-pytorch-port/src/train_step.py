"""Per-device pmap'd training step for the ELF diffusion language model."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from utils.train_utils import TrainState
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps,
    net_out_to_v_x, restore_cond,
)


Array = jnp.ndarray


def train_step(
    state: TrainState,
    encoder_params: Dict,
    encoder_apply_fn,
    batch: Dict[str, Array],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single training step."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std

    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    new_dropout_rng, current_step_rng = jax.random.split(state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(current_step_rng, jax.lax.axis_index(axis_name="batch"))
    # The 11-way split (rather than 10) preserves the exact RNG stream of our
    # released checkpoints so resumed runs are bit-for-bit reproducible.
    (
        t_rng, noise_rng, self_cond_mask_rng, self_cond_cfg_rng, _,
        model_dropout_rng, decoder_step_rng, decoder_rng,
        decoder_lambda_rng, decoder_noise_rng, _,
    ) = jax.random.split(current_step_rng, 11)

    # encoder_attention_mask: cond sees cond, x sees all
    encoder_attention_mask = batch["encoder_attention_mask"]

    # Label drop before encoding: prevent target tokens from attending to
    # condition tokens so x0 is truly unconditional for dropped samples.
    if config.label_drop_prob > 0:
        drop = batch["label_drop_mask"][:, None, None]  # (B, 1, 1)
        cond_mask = batch["cond_seq_mask"]  # (B, S)
        # block_mask is 1 only at (non-cond row, cond col) — leaves cond↔cond unchanged
        block_mask = (1 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=batch["input_ids"],
        attention_mask=encoder_attention_mask,
        encoder_apply_fn=encoder_apply_fn,
        encoder_params=encoder_params,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    t = sample_timesteps(
        t_rng, batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )

    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)

    cond_seq_mask = batch["cond_seq_mask"][:, :, None]
    attention_mask = batch["attention_mask"]
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = jnp.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    drop = batch["label_drop_mask"][:, None]
    if config.label_drop_prob > 0:
        denoiser_z = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(denoiser_z), denoiser_z)
        x0 = jnp.where(drop[:, :, None] & (cond_seq_mask > 0), jnp.zeros_like(x0), x0)

    decoder_targets = batch["input_ids"]  # (B, S)
    decoder_step_active = jax.random.bernoulli(decoder_step_rng, decoder_prob)

    # Decoder-branch input: logit-normal-noised latent (decoder_z) at t=1
    decoder_lambda_rng, decoder_noise_rng = jax.random.split(decoder_rng)
    decoder_z_vals = (
        jax.random.normal(decoder_lambda_rng, (batch_size * seq_length,))
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = jax.nn.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = jax.random.normal(decoder_noise_rng, x0.shape, dtype=x0.dtype) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / jnp.maximum(1 - t_expanded, t_eps)

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (jax.random.uniform(self_cond_mask_rng, (batch_size,)) < self_cond_prob)
            .reshape(-1, 1, 1).astype(x0.dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            self_cond_cfg_rng, batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
        )
    else:
        self_cond_cfg_scale = None

    def get_z_input(params, z, t_input, self_cond_cfg_input, x_tokens):
        # Self-conditioning: with probability self_cond_prob, compute initial estimate
        if self_cond_prob == 0:
            return z
        z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_seq_mask)
        z_with_zeros = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_init = state.apply_fn(
            {"params": params}, z_with_zeros, t_input,
            deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_input,
        )
        net_out_init = jax.lax.stop_gradient(net_out_init)
        _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, t_eps)
        x_pred_init = restore_cond(x_pred_init, x_tokens, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.astype(z.dtype)
        x_pred_cond = restore_cond(x_pred_cond, x_tokens, cond_seq_mask)
        return jnp.concatenate([z, x_pred_cond], axis=-1)

    def reduce_token_loss(per_token_loss, loss_mask):
        loss_mask = loss_mask.astype(per_token_loss.dtype)
        safe_loss = jnp.where(loss_mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
        return (safe_loss * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)

    def get_sc_cond_and_uncond(params, z, t, cond_mask, x_tokens):
        kwargs = {
            "self_cond_cfg_scale": self_cond_cfg_scale,
            "deterministic": True,
        }
        if config.self_cond_prob == 0:
            net_out_uncod = state.apply_fn({"params": params}, z, t, **kwargs)
            v_uncond, _ = net_out_to_v_x(net_out_uncod, z, t, t_eps)
            return v_uncond, v_uncond

        z_uncond = restore_cond(jnp.zeros_like(z), x_tokens, cond_mask)
        z_input_uncond = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_uncond = state.apply_fn({"params": params}, z_input_uncond, t, **kwargs)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = jnp.concatenate([z, x_uncond], axis=-1)
        net_out_cond = state.apply_fn({"params": params}, z_input_cond, t, **kwargs)
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t, t_eps)
        return v_cond, v_uncond

    def get_sc_guided_v(params, z, t, base_v_target, x_tokens):
        """v target with self-conditioning guidance."""
        v_cond, v_uncond = get_sc_cond_and_uncond(
            params, z, t, cond_mask=cond_seq_mask, x_tokens=x_tokens
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = jnp.where(use_self_cond_mask, sc_guidance, jnp.zeros_like(sc_guidance))
        return jax.lax.stop_gradient(base_v_target + sc_guidance)

    def get_v_target(params, z, t, base_v_target, x_tokens):
        """Compute final v target with self-conditioning guidance."""
        if config.num_self_cond_cfg_tokens > 0:
            return get_sc_guided_v(params, z, t, base_v_target=base_v_target, x_tokens=x_tokens)
        return base_v_target

    def loss_fn(params):

        def _decoder_branch(_):
            # Decoder mode: encoder-noised latent (decoder_z) at t=1, CE loss on tokens.
            decoder_t = jnp.ones_like(t)
            decoder_input = (
                jnp.concatenate([decoder_z, jnp.zeros_like(decoder_z)], axis=-1)
                if config.self_cond_prob > 0 else decoder_z
            )
            _, decoder_logits = state.apply_fn(
                {"params": params}, decoder_input, decoder_t,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(True),
            )
            log_probs = jax.nn.log_softmax(decoder_logits.astype(jnp.float32), axis=-1)
            ce = -jnp.take_along_axis(log_probs, decoder_targets[..., None], axis=-1).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
            return ce_loss, ce_loss, jnp.zeros(())

        def _denoiser_branch(_):
            # Denoiser mode: x0-noised latent (denoiser_z) at random t, L2 loss on velocity.
            denoiser_t = t
            denoiser_input = get_z_input(
                params, denoiser_z, denoiser_t,
                self_cond_cfg_input=self_cond_cfg_scale,
                x_tokens=x0,
            )
            net_out, _ = state.apply_fn(
                {"params": params}, denoiser_input, denoiser_t,
                deterministic=False,
                rngs={"dropout": model_dropout_rng},
                self_cond_cfg_scale=self_cond_cfg_scale,
                decoder_step_active=jnp.array(False),
            )
            v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
            v_final_target = get_v_target(
                params, denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
            )
            per_dim_loss = (v_pred - v_final_target) ** 2
            l2_loss = reduce_token_loss(jnp.mean(per_dim_loss, axis=-1), loss_mask)
            return l2_loss, jnp.zeros(()), l2_loss

        loss, ce_loss, l2_loss = jax.lax.cond(
            decoder_step_active, _decoder_branch, _denoiser_branch, None,
        )
        return loss, (l2_loss, ce_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l2_loss_val, ce_loss_val)), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    l2_loss_val = jax.lax.pmean(l2_loss_val, axis_name="batch")
    ce_loss_val = jax.lax.pmean(ce_loss_val, axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    # Update EMA only on actual optimizer steps, not on gradient accumulation steps.
    # With optax.MultiSteps, params only change every grad_accum_steps mini-batches; updating
    # EMA every mini-batch would make effective decay decay^grad_accum_steps instead of decay.
    def ema_update(ema_params, params, decay):
        return jax.tree_util.tree_map(lambda e, p: e * decay + p * (1 - decay), ema_params, params)

    is_optimizer_step = (new_state.step % config.grad_accum_steps) == 0
    new_ema_params1 = jax.lax.cond(
        is_optimizer_step,
        lambda: ema_update(state.ema_params1, new_state.params, config.ema_decay1),
        lambda: state.ema_params1,
    )
    new_state = new_state.replace(ema_params1=new_ema_params1, dropout_rng=new_dropout_rng)

    # Rescale per-branch losses by their sampling probability so they reflect the
    # per-branch loss rather than the expected loss conditioned on the branch firing.
    decoder_prob_arr = jnp.asarray(decoder_prob, dtype=jnp.float32)
    denoiser_prob_arr = jnp.asarray(1.0 - decoder_prob, dtype=jnp.float32)
    active_ce_loss_val = jnp.where(
        decoder_prob_arr > 0.0, ce_loss_val / decoder_prob_arr, jnp.zeros_like(ce_loss_val),
    )
    active_l2_loss_val = jnp.where(
        denoiser_prob_arr > 0.0, l2_loss_val / denoiser_prob_arr, jnp.zeros_like(l2_loss_val),
    )
    metrics = {
        "loss": loss,
        "l2_loss": active_l2_loss_val,
        "ce_loss": active_ce_loss_val,
    }
    return new_state, metrics
