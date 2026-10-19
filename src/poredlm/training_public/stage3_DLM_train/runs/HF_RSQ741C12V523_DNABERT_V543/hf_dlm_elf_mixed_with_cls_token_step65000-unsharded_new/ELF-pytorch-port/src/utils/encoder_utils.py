from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(2,))
def encode_text(
    input_ids, attention_mask, encoder_apply_fn, encoder_params,
    latent_mean, latent_std,
):
    """Encoder pass from text to latent with normalization."""
    latents = encoder_apply_fn(
        {"params": encoder_params}, input_ids=input_ids, attention_mask=attention_mask,
        deterministic=True,
    )
    return (latents - latent_mean) / latent_std


def build_self_attn_cond_masks(is_cond, is_valid, xp=jnp):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
