"""Training-time utilities: train state, optimizer/schedule helpers.

Extracted from train.py for reuse and readability.
"""

import queue
import threading
from typing import Any

import jax
import optax
from flax.training import train_state

from utils.logging_utils import log_for_0

PRNGKey = jax.random.PRNGKey


# ============================================
# Train State with EMA
# ============================================
class TrainState(train_state.TrainState):
    dropout_rng: PRNGKey
    ema_params1: Any = None
    epoch: int = 0


def prefetch_to_device(iterator, size=2):
    """Prefetch batches to device asynchronously."""
    q = queue.Queue(maxsize=size)

    def enqueue():
        for item in iterator:
            q.put(item)
        q.put(None)

    threading.Thread(target=enqueue, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


# ============================================
# Optimizer
# ============================================
def get_optimizer(config, lr_schedule, grad_accum_steps: int = 1):
    """Build optax chain (gradient clipping + AdamW/Muon).

    grad_accum_steps > 1 wraps the inner optimizer in optax.MultiSteps so optimizer
    state only updates every K mini-batches.
    """
    if config.optimizer == "muon":
        inner = optax.contrib.muon(learning_rate=lr_schedule)
    elif config.optimizer == "adamw":
        inner = optax.adamw(
            learning_rate=lr_schedule, weight_decay=config.weight_decay,
            b1=config.adam_b1, b2=config.adam_b2,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}. Choose 'adamw' or 'muon'.")

    log_for_0(f"Using {'Muon' if config.optimizer == 'muon' else 'AdamW'} optimizer")
    if grad_accum_steps > 1:
        inner = optax.MultiSteps(inner, every_k_schedule=grad_accum_steps)
    return optax.chain(optax.clip_by_global_norm(1.0), inner)


# ============================================
# Learning Rate Schedule
# ============================================
def create_learning_rate_fn(
    num_train_steps: int,
    num_warmup_steps: int,
    learning_rate: float,
    schedule: str = "constant",
    min_lr: float = 0.0,
):
    """Create learning rate schedule with linear warmup."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps,
    )
    if schedule == "cosine":
        decay_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=num_train_steps - num_warmup_steps,
            alpha=min_lr / learning_rate if learning_rate > 0 else 0.0,
        )
    else:
        decay_fn = optax.constant_schedule(learning_rate)
    return optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
