import logging
import os
import pickle
import re
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import checkpoints

from utils.logging_utils import log_for_0


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def upload_output_dir_to_hf(output_dir: str, hf_repo_id: Optional[str], reason: str = "artifacts"):
    if not hf_repo_id or jax.process_index() != 0:
        return

    folder_path = _local_path(output_dir)
    if not os.path.isdir(folder_path):
        log_for_0(f"HF upload skipped; output directory does not exist: {folder_path}", level=logging.WARNING)
        return

    try:
        from huggingface_hub import HfApi

        repo_id = hf_repo_id.strip("/")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        log_for_0(f"Uploading {reason} to HF: {repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=folder_path, repo_type="model")
        log_for_0(f"Uploaded {reason} to HF: {repo_id}")
    except Exception as e:
        log_for_0(f"Failed to upload {reason} to HF: {e}", level=logging.WARNING)


def _split_hf_path(path: str, min_parts: int) -> Optional[Tuple[str, str]]:
    if "://" in path:
        return None
    if path.startswith(("/", ".", "~")):
        return None
    if os.path.exists(_local_path(path)):
        return None

    parts = path.split("/")
    if len(parts) < min_parts:
        return None

    repo_id = "/".join(parts[:2])
    sub_path = "/".join(parts[2:])
    return repo_id, sub_path



def save_checkpoint(state: Any, output_dir: str, step: int, hf_repo_id: str = None):
    """Save model checkpoint locally, optionally mirroring the output dir to HF."""
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    state_dict = {
        "params": state.params,
        "ema_params1": state.ema_params1,
        "opt_state": state.opt_state,
        "step": int(state.step),
        "epoch": int(state.epoch),
        "dropout_rng": state.dropout_rng,
    }

    ckpt_dir = _local_path(output_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_for_0(f"Saving checkpoint to {ckpt_dir}")

    checkpoints.save_checkpoint_multiprocess(
        ckpt_dir, state_dict, step, keep=10, overwrite=True,
    )

    log_for_0(f"Checkpoint written to {ckpt_dir}")
    upload_output_dir_to_hf(output_dir, hf_repo_id, reason="checkpoint")

# ============================================
# Encoder checkpoint (single pickle file)
# ============================================

def load_encoder_checkpoint(checkpoint_path: str):
    """Load a pickled encoder checkpoint from HF first, then local fallback.

    HF form: '<org>/<repo>/<filename>'.
    """
    if not checkpoint_path:
        raise ValueError(
            "encoder_checkpoint is not set. Provide a local path or HF Hub path "
            "like 'embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl'."
        )

    log_for_0(f"Loading encoder checkpoint from {checkpoint_path}...")
    loaded_params, loaded_from = None, None
    errors = []

    try:
        hf_path = _download_hf_file(checkpoint_path)
        if hf_path:
            loaded_params = _load_pickle(hf_path)
            loaded_from = "HF"
    except Exception as e:
        errors.append(f"HF: {e}")
        log_for_0(f"HF encoder checkpoint load failed ({e}); falling back to local path.")

    if loaded_params is None:
        local_path = _local_path(checkpoint_path)
        try:
            loaded_params = _load_pickle(local_path)
            loaded_from = "local"
        except Exception as e:
            errors.append(f"local: {e}")
            raise FileNotFoundError(
                f"Failed to load encoder checkpoint from {checkpoint_path}. "
                f"Tried: {'; '.join(errors)}"
            ) from e

    if isinstance(loaded_params, dict) and "params" in loaded_params:
        loaded_params = loaded_params["params"]
    log_for_0(f"Loaded {loaded_from} encoder checkpoint.")
    return loaded_params


def _load_pickle(path: str):
    log_for_0(f"Loading encoder checkpoint from {path}...")
    with open(path, "rb") as f:
        return pickle.load(f)


def _download_hf_file(path: str) -> Optional[str]:
    """Download a single file from HF Hub and return its local cache path."""
    hf_path = _split_hf_path(path, min_parts=3)
    if hf_path is None:
        return None
    repo_id, filename = hf_path

    try:
        from huggingface_hub import hf_hub_download

        log_for_0(f"Downloading checkpoint file from HF: {repo_id}/{filename}")
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    except Exception as e:
        raise FileNotFoundError(f"HF checkpoint file not found: {path} ({e})") from e


def _checkpoint_step(checkpoint_name: str) -> int:
    """Extract the trailing checkpoint step from a name; -1 if absent."""
    match = re.search(r"(\d+)$", checkpoint_name)
    return int(match.group(1)) if match else -1


# ============================================
# Resume: list + load (local or HF)
# ============================================

def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Find local checkpoint paths in a directory, sorted by step ascending."""
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    names = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith(prefix)],
        key=_checkpoint_step,
    )
    return [os.path.join(ckpt_dir, name) for name in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    """Return the latest local checkpoint path, or None."""
    all_ckpts = find_all_checkpoints(ckpt_dir, prefix)
    return all_ckpts[-1] if all_ckpts else None


def _download_hf_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Download an HF checkpoint snapshot and return the local checkpoint path."""
    hf_path = _split_hf_path(checkpoint_path, min_parts=2)
    if hf_path is None:
        return None
    repo_id, sub_path = hf_path

    from huggingface_hub import snapshot_download

    log_for_0(f"Downloading checkpoint from HF: {repo_id}" + (f"/{sub_path}" if sub_path else ""))
    local_dir = snapshot_download(
        repo_id=repo_id, repo_type="model",
        allow_patterns=[f"{sub_path}/**"] if sub_path else None,
    )
    return os.path.join(local_dir, sub_path) if sub_path else local_dir


def _checkpoint_target(state_template: Any):
    return {
        "params": state_template.params,
        "ema_params1": state_template.ema_params1,
        "opt_state": state_template.opt_state,
        "step": state_template.step,
        "epoch": state_template.epoch,
        "dropout_rng": state_template.dropout_rng,
    }


def _restore_checkpoint(checkpoint_path: str, target: Any):
    """Restore a checkpoint from a file or directory.

    Tries (in order):
      1. flax.serialization.from_bytes on a file (format written by save_checkpoint)
      2. flax.training.checkpoints.restore_checkpoint for HF pre-trained checkpoints
         that may have been saved with the old Flax msgpack / orbax format.
    """
    local = _local_path(checkpoint_path)

    # Resolve directory → latest checkpoint file
    resolved = local
    if os.path.isdir(local):
        latest = find_latest_checkpoint(local)
        if latest is not None and os.path.isfile(latest):
            resolved = latest

    if os.path.isfile(resolved):
        try:
            with open(resolved, "rb") as f:
                data = f.read()
            return serialization.from_bytes(target, data)
        except Exception:
            pass

    # Fallback: old Flax/orbax format (e.g., HF pre-trained checkpoints saved before
    # this change).
    try:
        from flax.training import checkpoints as _ckpts
        return _ckpts.restore_checkpoint(local, target=target)
    except Exception:
        return None


def _validate_checkpoint(ckpt: Any):
    if ckpt is None:
        raise ValueError("checkpoint restore returned None")
    required_keys = ("params", "opt_state", "step", "epoch", "dropout_rng")
    missing_keys = [key for key in required_keys if key not in ckpt]
    if missing_keys:
        raise ValueError(f"checkpoint restore missing keys: {missing_keys}")


def load_checkpoint(checkpoint_path: str, state_template: Any) -> Tuple[Any, int]:
    """Load an ELF checkpoint.

    Uses an existing local path first; otherwise tries HF and then local fallback.
    """
    log_for_0(f"Loading ELF checkpoint from {checkpoint_path}...")

    target = _checkpoint_target(state_template)
    ckpt, loaded_from = None, None
    errors = []

    local_path = _local_path(checkpoint_path)
    if os.path.exists(local_path):
        try:
            log_for_0(f"Loading local checkpoint from {local_path}...")
            ckpt = _restore_checkpoint(local_path, target)
            _validate_checkpoint(ckpt)
            loaded_from = "local"
        except Exception as e:
            errors.append(f"local: {e}")

    if ckpt is None:
        try:
            hf_path = _download_hf_checkpoint(checkpoint_path)
            if hf_path:
                log_for_0(f"Loading HF checkpoint from {hf_path}...")
                ckpt = _restore_checkpoint(hf_path, target)
                _validate_checkpoint(ckpt)
                loaded_from = "HF"
        except Exception as e:
            errors.append(f"HF: {e}")
            log_for_0(f"HF checkpoint restore failed ({e}); falling back to local path.")

    if ckpt is None and not os.path.exists(local_path):
        try:
            log_for_0(f"Loading local checkpoint from {local_path}...")
            ckpt = _restore_checkpoint(local_path, target)
            _validate_checkpoint(ckpt)
            loaded_from = "local"
        except Exception as e:
            errors.append(f"local: {e}")

    if ckpt is None:
        raise ValueError(
            f"Failed to load checkpoint from {checkpoint_path}. "
            f"Tried: {'; '.join(errors)}"
        )

    log_for_0(f"Loaded checkpoint keys: {ckpt.keys()}")

    restored_state = state_template.replace(
        params=jax.tree_util.tree_map(jnp.array, ckpt["params"]),
        ema_params1=jax.tree_util.tree_map(jnp.array, ckpt.get("ema_params1", ckpt["params"])),
        opt_state=ckpt["opt_state"],
        step=ckpt["step"],
        epoch=ckpt["epoch"],
        dropout_rng=jnp.array(ckpt["dropout_rng"]),
    )
    step, epoch = int(ckpt["step"]), int(ckpt["epoch"])
    log_for_0(f"Loaded {loaded_from} checkpoint from step {step} (epoch {epoch})")
    return restored_state, step
