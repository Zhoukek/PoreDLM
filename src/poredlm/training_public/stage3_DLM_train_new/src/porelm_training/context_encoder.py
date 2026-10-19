from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml
from transformers import AutoModel


class MaskedSignalEncoder(nn.Module):
    """Compatibility loader for native Stage-2 PyTorch checkpoints."""

    def __init__(
        self,
        *,
        max_length: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        vocab_size: int,
        pad_token_id: int,
    ):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.num_heads = num_heads
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Any:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        if attention_mask.ndim == 3:
            block_mask = ~attention_mask.bool().repeat_interleave(self.num_heads, dim=0)
            x = self.encoder(x, mask=block_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        output = self.norm(x)
        return SimpleNamespace(last_hidden_state=output) if return_dict else (output,)


def _strip_wrapping_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        result[key] = value
    return result


def _native_checkpoint(path: Path) -> tuple[Path, Optional[Path]] | None:
    if path.is_file() and path.suffix in {".pt", ".pth"}:
        state = path.parent / "trainer_state.pt"
        return path, state if state.exists() else None
    if not path.is_dir():
        return None
    state = path / "trainer_state.pt"
    weights = path / "model_state.pt"
    if weights.exists():
        return weights, state if state.exists() else None
    if state.exists():
        metadata = torch.load(state, map_location="cpu", weights_only=False)
        if metadata.get("model_format") == "hf_pretrained" and "model_state_path" not in metadata:
            return None
        return path / metadata.get("model_state_path", "model_state.pt"), state
    return None


def _load_native_config(weights: Path, metadata_path: Optional[Path], payload: Any) -> dict[str, Any]:
    for candidate in (payload, torch.load(metadata_path, map_location="cpu", weights_only=False) if metadata_path else {}):
        if isinstance(candidate, dict) and isinstance(candidate.get("config"), dict):
            return candidate["config"]
    for config_path in (weights.parent / "config.yaml", weights.parent.parent / "config.yaml"):
        if config_path.exists():
            with config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if isinstance(config, dict):
                return config
    raise ValueError(f"Could not locate the Stage-2 configuration for {weights}")


def load_context_encoder(path_value: str) -> nn.Module:
    path = Path(path_value).expanduser()
    native = _native_checkpoint(path)
    if native is None:
        return AutoModel.from_pretrained(path_value, trust_remote_code=True)

    weights, metadata = native
    if not weights.exists():
        raise FileNotFoundError(weights)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported Stage-2 checkpoint payload in {weights}")
    config = _load_native_config(weights, metadata, payload)
    model_config = config.get("model", {})
    encoder = MaskedSignalEncoder(
        max_length=int(model_config.get("max_position_embeddings", model_config.get("max_seq_len", 1024))),
        hidden_size=int(model_config.get("d_model", model_config.get("hidden_size", 512))),
        num_layers=int(model_config.get("layers", model_config.get("num_hidden_layers", 8))),
        num_heads=int(model_config.get("heads", model_config.get("num_attention_heads", 8))),
        dropout=float(model_config.get("dropout", model_config.get("hidden_dropout_prob", 0.1))),
        vocab_size=int(model_config.get("vocab_size", 2056)),
        pad_token_id=int(model_config.get("pad_token_id", 0)),
    )
    state_dict = {
        key: value
        for key, value in _strip_wrapping_prefixes(state_dict).items()
        if not key.startswith("lm_head.")
    }
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("lm_head.")]
    if missing or unexpected:
        raise ValueError(f"Stage-2 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return encoder

