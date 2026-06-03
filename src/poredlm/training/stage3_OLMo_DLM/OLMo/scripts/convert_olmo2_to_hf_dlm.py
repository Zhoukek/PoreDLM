from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:  # pragma: no cover - optional dependency
    save_safetensors = None


OLMO_ROOT = Path(__file__).resolve().parents[1]
if str(OLMO_ROOT) not in sys.path:
    sys.path.insert(0, str(OLMO_ROOT))

ELF_SRC = OLMO_ROOT.parent / "ELF-pytorch-port" / "src"
if ELF_SRC.is_dir() and str(ELF_SRC) not in sys.path:
    sys.path.insert(0, str(ELF_SRC))

from olmo.config import TrainConfig  # noqa: E402
from olmo.model_DLM import OLMoDLM  # noqa: E402


MODELING_POREDLM = r'''
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertModel, PretrainedConfig, PreTrainedModel


class PoreDLMConfig(PretrainedConfig):
    model_type = "poredlm_dlm"

    def __init__(
        self,
        context_encoder_config: Optional[dict[str, Any]] = None,
        dlm_config: Optional[dict[str, Any]] = None,
        model_config: Optional[dict[str, Any]] = None,
        elf_src_path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.context_encoder_config = context_encoder_config or {}
        self.dlm_config = dlm_config or {}
        self.model_config = model_config or {}
        self.elf_src_path = elf_src_path


class PoreDLMForDiffusion(PreTrainedModel):
    config_class = PoreDLMConfig
    base_model_prefix = "poredlm"
    main_input_name = "input_ids"

    def __init__(self, config: PoreDLMConfig):
        super().__init__(config)
        if config.elf_src_path and Path(config.elf_src_path).is_dir() and config.elf_src_path not in sys.path:
            sys.path.insert(0, config.elf_src_path)
        try:
            from torch_elf.model import ELF_models
        except ImportError as exc:
            raise ImportError(
                "Cannot import torch_elf. Add stage3_OLMo_DLM/ELF-pytorch-port/src to PYTHONPATH "
                "before loading this HF model."
            ) from exc

        self.context_encoder = BertModel(BertConfig.from_dict(config.context_encoder_config))
        self.context_hidden_size = self.context_encoder.config.hidden_size

        dlm = config.dlm_config
        model_cfg = config.model_config
        model_name = dlm.get("model", "ELF-B")
        if model_name not in ELF_models:
            raise ValueError(f"Unknown ELF model {model_name!r}; expected one of {sorted(ELF_models.keys())}")
        self.elf_denoiser = ELF_models[model_name](
            text_encoder_dim=self.context_hidden_size,
            max_length=int(dlm.get("max_length") or model_cfg.get("max_sequence_length") or 1024),
            attn_drop=float(dlm.get("attn_dropout", 0.0)),
            proj_drop=float(dlm.get("proj_dropout", 0.0)),
            num_time_tokens=int(dlm.get("num_time_tokens", 4)),
            num_self_cond_cfg_tokens=int(dlm.get("num_self_cond_cfg_tokens", 4)),
            vocab_size=int(model_cfg.get("vocab_size", 50257)),
            num_model_mode_tokens=int(dlm.get("num_model_mode_tokens", 0)),
            bottleneck_dim=int(dlm.get("bottleneck_dim", 128)),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cond_seq_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[torch.Tensor] = None,
        return_context: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if encoder_attention_mask is None:
            encoder_attention_mask = attention_mask
        if encoder_attention_mask is None:
            encoder_attention_mask = input_ids.new_ones(input_ids.shape)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        context_dtype = next(self.elf_denoiser.parameters()).dtype
        context = self.context_encoder(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state.to(dtype=context_dtype)

        if cond_seq_mask is not None:
            cond_seq_mask = cond_seq_mask.to(device=context.device, dtype=context.dtype).unsqueeze(-1)
        if t is None:
            t = torch.ones(input_ids.shape[0], device=context.device, dtype=context.dtype)
        if self_cond is not None:
            model_input = torch.cat([context, self_cond.to(context)], dim=-1)
        else:
            model_input = context

        pred, decoder_logits = self.elf_denoiser(
            model_input,
            t.to(device=context.device, dtype=context.dtype),
            attention_mask=attention_mask,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,
        )
        if cond_seq_mask is not None:
            pred = cond_seq_mask * context + (1.0 - cond_seq_mask) * pred
        output = {"last_hidden_state": pred}
        if decoder_logits is not None:
            output["logits"] = decoder_logits
        if return_context:
            output["context_hidden_state"] = context
        return output


__all__ = ["PoreDLMConfig", "PoreDLMForDiffusion"]
'''


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _strip_fsdp_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("_fsdp_wrapped_module.", ""): value for key, value in state_dict.items()}


def _load_model(input_dir: Path) -> tuple[OLMoDLM, TrainConfig]:
    config_path = input_dir / "config.yaml"
    model_path = input_dir / "model.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.yaml in {input_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.pt in {input_dir}")

    cfg = TrainConfig.load(config_path, validate_paths=False)
    if cfg.dlm.context_encoder_path is None:
        raise ValueError("config.yaml must contain dlm.context_encoder_path to rebuild OLMoDLM.")

    cfg.model.init_device = "cpu"
    model = OLMoDLM(
        cfg.model,
        context_encoder_path=cfg.dlm.context_encoder_path,
        freeze_context_encoder=cfg.dlm.freeze_context_encoder,
        dlm_config=cfg.dlm,
    )
    state_dict = _strip_fsdp_prefixes(torch.load(model_path, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing[:20]}")
    model.eval()
    return model, cfg


def _write_config(output_dir: Path, model: OLMoDLM, cfg: TrainConfig) -> None:
    hf_config = {
        "model_type": "poredlm_dlm",
        "architectures": ["PoreDLMForDiffusion"],
        "auto_map": {
            "AutoConfig": "modeling_poredlm.PoreDLMConfig",
            "AutoModel": "modeling_poredlm.PoreDLMForDiffusion",
        },
        "context_encoder_config": model.context_encoder.config.to_dict(),
        "dlm_config": _json_safe(cfg.dlm.asdict()),
        "model_config": _json_safe(cfg.model.asdict()),
        "elf_src_path": str(ELF_SRC) if ELF_SRC.is_dir() else None,
        "torch_dtype": "float32",
    }
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "modeling_poredlm.py").write_text(MODELING_POREDLM.lstrip(), encoding="utf-8")


def _write_weights(output_dir: Path, model: OLMoDLM, safe_serialization: bool) -> None:
    state_dict = {key: tensor.detach().cpu().contiguous() for key, tensor in model.state_dict().items()}
    if safe_serialization:
        if save_safetensors is None:
            raise RuntimeError("safetensors is not installed; pass --no_safe_serialization to write pytorch_model.bin.")
        save_safetensors(state_dict, output_dir / "model.safetensors", metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "pytorch_model.bin")


def _decode_token(tokenizer: Tokenizer, token_id: int | None, fallback: str | None = None) -> str | None:
    if token_id is None:
        return fallback
    try:
        token = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        token = ""
    return token or fallback


def _write_tokenizer(output_dir: Path, tokenizer_json_path: Path | None, cfg: TrainConfig) -> None:
    tokenizer_path = tokenizer_json_path or Path(cfg.tokenizer.identifier)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")

    base = Tokenizer.from_file(str(tokenizer_path))
    vocab = base.get_vocab()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base,
        pad_token=_decode_token(base, cfg.model.pad_token_id, "[PAD]"),
        eos_token=_decode_token(base, cfg.model.eos_token_id, "[EOS]"),
        unk_token="[UNK]" if "[UNK]" in vocab else None,
        mask_token="[MASK]" if "[MASK]" in vocab else None,
    )
    tokenizer.model_max_length = cfg.model.max_sequence_length
    tokenizer.save_pretrained(output_dir)


def convert(
    input_dir: Path,
    output_dir: Path,
    tokenizer_json_path: Path | None,
    include_tokenizer: bool,
    safe_serialization: bool,
    overwrite: bool,
) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = _load_model(input_dir)
    _write_config(output_dir, model, cfg)
    _write_weights(output_dir, model, safe_serialization=safe_serialization)
    if include_tokenizer:
        _write_tokenizer(output_dir, tokenizer_json_path, cfg)
    shutil.copy2(input_dir / "config.yaml", output_dir / "olmo_train_config.yaml")

    print(f"Saved PoreDLM HF checkpoint to {output_dir}")
    print("Load with: AutoModel.from_pretrained(path, trust_remote_code=True)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Stage 3 OLMoDLM checkpoint to a HF custom model folder.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing config.yaml and model.pt.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output HF model directory.")
    parser.add_argument("--tokenizer_json_path", type=Path, default=None, help="Tokenizer JSON to save with the HF model.")
    parser.add_argument("--no_tokenizer", action="store_false", dest="include_tokenizer")
    parser.add_argument("--no_safe_serialization", action="store_false", dest="safe_serialization")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_dir before writing.")
    args = parser.parse_args()
    convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_json_path=args.tokenizer_json_path,
        include_tokenizer=args.include_tokenizer,
        safe_serialization=args.safe_serialization,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
