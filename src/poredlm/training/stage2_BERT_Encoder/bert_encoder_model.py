"""BERT encoder model built with HuggingFace Transformers."""

from __future__ import annotations

from transformers import BertConfig, BertForMaskedLM


def build_bert_config(config: dict) -> BertConfig:
    """Create a HuggingFace BertConfig from YAML model settings."""

    model_cfg = config.get("model", {})
    return BertConfig(
        vocab_size=int(model_cfg.get("vocab_size", 65536)),
        hidden_size=int(model_cfg.get("hidden_size", 768)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 12)),
        num_attention_heads=int(model_cfg.get("num_attention_heads", 12)),
        intermediate_size=int(model_cfg.get("intermediate_size", 3072)),
        hidden_dropout_prob=float(model_cfg.get("hidden_dropout_prob", 0.1)),
        attention_probs_dropout_prob=float(model_cfg.get("attention_probs_dropout_prob", 0.1)),
        max_position_embeddings=int(model_cfg.get("max_position_embeddings", 4096)),
        type_vocab_size=int(model_cfg.get("type_vocab_size", 1)),
        pad_token_id=int(model_cfg.get("pad_token_id", 0)),
    )


def build_bert_mlm(config: dict) -> BertForMaskedLM:
    """Build a BERT MLM model for VQ token ids."""

    return BertForMaskedLM(build_bert_config(config))
