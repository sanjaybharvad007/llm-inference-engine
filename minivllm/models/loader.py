"""HuggingFace checkpoint loading (config.json + *.safetensors)."""
from __future__ import annotations

import glob
import json
import os

import torch

from ..config import ModelConfig


def hf_config_to_model_config(cfg: dict, dtype: torch.dtype) -> ModelConfig:
    return ModelConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_layers=cfg["num_hidden_layers"],
        num_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        head_dim=cfg.get("head_dim"),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        bos_token_id=cfg.get("bos_token_id", 1),
        eos_token_id=cfg.get("eos_token_id", 2),
        dtype=dtype,
    )


def load_hf_llama(model_cls, path: str, dtype: torch.dtype, device: str):
    from safetensors.torch import load_file  # optional dependency

    with open(os.path.join(path, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = hf_config_to_model_config(hf_cfg, dtype)
    with torch.device("meta"):
        model = model_cls(cfg)
    model = model.to_empty(device=device)
    model.rope = type(model.rope)(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta).to(device)

    state = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        state.update(load_file(shard, device=device))
    mapped = {}
    for name, tensor in state.items():
        n = name.replace("model.", "", 1) if name.startswith("model.") else name
        mapped[n] = tensor.to(dtype)
    if cfg.tie_word_embeddings and "lm_head.weight" not in mapped:
        mapped["lm_head.weight"] = mapped["embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [m for m in missing if not m.startswith("rope.")]
    if missing:
        raise RuntimeError(f"missing weights: {missing[:5]}...")
    return model.eval()
