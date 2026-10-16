"""Llama-architecture decoder (RMSNorm, RoPE, GQA, SwiGLU) running on a paged KV cache.

The forward pass takes a *flattened* batch: ``input_ids [T]`` for all
sequences' new tokens concatenated, with per-token positions and cache slots
and per-sequence block tables / context lengths.  Prefill chunks and decode
tokens are mixed freely in one call.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..distributed import parallel_state as ps
from ..distributed.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelLinear
from ..kernels.cache_ops import write_kv
from ..kernels.paged_attention import paged_attention


@dataclass
class AttnMetadata:
    positions: torch.Tensor        # int64 [T]
    slot_mapping: torch.Tensor     # int64 [T] physical cache slot of every new token
    block_tables: torch.Tensor     # int32 [num_seqs, max_blocks]
    cu_seqlens_q: torch.Tensor     # int32 [num_seqs + 1]
    context_lens: torch.Tensor     # int32 [num_seqs] (= computed + new tokens)
    max_q_len: int
    logits_indices: torch.Tensor   # int64 [num_sampling_seqs] rows of hidden whose logits we need


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_pos: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_pos).float()
        freqs = torch.outer(t, inv_freq)                        # [max_pos, D/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: [T, H, D] -> rotated (HF Llama half-split convention)."""
        cos = self.cos[positions][:, None, :].to(x.dtype)       # [T, 1, D/2]
        sin = self.sin[positions][:, None, :].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Heads are sharded across tensor-parallel ranks: each rank owns
    num_heads/tp query heads and num_kv_heads/tp KV heads (and the matching
    slice of the KV cache); o_proj reduces the partial outputs."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        tp = ps.tp_size()
        assert cfg.num_heads % tp == 0 and cfg.num_kv_heads % tp == 0, "heads must divide the TP size"
        self.h_q, self.h_kv, self.d = cfg.num_heads // tp, cfg.num_kv_heads // tp, cfg.head_dim
        self.q_proj = ColumnParallelLinear(cfg.hidden_size, cfg.num_heads * self.d)
        self.k_proj = ColumnParallelLinear(cfg.hidden_size, cfg.num_kv_heads * self.d)
        self.v_proj = ColumnParallelLinear(cfg.hidden_size, cfg.num_kv_heads * self.d)
        self.o_proj = RowParallelLinear(cfg.num_heads * self.d, cfg.hidden_size)
        self.scale = self.d ** -0.5

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding, kv_cache: tuple[torch.Tensor, torch.Tensor], meta: AttnMetadata):
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.h_q, self.d)
        k = self.k_proj(x).view(t, self.h_kv, self.d)
        v = self.v_proj(x).view(t, self.h_kv, self.d)
        q, k = rope(q, meta.positions), rope(k, meta.positions)
        key_cache, value_cache = kv_cache
        write_kv(k, v, key_cache, value_cache, meta.slot_mapping)      # new tokens become part of the context
        o = paged_attention(q, key_cache, value_cache, meta.block_tables, meta.cu_seqlens_q,
                            meta.context_lens, meta.max_q_len, self.scale)
        return self.o_proj(o.reshape(t, self.h_q * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, rope, kv_cache, meta):
        x = x + self.self_attn(self.input_layernorm(x), rope, kv_cache, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = VocabParallelLinear(cfg.hidden_size, cfg.vocab_size)
        if cfg.tie_word_embeddings and ps.tp_size() == 1:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.quant_config = None   # set by minivllm.quant.quantize_model

    # Parameter-name suffixes that are sharded under tensor parallelism.
    TP_COLUMN = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")
    TP_ROW = ("o_proj.weight", "down_proj.weight")
    TP_VOCAB = ("lm_head.weight",)

    def load_full_state_dict(self, full: dict[str, torch.Tensor]) -> None:
        """Load an unsharded state dict, slicing it for this rank."""
        from ..distributed.layers import shard_state_dict

        full = dict(full)
        if self.cfg.tie_word_embeddings and "lm_head.weight" not in full:
            full["lm_head.weight"] = full["embed_tokens.weight"]
        sd = shard_state_dict(full, ps.tp_rank(), ps.tp_size(), list(self.TP_COLUMN), list(self.TP_ROW), list(self.TP_VOCAB))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("rope.")]
        if missing or unexpected:
            raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def forward(self, input_ids: torch.Tensor, kv_caches: list[tuple[torch.Tensor, torch.Tensor]], meta: AttnMetadata) -> torch.Tensor:
        """Returns logits [len(meta.logits_indices), vocab] in fp32."""
        x = self.embed_tokens(input_ids)
        for layer, cache in zip(self.layers, kv_caches):
            x = layer(x, self.rope, cache, meta)
        x = self.norm(x[meta.logits_indices])
        return self.lm_head(x).float()

    # -- cache sizing ----------------------------------------------------------
    @property
    def local_kv_heads(self) -> int:
        return self.cfg.num_kv_heads // ps.tp_size()

    def kv_bytes_per_block(self, block_size: int) -> int:
        c = self.cfg
        return 2 * c.num_layers * block_size * self.local_kv_heads * c.head_dim * torch.finfo(c.dtype).bits // 8

    def allocate_kv_caches(self, num_blocks: int, block_size: int, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        c = self.cfg
        shape = (num_blocks, block_size, self.local_kv_heads, c.head_dim)
        return [
            (torch.zeros(shape, dtype=c.dtype, device=device), torch.zeros(shape, dtype=c.dtype, device=device))
            for _ in range(c.num_layers)
        ]

    # -- weights ------------------------------------------------------------------
    @classmethod
    def from_hf(cls, path: str, dtype: torch.dtype = torch.float16, device: str = "cpu") -> "LlamaForCausalLM":
        """Load a HuggingFace Llama/Mistral/Qwen2-style checkpoint (safetensors)."""
        from .loader import load_hf_llama

        return load_hf_llama(cls, path, dtype=dtype, device=device)


def reference_forward_full(model: LlamaForCausalLM, token_ids: list[int]) -> torch.Tensor:
    """Un-paged, un-batched forward over a whole sequence (for tests): logits [T, vocab].

    Uses a temporary cache large enough for the sequence and a single
    'prefill everything' call; this is the ground truth that batched /
    chunked / cached execution must reproduce.
    """
    dev = next(model.parameters()).device
    t = len(token_ids)
    bs = 16
    nblk = (t + bs - 1) // bs + 1
    caches = model.allocate_kv_caches(nblk, bs, dev)
    ids = torch.tensor(token_ids, device=dev)
    meta = AttnMetadata(
        positions=torch.arange(t, device=dev),
        slot_mapping=torch.arange(t, device=dev),
        block_tables=torch.arange(nblk, dtype=torch.int32, device=dev)[None],
        cu_seqlens_q=torch.tensor([0, t], dtype=torch.int32, device=dev),
        context_lens=torch.tensor([t], dtype=torch.int32, device=dev),
        max_q_len=t,
        logits_indices=torch.arange(t, device=dev),
    )
    with torch.no_grad():
        return model(ids, caches, meta)
