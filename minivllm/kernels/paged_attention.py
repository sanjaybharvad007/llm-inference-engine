"""Paged attention (PagedAttention-style) over a block-table KV cache.

One kernel serves both prefill chunks and decode steps: every sequence in the
batch contributes ``q_len`` query tokens (1 for decode, up to the chunk size for
prefill) that attend to *all* of its context — the tokens already in the cache
plus the ones written this step — through its block table.  This is what makes
chunked prefill, prefix caching and speculative verification all look the same
to the model.

Cache layout: ``key_cache / value_cache: [num_blocks, block_size, num_kv_heads, head_dim]``.

The Triton kernel is a FlashAttention-2 style online-softmax loop whose K/V
tile addresses go through the block table (a gather per tile row).  The torch
reference implementation is used on CPU and in tests.
"""
from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


# ----------------------------------------------------------------------------
# Reference
# ----------------------------------------------------------------------------


def gather_kv(cache: torch.Tensor, block_table: torch.Tensor, ctx_len: int, block_size: int) -> torch.Tensor:
    """[ctx_len, H_kv, D] for one sequence from the paged cache."""
    nblk = (ctx_len + block_size - 1) // block_size
    blocks = cache[block_table[:nblk].long()]            # [nblk, block_size, H, D]
    return blocks.reshape(-1, cache.shape[2], cache.shape[3])[:ctx_len]


def paged_attention_reference(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    total_q, h_q, d = q.shape
    h_kv = key_cache.shape[2]
    block_size = key_cache.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    out = torch.empty_like(q)
    for s in range(cu_seqlens_q.numel() - 1):
        qs, qe = int(cu_seqlens_q[s]), int(cu_seqlens_q[s + 1])
        if qe == qs:
            continue
        ctx = int(context_lens[s])
        k = gather_kv(key_cache, block_tables[s], ctx, block_size).float()
        v = gather_kv(value_cache, block_tables[s], ctx, block_size).float()
        qq = q[qs:qe].float()                                # [q_len, H_q, D]
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qq, k) * scale
        q_len = qe - qs
        pos = torch.arange(ctx - q_len, ctx, device=q.device)[:, None]
        kpos = torch.arange(ctx, device=q.device)[None, :]
        scores = scores.masked_fill((kpos > pos)[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[qs:qe] = torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)
    return out


# ----------------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _paged_attn_kernel(
        Q, O, K_CACHE, V_CACHE, BLOCK_TABLES, CU_SEQLENS_Q, CONTEXT_LENS,
        stride_qt, stride_qh,
        stride_ot, stride_oh,
        stride_kb, stride_kt, stride_kh,
        stride_bt,
        H_Q, H_KV, sm_scale,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_sh = tl.program_id(1)
        s = off_sh // H_Q
        h = off_sh % H_Q
        h_kv = h // (H_Q // H_KV)

        q_start = tl.load(CU_SEQLENS_Q + s)
        q_end = tl.load(CU_SEQLENS_Q + s + 1)
        q_len = q_end - q_start
        ctx_len = tl.load(CONTEXT_LENS + s)
        if start_m * BLOCK_M >= q_len:
            return

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q_ptrs = Q + (q_start + offs_m)[:, None] * stride_qt + h * stride_qh + offs_d[None, :]
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)
        q_pos = ctx_len - q_len + offs_m            # absolute position of each query row

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
        qk_scale = sm_scale * 1.4426950408889634

        hi = tl.minimum(ctx_len, ctx_len - q_len + (start_m + 1) * BLOCK_M)   # causal upper bound
        for start_n in range(0, hi, BLOCK_N):
            kv_pos = start_n + offs_n
            valid_n = kv_pos < ctx_len
            # block-table gather: physical block for every key position in the tile
            phys = tl.load(BLOCK_TABLES + s * stride_bt + kv_pos // BLOCK_SIZE, mask=valid_n, other=0)
            kv_off = phys * stride_kb + (kv_pos % BLOCK_SIZE) * stride_kt + h_kv * stride_kh
            k = tl.load(K_CACHE + kv_off[:, None] + offs_d[None, :], mask=valid_n[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            mask = valid_n[None, :] & (kv_pos[None, :] <= q_pos[:, None])
            qk = tl.where(mask, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.math.exp2(m_i - m_safe)
            p = tl.math.exp2(qk - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(V_CACHE + kv_off[:, None] + offs_d[None, :], mask=valid_n[:, None], other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        out = acc / l_safe[:, None]
        o_ptrs = O + (q_start + offs_m)[:, None] * stride_ot + h * stride_oh + offs_d[None, :]
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_m[:, None] < q_len)


def paged_attention_triton(q, key_cache, value_cache, block_tables, cu_seqlens_q, context_lens, max_q_len, scale=None):
    total_q, h_q, d = q.shape
    h_kv = key_cache.shape[2]
    block_size = key_cache.shape[1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    o = torch.empty_like(q)
    num_seqs = cu_seqlens_q.numel() - 1
    block_m = 16 if max_q_len <= 16 else 64
    block_n = max(16, block_size) if block_size <= 64 else 64
    interp = os.environ.get("TRITON_INTERPRET", "0") == "1"
    grid = (triton.cdiv(max_q_len, block_m), num_seqs * h_q)
    _paged_attn_kernel[grid](
        q, o, key_cache, value_cache, block_tables, cu_seqlens_q, context_lens,
        q.stride(0), q.stride(1), o.stride(0), o.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        block_tables.stride(0),
        h_q, h_kv, scale,
        BLOCK_SIZE=block_size, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=d,
        num_warps=4 if not interp else 1,
    )
    return o


def paged_attention(q, key_cache, value_cache, block_tables, cu_seqlens_q, context_lens, max_q_len, scale=None):
    """Dispatch: Triton on CUDA (or under TRITON_INTERPRET=1), torch reference otherwise."""
    use_triton = HAS_TRITON and (q.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")
    if use_triton and os.environ.get("MINIVLLM_ATTN_BACKEND", "auto") != "reference":
        return paged_attention_triton(q, key_cache, value_cache, block_tables, cu_seqlens_q, context_lens, max_q_len, scale)
    return paged_attention_reference(q, key_cache, value_cache, block_tables, cu_seqlens_q, context_lens, scale)
