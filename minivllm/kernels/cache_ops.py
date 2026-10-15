"""Writing new K/V vectors into the paged cache (``reshape_and_cache``)."""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


def write_kv_reference(key, value, key_cache, value_cache, slot_mapping):
    """key/value: [T, H_kv, D]; slot_mapping: int64 [T] physical slots."""
    h, d = key_cache.shape[2], key_cache.shape[3]
    key_cache.view(-1, h, d)[slot_mapping] = key.to(key_cache.dtype)
    value_cache.view(-1, h, d)[slot_mapping] = value.to(value_cache.dtype)


if HAS_TRITON:

    @triton.jit
    def _write_kv_kernel(K, V, K_CACHE, V_CACHE, SLOTS, stride_kt, HD: tl.constexpr, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        slot = tl.load(SLOTS + t)
        offs = tl.arange(0, BLOCK)
        m = offs < HD
        k = tl.load(K + t * stride_kt + offs, mask=m, other=0.0)
        v = tl.load(V + t * stride_kt + offs, mask=m, other=0.0)
        tl.store(K_CACHE + slot * HD + offs, k.to(K_CACHE.dtype.element_ty), mask=m)
        tl.store(V_CACHE + slot * HD + offs, v.to(V_CACHE.dtype.element_ty), mask=m)


def write_kv(key, value, key_cache, value_cache, slot_mapping):
    use_triton = HAS_TRITON and (key.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")
    if not use_triton or key.shape[0] == 0:
        return write_kv_reference(key, value, key_cache, value_cache, slot_mapping)
    key = key.contiguous()
    value = value.contiguous()
    hd = key.shape[1] * key.shape[2]
    block = max(16, 1 << (hd - 1).bit_length())
    _write_kv_kernel[(key.shape[0],)](
        key, value, key_cache, value_cache, slot_mapping.to(torch.int32), key.stride(0), HD=hd, BLOCK=block
    )
