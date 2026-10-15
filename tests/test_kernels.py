"""Paged attention kernels vs. a dense reference built by gathering the cache."""
import math
import os

import pytest
import torch

from minivllm.kernels.cache_ops import write_kv, write_kv_reference
from minivllm.kernels.paged_attention import HAS_TRITON, paged_attention_reference, paged_attention_triton

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
DEV = "cuda" if CUDA else "cpu"


def _dense_reference(q, kc, vc, bt, cu, ctx):
    """Independent dense re-implementation (no block gather helper reuse)."""
    h_q, d = q.shape[1], q.shape[2]
    h_kv, bs = kc.shape[2], kc.shape[1]
    out = torch.empty_like(q)
    for s in range(len(ctx)):
        qs, qe = int(cu[s]), int(cu[s + 1])
        n = int(ctx[s])
        k = torch.cat([kc[int(bt[s, i])] for i in range((n + bs - 1) // bs)])[:n].float()
        v = torch.cat([vc[int(bt[s, i])] for i in range((n + bs - 1) // bs)])[:n].float()
        for h in range(h_q):
            kh, vh = k[:, h // (h_q // h_kv)], v[:, h // (h_q // h_kv)]
            qq = q[qs:qe, h].float()
            sc = qq @ kh.T / math.sqrt(d)
            qpos = torch.arange(n - (qe - qs), n)[:, None]
            sc = sc.masked_fill(torch.arange(n)[None, :] > qpos, float("-inf"))
            out[qs:qe, h] = (torch.softmax(sc, -1) @ vh).to(q.dtype)
    return out


@pytest.mark.parametrize("q_lens,ctx_lens", [([1, 1, 1], [5, 40, 17]), ([5, 1], [37, 20]), ([16, 3], [16, 50]), ([33], [33])])
def test_reference_matches_dense(q_lens, ctx_lens):
    torch.manual_seed(0)
    nb, bs, hkv, hq, d = 64, 16, 2, 4, 32
    kc, vc = torch.randn(nb, bs, hkv, d), torch.randn(nb, bs, hkv, d)
    maxb = max((c + bs - 1) // bs for c in ctx_lens)
    bt = torch.randperm(nb - 1)[: len(ctx_lens) * maxb].reshape(len(ctx_lens), maxb).to(torch.int32) + 1
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32)
    q = torch.randn(int(cu[-1]), hq, d)
    ref = paged_attention_reference(q, kc, vc, bt, cu, torch.tensor(ctx_lens, dtype=torch.int32))
    torch.testing.assert_close(ref, _dense_reference(q, kc, vc, bt, cu, ctx_lens), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not HAS_TRITON or not (CUDA or INTERP), reason="needs GPU or TRITON_INTERPRET=1")
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("q_lens,ctx_lens", [([1, 1, 1], [5, 40, 17]), ([5, 1], [37, 20]), ([16, 3], [16, 50])])
def test_triton_matches_reference(q_lens, ctx_lens, block_size):
    torch.manual_seed(0)
    nb, hkv, hq, d = 32, 2, 8, 64
    dtype = torch.float16 if CUDA else torch.float32
    kc, vc = torch.randn(nb, block_size, hkv, d, device=DEV, dtype=dtype), torch.randn(nb, block_size, hkv, d, device=DEV, dtype=dtype)
    maxb = max((c + block_size - 1) // block_size for c in ctx_lens)
    bt = (torch.randperm(nb - 1)[: len(ctx_lens) * maxb].reshape(len(ctx_lens), maxb) + 1).to(torch.int32).to(DEV)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)), dtype=torch.int32, device=DEV)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=DEV)
    q = torch.randn(int(cu[-1]), hq, d, device=DEV, dtype=dtype)
    out = paged_attention_triton(q, kc, vc, bt, cu, ctx, max(q_lens))
    ref = paged_attention_reference(q, kc, vc, bt, cu, ctx)
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)


def test_write_kv_reference_and_triton_agree():
    torch.manual_seed(0)
    kc, vc = torch.zeros(8, 16, 2, 32, device=DEV), torch.zeros(8, 16, 2, 32, device=DEV)
    kc2, vc2 = kc.clone(), vc.clone()
    k, v = torch.randn(5, 2, 32, device=DEV), torch.randn(5, 2, 32, device=DEV)
    slots = torch.tensor([0, 17, 33, 100, 127], device=DEV)
    write_kv_reference(k, v, kc, vc, slots)
    write_kv(k, v, kc2, vc2, slots)
    assert torch.equal(kc, kc2) and torch.equal(vc, vc2)
    assert torch.equal(kc.view(-1, 2, 32)[17], k[1])
