"""Triton W8A16 GEMM: fp16/bf16 activations x int8/fp8 weights with the
dequantization fused into the K loop.

    y[M, N] = x[M, K] @ (w_q[N, K] * scale[N])^T

Each program computes a BLOCK_M x BLOCK_N tile.  Along K it loads an fp16 x
tile and an 8-bit w tile; the w tile is up-converted in registers, so global
traffic for the weight is 1 byte/element — this is the whole point for
decode, where M is tiny and the kernel is purely weight-bandwidth bound.
The per-row scale is applied once to the accumulator after the K loop
(mathematically identical, far cheaper than scaling every tile).
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

if HAS_TRITON:

    @triton.jit
    def _w8a16_gemm_kernel(
        x_ptr, w_ptr, s_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + rk
            x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :] * stride_xk,
                        mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
            w = tl.load(w_ptr + rn[None, :] * stride_wn + kk[:, None] * stride_wk,
                        mask=(rn[None, :] < N) & (kk[:, None] < K), other=0)        # [BLOCK_K, BLOCK_N] 8-bit
            acc += tl.dot(x, w.to(x.dtype))                                         # dequant in registers
        scale = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
        acc = acc * scale[None, :]
        tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, acc.to(y_ptr.dtype.element_ty),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


def w8a16_gemm(x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: [..., K] fp16/bf16; wq: [N, K] int8/fp8; scale: [N] fp32 -> [..., N] in x.dtype."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x2.shape
    N = wq.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    bm, bn, bk, nw, ns = _config(M, N, K)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    _w8a16_gemm_kernel[grid](
        x2, wq, scale, y, M, N, K,
        x2.stride(0), x2.stride(1), wq.stride(0), wq.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns,
    )
    return y.reshape(*lead, N)


def _config(M: int, N: int, K: int) -> tuple[int, int, int, int, int]:
    """Tile selection (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages).

    Decode (M <= 16) wants the widest N tile per program with a long K loop so
    each weight byte is loaded once and amortised over the few rows; prefill
    (large M) wants square tiles for reuse in both dimensions."""
    if INTERPRET:
        return 16, 32, 32, 4, 1
    if M <= 16:
        return 16, 128, 128, 4, 4
    if M <= 64:
        return 64, 128, 64, 8, 3
    return 128, 128, 32, 8, 3
