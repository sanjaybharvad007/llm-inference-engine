"""CUDA C++ paged attention: host-stub type check on CPU, numerics on a GPU."""
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from minivllm.kernels.paged_attention import paged_attention_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_kernel_type_checks_under_host_stub():
    """Every template instantiation compiles with a plain C++ compiler (catches
    indexing/type errors in the kernel without a GPU)."""
    cmd = ["g++", "-fsyntax-only", "-std=c++17", "-Wall", "-Werror", "-Wno-unknown-pragmas", "-x", "c++",
           "-Icsrc/host_stub_include", "-include", "csrc/cuda_host_stub.h", "csrc/syntax_check.cpp"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _case(dtype, head_dim, block_size, q_lens, ctx_lens, h_q=8, h_kv=2, num_blocks=64, dev="cuda"):
    torch.manual_seed(0)
    k = torch.randn(num_blocks, block_size, h_kv, head_dim, device=dev, dtype=dtype)
    v = torch.randn(num_blocks, block_size, h_kv, head_dim, device=dev, dtype=dtype)
    q = torch.randn(sum(q_lens), h_q, head_dim, device=dev, dtype=dtype)
    max_blocks = max((c + block_size - 1) // block_size for c in ctx_lens)
    perm = torch.randperm(num_blocks - 1, device=dev) + 1
    bt = torch.zeros(len(q_lens), max_blocks, dtype=torch.int32, device=dev)
    used = 0
    for i, c in enumerate(ctx_lens):
        n = (c + block_size - 1) // block_size
        bt[i, :n] = perm[used:used + n].int()
        used += n
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev)
    ctx = torch.tensor(ctx_lens, dtype=torch.int32, device=dev)
    return q, k, v, bt, cu, ctx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_dim,block_size", [(64, 8), (128, 16), (128, 32)])
def test_cuda_kernel_matches_reference(dtype, head_dim, block_size):
    from minivllm.kernels.cuda_ext import available
    from minivllm.kernels.paged_attention import paged_attention_cuda

    if not available():
        pytest.skip("CUDA extension not built (MINIVLLM_JIT=1 to build on the fly)")
    q, k, v, bt, cu, ctx = _case(dtype, head_dim, block_size, q_lens=[1, 5, 1, 3], ctx_lens=[1, 37, 1500, 2100])
    out = paged_attention_cuda(q, k, v, bt, cu, ctx)
    ref = paged_attention_reference(q, k, v, bt, cu, ctx)
    tol = 2e-2 if dtype != torch.float32 else 1e-4
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=tol), (out.float() - ref.float()).abs().max()
