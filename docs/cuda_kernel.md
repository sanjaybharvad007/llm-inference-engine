# CUDA C++ paged attention (`csrc/`)

`paged_attention_kernel.cuh` is a hand-written decode kernel in the style of
vLLM's v1 kernel, with flash-decoding-style chunking so shared memory does
not grow with the context length.

## Mapping

- grid = `(query tokens, query heads)`, block = `HEAD_DIM` threads
  (64 or 128): **every thread owns one output dimension**.
- Keys stream through the block table in chunks of `CHUNK = 1024` positions:
  1. **QK**: thread `t` scores keys `t, t+NT, ...` of the chunk. `q` sits in
     shared memory as fp32; the key row is read with 16-byte vector loads
     (`uint4` = 8 halves) and the dot product is unrolled over `HEAD_DIM`.
  2. **Softmax**: chunk max and sum via warp shuffles + one shared-memory
     round; the running `(m, l)` and the accumulator are rescaled by
     `exp(m_old - m_new)` exactly as in FlashAttention, so the result is
     independent of the chunking.
  3. **PV**: every thread walks the chunk's value rows and accumulates its own
     dimension — value rows are read fully coalesced, exactly once.
- Multi-token queries (speculative verification, tiny prefill chunks) get one
  block per token with the causal bound `ctx_len - q_len + i + 1`; prefill
  chunks longer than `CUDA_MAX_Q_LEN = 8` go to the Triton kernel, which
  reuses K/V tiles across query rows.
- GQA: `h_kv = h / (H_q / H_kv)`; dtypes fp16 / bf16 / fp32; block sizes
  8 / 16 / 32.

## Build

```bash
pip install -e .                 # builds minivllm._C when nvcc is on PATH
MINIVLLM_JIT=1 python ...        # or compile on first use with torch.utils.cpp_extension.load
MINIVLLM_ATTN_BACKEND=cuda|triton|reference   # force a backend
```

Without a GPU the kernel is still type-checked: `csrc/syntax_check.cpp`
instantiates every `(dtype, head_dim, block_size)` variant under
`csrc/cuda_host_stub.h` with plain `g++ -fsyntax-only` (run by
`tests/test_cuda_kernel.py` and CI).

## Where the time goes / next steps

The kernel reads each K and V byte once, so at decode it is bound by HBM
bandwidth like every paged-attention kernel; `benchmarks/bench_decode.py attn`
prints achieved GB/s next to the Triton kernel. The obvious next steps,
in order of payoff: split-K across thread blocks for long contexts at small
batch (more SMs busy), `cp.async` double-buffering of K/V tiles, and packing
two query heads of the same KV head into one block (GQA reuse).
