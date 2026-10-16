# llm-inference-engine (minivllm)

An LLM inference and serving engine built from the ground up around the ideas
in vLLM / PagedAttention, small enough to read in an afternoon and tested
end-to-end against an un-batched reference so every optimisation is provably
output-preserving.

```bash
python -m minivllm serve --model /path/to/Llama-3-8B-Instruct          # OpenAI-compatible API on :8000
python -m minivllm serve --model /path/to/Llama-3-70B -tp 4 \
    --quantization int8 --draft-model /path/to/Llama-3.2-1B            # 4-GPU tensor parallel, W8A16, spec decoding
    # (launch TP with: torchrun --nproc_per_node=4 -m minivllm serve ... -tp 4)
python -m minivllm serve --tiny                                         # random tiny model, CPU, for a smoke test
curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

```python
from minivllm import LLMEngine, SamplingParams
engine = LLMEngine.from_pretrained("/path/to/llama")
print(engine.generate_text(["Explain KV caching in one sentence."], SamplingParams(max_tokens=64, temperature=0.0)))
```

## What's implemented

| Feature | Where | Notes |
|---|---|---|
| **Paged KV cache** | `core/block_manager.py` | fixed-size blocks, per-sequence block tables, ref-counted, no fragmentation beyond the last block |
| **Automatic prefix caching** | `core/block_manager.py` | chained content hashes over full blocks; shared prompts reuse KV; LRU eviction of unreferenced cached blocks; only committed once the KV is actually computed |
| **Continuous batching** | `core/scheduler.py` | requests join/leave the batch every step; decode tokens scheduled before prefill |
| **Chunked prefill** | `core/scheduler.py` | long prompts split across steps under a token budget so decode latency stays flat |
| **Preemption (recompute)** | `core/scheduler.py` | lowest-priority sequence evicted under memory pressure, re-queued at the front, re-prefills (hitting its own cached blocks) |
| **Paged attention kernels** | `kernels/paged_attention.py`, `csrc/` | **CUDA C++** decode kernel (vLLM-v1 style: one block per token/head, chunked online softmax, vectorised K loads, coalesced V) + a **Triton** FlashAttention-style tiled kernel for prefill chunks; torch reference for CPU; one dispatch, `MINIVLLM_ATTN_BACKEND` to force a backend |
| **Tensor parallelism** | `distributed/` | Megatron-style Column/Row/Vocab-parallel linears (2 all-reduces per layer), NCCL on GPU / gloo on CPU, driver/worker step protocol; a 2-rank gloo test asserts output identical to TP=1 |
| **Weight-only quantization** | `quant/` | per-channel INT8 and FP8-E4M3 weights, fp16 activations; Triton W8A16 GEMM with dequant fused into the K loop (tested under the interpreter); commutes with TP sharding |
| **Draft-model speculative decoding** | `core/draft_model.py` | small model proposes k tokens on its own paged cache (prefix-matched, no rollback bookkeeping), target verifies in one pass; greedy output identical to plain decoding incl. under preemption (tested) |
| **KV write kernel** | `kernels/cache_ops.py` | `reshape_and_cache`-style scatter into the paged cache |
| **CUDA graphs** | `runner/model_runner.py` | decode steps captured per batch-size bucket with static buffers; block tables / context lengths are inputs so no re-capture as sequences grow |
| **N-gram speculative decoding** | `core/speculative.py` | prompt-lookup drafts verified in one forward pass; greedy acceptance is bit-identical to plain greedy (tested), sampling uses rejection sampling |
| **Sampling** | `core/sampler.py` | temperature, top-k, top-p, repetition penalty, stop tokens, seeded |
| **Llama model** | `models/llama.py` | RMSNorm, RoPE, GQA, SwiGLU on the paged cache; loads HF safetensors (Llama / Mistral / Qwen2 layouts) |
| **OpenAI API** | `server/api.py` | `/v1/completions`, `/v1/chat/completions` (SSE streaming), `/v1/models`, `/health`; client disconnect aborts the request |
| **Observability** | `server/metrics.py` | Prometheus: TTFT / TPOT / E2E histograms, running/waiting, KV usage, prefix hit rate, preemptions, spec acceptance |
| **Deploy** | `deploy/` | Dockerfile, k8s Deployment + Service + HPA on queue depth |

## How a step works

```
scheduler.schedule()            decode seqs (1 slot each, drafts +k) → prefill chunks → admit waiting (prefix-cache aware)
      │                         preempt lowest-priority running seq if a block can't be found
      ▼
runner.build_inputs()           flatten all new tokens: ids, positions, slot_mapping (physical cache slots),
      │                         cu_seqlens_q, context_lens, padded block_tables, logits_indices
      ▼
model(...)                      per layer: write new K/V into the paged cache → paged attention over
      │                         [cached context + this step's tokens] through the block table
      │                         (pure-decode batches replay a CUDA graph instead)
      ▼
sample / verify                 one token per finished-prefill seq, or accept-longest-prefix over drafts
      │                         update num_computed_tokens, commit full blocks to the prefix cache,
      ▼                         finish on EOS / stop / max_tokens, free blocks
RequestOutput per seq           streamed to clients by AsyncLLMEngine
```

The whole engine has a single notion of "tokens that still need their KV
computed", so prompt prefill, chunked prefill, recompute after preemption, and
speculative verification are all the same code path with different chunk sizes.

## Correctness tests

`tests/test_engine.py` generates with the paged / batched / chunked / cached /
preempting / speculative engine and asserts **token-for-token equality** with a
naive loop that re-runs a dense, un-paged forward over the full sequence for
every token. CPU-only CI runs all of it with a random tiny Llama; the Triton
kernels run on the same tests under `TRITON_INTERPRET=1`.

```bash
pip install -e ".[dev]"
pytest -q                                           # engine, TP (2 gloo ranks), quant, draft model, server, kernels
TRITON_INTERPRET=1 pytest -q tests/test_kernels.py tests/test_quant.py tests/test_engine.py -k "triton or w8a16"
MINIVLLM_JIT=1 pytest -q tests/test_cuda_kernel.py  # on a GPU: builds csrc/ on the fly and checks it against the reference
```

The CUDA kernel is also type-checked in CI without a GPU: every template
instantiation is compiled with `g++` under `csrc/cuda_host_stub.h`.

## Benchmarks

```bash
# offline throughput + feature ablation (cuda graphs / prefix cache / chunked prefill / spec decode)
python benchmarks/bench_offline.py --model /path/to/llama --num-prompts 256 --input-len 512 --output-len 128 \
    --shared-prefix-len 256 --max-num-seqs 1 8 32 128 --ablate

# online: Poisson arrivals against the server, TTFT / TPOT / E2E percentiles
python benchmarks/bench_serving.py --url http://localhost:8000 --rate 8 --num-requests 500

# decode micro-benchmarks: CUDA vs Triton paged attention (GB/s), W8A16 vs fp16 GEMM, plain vs n-gram vs draft model
python benchmarks/bench_decode.py attn --ctx 4096 --batch 1 8 32
python benchmarks/bench_decode.py gemm --in 4096 --out 11008 --m 1 4 16
python benchmarks/bench_decode.py spec --model /path/to/7b --draft-model /path/to/160m
```

Fill in your GPU numbers here:

| config (Llama-3-8B, A100/H100) | max_seqs | req/s | gen tok/s | mean TTFT | p99 TTFT |
|---|---|---|---|---|---|
| default | 128 | | | | |
| no CUDA graphs | 128 | | | | |
| no prefix cache (shared 256-tok prefix) | 128 | | | | |
| n-gram spec decode (k=4) | 32 | | | | |
| draft-model spec decode (1B draft, k=4) | 32 | | | | |
| int8 weight-only | 128 | | | | |
| TP=2 (70B) | 128 | | | | |

CPU tiny-model ablation (what CI runs; relative numbers only):
speculative n-gram decoding gave +15% tokens/s at 32 concurrent sequences on a
repetitive workload, prefix caching cut mean TTFT 28% with a 50%-shared prefix.

## Layout

```
minivllm/config.py            EngineConfig / ModelConfig / SamplingParams
minivllm/core/                sequence.py  block_manager.py  scheduler.py  sampler.py  speculative.py  draft_model.py  engine.py
minivllm/kernels/             paged_attention.py (dispatch: CUDA / Triton / reference)  cache_ops.py  cuda_ext.py
minivllm/distributed/         parallel_state.py (process group, collectives)  layers.py (Column/Row/Vocab-parallel)
minivllm/quant/               quantize.py (INT8 / FP8 weight-only, QuantLinear)  kernels.py (Triton W8A16 GEMM)
minivllm/models/              llama.py (paged-cache, tensor-parallel Llama)  loader.py (HF safetensors)
minivllm/runner/              model_runner.py (input flattening, CUDA graphs, TP driver/worker loop)
minivllm/server/              api.py (OpenAI API, SSE)  async_engine.py  metrics.py (Prometheus)
csrc/                         paged_attention_kernel.cuh  paged_attention.cu (torch bindings)  cuda_host_stub.h
docs/                         tensor_parallel.md  quantization.md  speculative_decoding.md  cuda_kernel.md
benchmarks/  deploy/  tests/
```
