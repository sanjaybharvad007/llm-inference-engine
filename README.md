# llm-inference-engine (minivllm)

An LLM inference and serving engine built from the ground up around the ideas
in vLLM / PagedAttention, small enough to read in an afternoon and tested
end-to-end against an un-batched reference so every optimisation is provably
output-preserving.

```bash
python -m minivllm serve --model /path/to/Llama-3-8B-Instruct          # OpenAI-compatible API on :8000
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
| **Paged attention kernel** | `kernels/paged_attention.py` | one Triton kernel for prefill chunks *and* decode: online softmax, block-table gather per tile, GQA, causal; torch reference for CPU |
| **KV write kernel** | `kernels/cache_ops.py` | `reshape_and_cache`-style scatter into the paged cache |
| **CUDA graphs** | `runner/model_runner.py` | decode steps captured per batch-size bucket with static buffers; block tables / context lengths are inputs so no re-capture as sequences grow |
| **Speculative decoding** | `core/speculative.py` | n-gram / prompt-lookup drafts verified in one forward pass; greedy acceptance is bit-identical to plain greedy (tested), sampling uses rejection sampling |
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
pytest -q                                           # 43 tests
TRITON_INTERPRET=1 pytest -q tests/test_kernels.py tests/test_engine.py -k triton
```

## Benchmarks

```bash
# offline throughput + feature ablation (cuda graphs / prefix cache / chunked prefill / spec decode)
python benchmarks/bench_offline.py --model /path/to/llama --num-prompts 256 --input-len 512 --output-len 128 \
    --shared-prefix-len 256 --max-num-seqs 1 8 32 128 --ablate

# online: Poisson arrivals against the server, TTFT / TPOT / E2E percentiles
python benchmarks/bench_serving.py --url http://localhost:8000 --rate 8 --num-requests 500
```

Fill in your GPU numbers here:

| config (Llama-3-8B, A100/H100) | max_seqs | req/s | gen tok/s | mean TTFT | p99 TTFT |
|---|---|---|---|---|---|
| default | 128 | | | | |
| no CUDA graphs | 128 | | | | |
| no prefix cache (shared 256-tok prefix) | 128 | | | | |
| n-gram spec decode (k=4) | 32 | | | | |

CPU tiny-model ablation (what CI runs; relative numbers only):
speculative n-gram decoding gave +15% tokens/s at 32 concurrent sequences on a
repetitive workload, prefix caching cut mean TTFT 28% with a 50%-shared prefix.

## Layout

```
minivllm/config.py            EngineConfig / ModelConfig / SamplingParams
minivllm/core/                sequence.py  block_manager.py  scheduler.py  sampler.py  speculative.py  engine.py
minivllm/kernels/             paged_attention.py (Triton + reference)  cache_ops.py
minivllm/models/              llama.py (paged-cache Llama)  loader.py (HF safetensors)
minivllm/runner/              model_runner.py (input flattening, CUDA graphs)
minivllm/server/              api.py (OpenAI API, SSE)  async_engine.py  metrics.py (Prometheus)
benchmarks/  deploy/  tests/
```
