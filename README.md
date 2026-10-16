# llm-inference-engine (minivllm)

**An LLM inference and serving engine built around PagedAttention: paged KV cache with prefix caching, continuous batching with chunked prefill, CUDA graphs, CUDA C++ and Triton paged-attention kernels, tensor parallelism, INT8/FP8 weight-only quantization, n-gram and draft-model speculative decoding, and an OpenAI-compatible server.**

[![CI](https://github.com/sanjaybharvad007/llm-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjaybharvad007/llm-inference-engine/actions/workflows/ci.yml)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)
![Triton](https://img.shields.io/badge/Triton-3.x-blue)
![NCCL](https://img.shields.io/badge/Tensor%20Parallel-NCCL%20%7C%20gloo-informational)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```bash
python -m minivllm serve --model /path/to/Llama-3-8B-Instruct                  # OpenAI-compatible API on :8000
torchrun --nproc_per_node=4 -m minivllm serve --model /path/to/Llama-3-70B -tp 4 \
    --quantization int8 --draft-model /path/to/Llama-3.2-1B                    # 4-GPU TP, W8A16, speculative decoding
curl localhost:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
```

```python
from minivllm import LLMEngine, SamplingParams

engine = LLMEngine.from_pretrained("/path/to/llama")
print(engine.generate_text(["Explain KV caching in one sentence."], SamplingParams(max_tokens=64, temperature=0.0)))
```

Every optimisation in the engine is held to one standard: the output must be **token-for-token identical** to a naive, un-batched, un-paged forward pass over the full sequence. The test-suite checks that for batching, chunked prefill, prefix caching, preemption, CUDA graphs, tensor parallelism and both speculative decoders.

---

## Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Design](#design)
- [Performance](#performance)
- [Testing](#testing)
- [Deployment](#deployment)
- [Repository layout](#repository-layout)
- [Roadmap](#roadmap)
- [References](#references)

---

## Highlights

- **Paged KV cache** with fixed-size blocks, per-sequence block tables and reference counting; **automatic prefix caching** by chained content hashes with LRU eviction.
- **Continuous batching** with **chunked prefill** under a token budget and **recompute preemption** under memory pressure.
- **Two paged-attention kernels**: a CUDA C++ decode kernel (one block per token and head, chunked online softmax, vectorised K loads, coalesced V reads) and a Triton FlashAttention-style tiled kernel for prefill; one dispatch, one torch reference.
- **CUDA graphs** per batch-size bucket with static buffers; block tables and context lengths are inputs, so sequences grow without re-capture.
- **Tensor parallelism**: Megatron-style column/row/vocab-parallel layers, two all-reduces per layer, NCCL on GPU and gloo on CPU, driver/worker step protocol; a 2-rank test asserts identical output to TP=1.
- **Weight-only INT8 and FP8-E4M3 quantization** with a Triton W8A16 GEMM whose dequantisation is fused into the K loop.
- **Speculative decoding** two ways: free n-gram (prompt-lookup) drafts, and a draft model running on its own paged cache with prefix-matched state; both verified in a single target forward pass with greedy output identical to plain decoding.
- **Serving**: FastAPI OpenAI-compatible API with SSE streaming, request abort on disconnect, Prometheus metrics, Dockerfile and Kubernetes manifests with an HPA on queue depth.
- **69 tests**, all on CPU in CI, including a real 2-process tensor-parallel run and the Triton kernels under the interpreter.

## Architecture

```
                 HTTP (OpenAI API, SSE)                 Python API
                          |                                 |
                   AsyncLLMEngine  ──────────────────► LLMEngine.step()
                                                             |
       +-----------------------------------------------------+------------------------------------------+
       |                                                                                                |
  Scheduler                      BlockManager                       ModelRunner                     Sampler / verify
  decode seqs (+k drafts)        paged blocks, block tables         flatten batch -> payload         temperature, top-k/p,
  -> prefill chunks              chained-hash prefix cache          [TP: broadcast to workers]       repetition penalty,
  -> admit waiting               LRU eviction, refcounts            CUDA graph replay | forward      greedy / rejection
  preempt (recompute)            commit full blocks                 logits for sampled rows          sampling over drafts
                                                                        |
                                                   LlamaForCausalLM (tensor-parallel)
                                                   per layer: write K/V into paged cache
                                                              paged attention over block table
                                                              CUDA C++ kernel | Triton kernel | reference
                                                   Column/Row/Vocab-parallel linears or QuantLinear (W8A16)
                                                                        |
                                                   DraftModelProposer (driver-local, own paged cache)
```

A single notion, *tokens that still need their KV computed*, drives everything: prompt prefill, chunked prefill, recompute after preemption and speculative verification are the same code path with different chunk sizes.

## Features

| Feature | Where | Notes |
|---|---|---|
| Paged KV cache | `core/block_manager.py` | fixed-size blocks, per-sequence block tables, reference counted; block 0 reserved as the null block |
| Automatic prefix caching | `core/block_manager.py` | chained content hashes over full blocks; shared prompts reuse KV; LRU eviction of unreferenced blocks; committed only once the KV is computed |
| Continuous batching | `core/scheduler.py` | requests join and leave every step; decode tokens scheduled before prefill |
| Chunked prefill | `core/scheduler.py` | long prompts split across steps under `max_num_batched_tokens` so decode latency stays flat |
| Preemption (recompute) | `core/scheduler.py` | lowest-priority sequence evicted under memory pressure, re-queued at the front, re-prefills through its own cached blocks |
| Paged attention, CUDA C++ | `csrc/` | decode kernel: block per (token, head), `CHUNK=1024` online softmax, 16-byte K loads, coalesced V; fp16/bf16/fp32; block sizes 8/16/32 |
| Paged attention, Triton | `kernels/paged_attention.py` | FlashAttention-2 style tiles with block-table gather; prefill chunks and decode; CPU under `TRITON_INTERPRET=1` |
| KV write kernel | `kernels/cache_ops.py` | `reshape_and_cache`-style scatter into the paged cache |
| CUDA graphs | `runner/model_runner.py` | decode steps captured per batch bucket with static buffers |
| Tensor parallelism | `distributed/` | `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelLinear`; heads and KV cache sharded; `broadcast_object` driver/worker protocol |
| Weight-only quantization | `quant/` | per-channel INT8 and FP8-E4M3, `QuantLinear` keeps TP semantics, Triton W8A16 GEMM |
| N-gram speculative decoding | `core/speculative.py` | prompt-lookup drafts; greedy acceptance identical to plain greedy; rejection sampling otherwise |
| Draft-model speculative decoding | `core/draft_model.py` | private paged cache, prefix-matched state (no rollback bookkeeping), replicated on the driver rank |
| Sampling | `core/sampler.py` | temperature, top-k, top-p, repetition penalty, stop tokens, seeded |
| Llama model | `models/llama.py` | RMSNorm, RoPE, GQA, SwiGLU on the paged cache; loads HF safetensors (Llama / Mistral / Qwen2 layouts) |
| OpenAI API | `server/api.py` | `/v1/completions`, `/v1/chat/completions` (SSE), `/v1/models`, `/health`; client disconnect aborts the request |
| Observability | `server/metrics.py` | Prometheus: TTFT / TPOT / E2E histograms, running/waiting, KV usage, prefix hit rate, preemptions, speculative acceptance |
| Deployment | `deploy/` | Dockerfile, Kubernetes Deployment + Service + HPA |

## Installation

```bash
git clone https://github.com/sanjaybharvad007/llm-inference-engine
cd llm-inference-engine
pip install -e ".[gpu,hf]"                 # builds the CUDA extension (csrc/) when nvcc is on PATH
MINIVLLM_SKIP_CUDA=1 pip install -e .      # Triton / CPU only
MINIVLLM_JIT=1 python -m minivllm ...      # or compile csrc/ on first use
```

Requirements: PyTorch 2.4+, Triton 3.x on GPU, `safetensors` and `transformers` for HuggingFace checkpoints, and a CUDA 12.x toolkit for the C++ kernel.

## Usage

```bash
# serve
python -m minivllm serve --model /path/to/llama --max-num-seqs 256 --max-num-batched-tokens 4096
python -m minivllm serve --model /path/to/llama --speculative-ngram 4                 # prompt-lookup drafts
python -m minivllm serve --model /path/to/llama --draft-model /path/to/small --num-speculative-tokens 4
python -m minivllm serve --model /path/to/llama --quantization int8                   # or fp8 (sm_89+)
torchrun --nproc_per_node=2 -m minivllm serve --model /path/to/llama -tp 2
python -m minivllm serve --tiny                                                       # random tiny model, CPU

# query
curl localhost:8000/v1/completions -d '{"prompt":"The capital of France is","max_tokens":16,"temperature":0}'
curl localhost:8000/metrics   # Prometheus
```

```python
from minivllm import EngineConfig, LLMEngine, QuantConfig, SamplingParams, quantize_model

cfg = EngineConfig(block_size=16, max_num_seqs=128, enable_prefix_caching=True, speculative_max_draft=4)
engine = LLMEngine.from_pretrained("/path/to/llama", cfg)
quantize_model(engine.model, QuantConfig("int8"))

rid = engine.add_request(None, "Write a haiku about GPUs.", SamplingParams(max_tokens=32, temperature=0.7, top_p=0.9))
while engine.has_unfinished():
    for out in engine.step():
        print(out.request_id, out.new_token_ids, out.finished)
```

Environment: `MINIVLLM_ATTN_BACKEND=auto|cuda|triton|reference`, `MINIVLLM_JIT=1`, `MINIVLLM_SKIP_CUDA=1`, `MINIVLLM_INIT_METHOD` (process-group rendezvous).

## Design

### One step

```
scheduler.schedule()       decode sequences (1 slot each, +k drafts) -> prefill chunks -> admit waiting (prefix-cache aware)
                           preempt the lowest-priority running sequence if a block cannot be found
runner.build_payload()     flatten all new tokens: ids, positions, slot_mapping, cu_seqlens_q, context_lens,
                           padded block_tables, logits_indices          [TP: broadcast to worker ranks]
model(...)                 per layer: write new K/V into the paged cache, then paged attention over
                           cached context + this step's tokens through the block table
                           (pure-decode batches replay a CUDA graph)
sample / verify            one token per finished-prefill sequence, or longest accepted prefix over drafts;
                           update num_computed_tokens, commit full blocks to the prefix cache,
                           finish on EOS / stop / max_tokens, free blocks
RequestOutput per seq      streamed to clients by AsyncLLMEngine
```

### Block manager and prefix cache

Blocks are claimed for cached prefixes *before* any fresh allocation, so eviction triggered by the new allocation can never remove a block the same sequence is about to reuse. Capacity checks count evictable blocks that are about to be hit as unavailable. Requests that could never fit (longer than the model window or the whole cache) are rejected at admission rather than livelocking the scheduler.

### Attention kernels

Decode steps and short verification chunks go to the CUDA C++ kernel (`csrc/paged_attention_kernel.cuh`): one block per (query token, head) with `HEAD_DIM` threads, keys streamed through the block table in 1024-position chunks, chunk max and sum via warp shuffles, running `(m, l)` rescaled online, and each thread accumulating one output dimension so value rows are read coalesced exactly once. Prefill chunks go to the Triton kernel, which reuses K/V tiles across query rows. Details in [`docs/cuda_kernel.md`](docs/cuda_kernel.md).

### Tensor parallelism

Attention heads, FFN width and the vocabulary are sharded Megatron-style; each rank owns a `1/tp` slice of the paged KV cache, and the scheduler and block manager run only on rank 0. Rank 0 broadcasts the flattened step payload; workers execute the same forward (including CUDA-graph replay) and are released by `engine.shutdown()`. Quantization is applied after sharding and commutes with it because scales are per output row. Details in [`docs/tensor_parallel.md`](docs/tensor_parallel.md).

### Quantization

Per-output-channel symmetric scales; `int8` for any hardware, `fp8` (E4M3) where native. The Triton W8A16 GEMM loads an fp16 activation tile and an 8-bit weight tile, up-converts the weights in registers, feeds `tl.dot`, and applies the row scale once to the fp32 accumulator; tiles are chosen by `M` so decode amortises each weight byte over the few active rows. Details in [`docs/quantization.md`](docs/quantization.md).

### Speculative decoding

Drafts are appended to a sequence as a multi-token chunk and scored by the target in one forward pass. The draft model keeps a private paged cache keyed by request id and remembers which token prefix it holds; before proposing, it feeds only the tokens beyond the longest common prefix with the sequence, so rejected drafts, correction tokens and recompute after preemption need no explicit rollback. Under TP the draft model is replicated on the driver rank only. Details in [`docs/speculative_decoding.md`](docs/speculative_decoding.md).

## Performance

```bash
# offline throughput with per-feature ablation
python benchmarks/bench_offline.py --model /path/to/llama --num-prompts 256 --input-len 512 --output-len 128 \
    --shared-prefix-len 256 --max-num-seqs 1 8 32 128 --ablate

# online: Poisson arrivals, TTFT / TPOT / E2E percentiles
python benchmarks/bench_serving.py --url http://localhost:8000 --rate 8 --num-requests 500

# decode micro-benchmarks: paged attention GB/s (CUDA vs Triton), W8A16 vs fp16 GEMM, plain vs n-gram vs draft model
python benchmarks/bench_decode.py attn --ctx 4096 --batch 1 8 32
python benchmarks/bench_decode.py gemm --in 4096 --out 11008 --m 1 4 16
python benchmarks/bench_decode.py spec --model /path/to/7b --draft-model /path/to/160m
```

**Llama-3-8B, A100 / H100** (measure and paste; the harness prints every column)

| configuration | max_seqs | req/s | gen tok/s | mean TTFT | p99 TTFT |
|---|---:|---:|---:|---:|---:|
| default | 128 | | | | |
| no CUDA graphs | 128 | | | | |
| no prefix cache (shared 256-token prefix) | 128 | | | | |
| n-gram speculative decoding (k=4) | 32 | | | | |
| draft-model speculative decoding (1B draft, k=4) | 32 | | | | |
| int8 weight-only | 128 | | | | |
| TP=2 (70B) | 128 | | | | |

**CPU tiny-model ablation** (what CI exercises; relative numbers): n-gram speculative decoding gave +15% tokens/s at 32 concurrent sequences on a repetitive workload; prefix caching cut mean TTFT by 28% with a 50%-shared prefix.

## Testing

```bash
pip install -e ".[dev]"
pytest -q                                            # 69 tests: engine equivalence, scheduler, block manager,
                                                     # 2-rank gloo tensor parallel, quantization, draft model, server
TRITON_INTERPRET=1 pytest -q tests/test_kernels.py tests/test_quant.py tests/test_engine.py -k "triton or w8a16"
MINIVLLM_JIT=1 pytest -q tests/test_cuda_kernel.py   # on a GPU: build csrc/ and compare with the reference
```

`tests/test_engine.py` generates with the paged, batched, chunked, cached, preempting and speculative engine and asserts equality with a naive loop that re-runs a dense forward over the full sequence for every token. `tests/test_tensor_parallel.py` spawns two gloo processes and compares against the single-process engine. The CUDA kernel is type-checked in CI without a GPU: every template instantiation is compiled with `g++` under `csrc/cuda_host_stub.h`.

## Deployment

```bash
docker build -f deploy/Dockerfile -t minivllm .
docker run --gpus all -p 8000:8000 -v /models:/models -e MODEL_PATH=/models/llama minivllm
kubectl apply -f deploy/k8s/                         # Deployment, Service, HPA on queue depth
```

## Repository layout

```
minivllm/config.py          EngineConfig / ModelConfig / SamplingParams
minivllm/core/              sequence.py  block_manager.py  scheduler.py  sampler.py  speculative.py  draft_model.py  engine.py
minivllm/kernels/           paged_attention.py (dispatch: CUDA / Triton / reference)  cache_ops.py  cuda_ext.py
minivllm/distributed/       parallel_state.py (process group, collectives)  layers.py (Column/Row/Vocab-parallel)
minivllm/quant/             quantize.py (INT8 / FP8 weight-only, QuantLinear)  kernels.py (Triton W8A16 GEMM)
minivllm/models/            llama.py (paged-cache, tensor-parallel Llama)  loader.py (HF safetensors)
minivllm/runner/            model_runner.py (input flattening, CUDA graphs, TP driver/worker loop)
minivllm/server/            api.py (OpenAI API, SSE)  async_engine.py  metrics.py (Prometheus)
csrc/                       paged_attention_kernel.cuh  paged_attention.cu (torch bindings)  cuda_host_stub.h  syntax_check.cpp
docs/                       cuda_kernel.md  tensor_parallel.md  quantization.md  speculative_decoding.md
benchmarks/                 bench_offline.py  bench_serving.py  bench_decode.py
deploy/                     Dockerfile  k8s/
tests/                      test_engine.py  test_scheduler.py  test_block_manager.py  test_kernels.py  test_cuda_kernel.py
                            test_tensor_parallel.py  test_quant.py  test_draft_model.py  test_server.py
```

## Roadmap

- Split-K across thread blocks in the CUDA decode kernel for long contexts at small batch
- `cp.async` double buffering and GQA head packing in the CUDA kernel
- Pipeline parallelism across nodes
- Activation quantization (W8A8) with per-token scales
- Structured output (grammar-constrained decoding)

## References

- Kwon, W. et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- Yu, G. et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022 (continuous batching).
- Agrawal, A. et al. *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.* 2023.
- Shoeybi, M. et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019.
- Leviathan, Y. et al. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen, C. et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
- Saxena, A. *Prompt Lookup Decoding.* 2023.
- Dao, T. *FlashAttention-2.* 2023; Dao, T. et al. *Flash-Decoding for long-context inference.* 2023.

## License

MIT. See [LICENSE](LICENSE).
