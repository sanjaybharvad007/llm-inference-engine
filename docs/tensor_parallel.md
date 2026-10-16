# Tensor parallelism

One process per GPU, Megatron-LM style sharding, NCCL collectives.

## Sharding

| Projection | Layer | Shard | Collective |
|---|---|---|---|
| q/k/v, gate/up | `ColumnParallelLinear` | output dim (heads / FFN width) | none — the output is consumed shard-locally |
| o_proj, down_proj | `RowParallelLinear` | input dim | one **all-reduce** (sum of partial products) |
| lm_head | `VocabParallelLinear` | vocab rows | one **all-gather** so sampling sees the full vocabulary |

Attention heads are divided across ranks (`num_heads/tp` query heads,
`num_kv_heads/tp` KV heads per rank), so each rank also owns a `1/tp` slice of
the paged KV cache — the block manager and scheduler run only on rank 0 and
are unaware of TP. Every decoder layer costs exactly two all-reduces
regardless of width; bias terms are added once, after the reduction.

Weights are loaded unsharded and sliced per rank by
`LlamaForCausalLM.load_full_state_dict` (`distributed.layers.shard_state_dict`).
Quantization (`minivllm.quant`) is applied *after* sharding; scales are
per-output-row so the two commute.

## Driver / worker protocol

Rank 0 runs the scheduler and the sampler. Each step it flattens the batch
into a plain-Python payload (`ModelRunner.build_payload`) and broadcasts it
(`broadcast_object`); workers sit in `ModelRunner.worker_loop()` and execute
the same forward (including CUDA-graph replay) on their shard. The
all-reduces inside the model keep the ranks in lock-step; logits are
all-gathered so only rank 0 needs them. `engine.shutdown()` broadcasts `None`
to release the workers. The number of KV blocks is broadcast from rank 0 so
all ranks allocate identical caches.

```bash
torchrun --nproc_per_node=4 -m minivllm serve --model /path/to/Llama-3-70B -tp 4
```

The draft model used for speculative decoding is **replicated on the driver
only** (`parallel_state.replicated()`): it is small, so collectives would
cost more than the FLOPs they save, and the workers never see it.

## Test

`tests/test_tensor_parallel.py` spawns two gloo ranks on CPU, loads the same
full state dict into both a TP=1 engine and the 2-rank engine, and asserts the
greedy outputs are identical token-for-token.
