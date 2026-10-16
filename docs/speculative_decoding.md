# Speculative decoding

Two proposers share one verifier (`core/speculative.verify`):

| Proposer | Cost of a proposal | When it helps |
|---|---|---|
| n-gram / prompt lookup (`core/speculative.py`) | free (string matching) | repetitive text: code, summarisation, RAG with quoted context |
| draft model (`core/draft_model.py`) | k forward passes of a small model | everything else; 60-80% acceptance with a ~10x smaller draft |

Verification appends the k drafts to the sequence as a multi-token chunk —
the same code path as a prefill chunk — and the target scores all k+1
positions in one forward pass. Greedy acceptance is the longest prefix the
target agrees with, plus the target's own token at the first disagreement, so
output is **token-identical to plain greedy decoding** (`tests/test_draft_model.py`
asserts this, including with preemption and stop tokens). Sampling uses
Leviathan-style rejection sampling against the draft distribution.

## Draft model design

- **Own paged KV cache.** The draft has a private cache
  (`EngineConfig.draft_num_blocks`) with a trivial allocator keyed by request
  id. The target's block manager, prefix cache and preemption never touch it.
- **Prefix matching instead of rollback.** The proposer remembers which token
  prefix each request's draft cache holds. Before proposing it feeds only the
  tokens beyond the longest common prefix with the sequence's real tokens —
  so rejected drafts, the target's correction token and a recompute after
  preemption are all handled by the same two lines, with no explicit rollback.
- **Driver-local under TP.** The draft is replicated on rank 0 only
  (`parallel_state.replicated()`); the workers only ever see the target's
  verification batch.
- **Graceful degradation.** If the draft cache is full, or the sequence is
  within k tokens of `max_model_len` / `max_tokens`, the step runs as plain
  decoding.

```bash
python -m minivllm serve --model /path/to/Llama-3-8B --draft-model /path/to/Llama-3.2-1B --num-speculative-tokens 4
python benchmarks/bench_decode.py spec --model ... --draft-model ...    # plain vs n-gram vs draft: tok/s, acceptance
```

## A bug this surfaced

A sequence preempted while it had drafts pending used to be re-admitted as a
prefill with the drafts still attached; the engine then ran the verify path on
a single logits row, mis-set `num_computed_tokens`, and eventually indexed
past its block table. Drafts are now dropped on preemption
(`Sequence.reset_for_recompute`) and verification is only run on decode
steps; `test_preemption_drops_pending_drafts` covers it.
