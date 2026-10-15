"""Turns a scheduler output into flattened model inputs and runs the model.

Decode steps are additionally captured into **CUDA graphs** (one per batch-size
bucket).  A decode step launches dozens of small kernels per layer; with CUDA
graphs the whole step is replayed with a single launch, which removes the
CPU launch overhead that otherwise dominates small-batch decode latency.
Inputs are copied into static buffers before replay; block tables and context
lengths are part of those buffers, so growing sequences need no re-capture.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from ..config import EngineConfig
from ..core.block_manager import BlockManager
from ..core.scheduler import SchedulerOutput
from ..models.llama import AttnMetadata, LlamaForCausalLM


@dataclass
class RunnerStats:
    num_forward: int = 0
    num_graph_replays: int = 0
    forward_time_s: float = 0.0
    tokens_processed: int = 0
    captured_batch_sizes: list[int] = field(default_factory=list)


class CUDAGraphRunner:
    """Captured decode step for a fixed batch size (every sequence contributes 1 token)."""

    def __init__(self, model: LlamaForCausalLM, kv_caches, batch_size: int, max_blocks: int, device):
        self.model = model
        self.bs = batch_size
        dev = device
        self.input_ids = torch.zeros(batch_size, dtype=torch.int64, device=dev)
        self.positions = torch.zeros(batch_size, dtype=torch.int64, device=dev)
        self.slot_mapping = torch.zeros(batch_size, dtype=torch.int64, device=dev)
        self.block_tables = torch.zeros((batch_size, max_blocks), dtype=torch.int32, device=dev)
        self.cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=dev)
        self.context_lens = torch.ones(batch_size, dtype=torch.int32, device=dev)
        self.logits_indices = torch.arange(batch_size, dtype=torch.int64, device=dev)
        self.meta = AttnMetadata(self.positions, self.slot_mapping, self.block_tables, self.cu_seqlens_q,
                                 self.context_lens, 1, self.logits_indices)
        self.kv_caches = kv_caches
        self.graph = torch.cuda.CUDAGraph()
        self.output: torch.Tensor | None = None

    def capture(self):
        # warm-up on a side stream (required before capture), then capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                self.model(self.input_ids, self.kv_caches, self.meta)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = self.model(self.input_ids, self.kv_caches, self.meta)

    def replay(self, input_ids, positions, slot_mapping, block_tables, context_lens) -> torch.Tensor:
        n = input_ids.shape[0]
        self.input_ids[:n].copy_(input_ids)
        self.positions[:n].copy_(positions)
        self.slot_mapping[:n].copy_(slot_mapping)
        self.block_tables.zero_()
        self.block_tables[:n, : block_tables.shape[1]].copy_(block_tables)
        self.context_lens.fill_(1)
        self.context_lens[:n].copy_(context_lens)
        if n < self.bs:  # padding rows: valid dummy work on block 0 / slot 0
            self.slot_mapping[n:].fill_(0)
            self.positions[n:].fill_(0)
        self.graph.replay()
        return self.output[:n]


class ModelRunner:
    def __init__(self, model: LlamaForCausalLM, config: EngineConfig, block_manager: BlockManager, kv_caches):
        self.model = model
        self.config = config
        self.bm = block_manager
        self.kv_caches = kv_caches
        self.device = torch.device(config.device)
        self.stats = RunnerStats()
        self.graphs: dict[int, CUDAGraphRunner] = {}
        self.max_blocks = (config.max_model_len + config.block_size - 1) // config.block_size
        self.use_graphs = config.use_cuda_graphs and self.device.type == "cuda"

    # -- CUDA graph capture ----------------------------------------------------
    def capture_graphs(self) -> None:
        if not self.use_graphs:
            return
        for bs in self.config.cuda_graph_batch_sizes:
            if bs > self.config.max_num_seqs:
                break
            runner = CUDAGraphRunner(self.model, self.kv_caches, bs, self.max_blocks, self.device)
            runner.capture()
            self.graphs[bs] = runner
            self.stats.captured_batch_sizes.append(bs)
        torch.cuda.synchronize()

    def _graph_for(self, n: int) -> CUDAGraphRunner | None:
        for bs in sorted(self.graphs):
            if bs >= n:
                return self.graphs[bs]
        return None

    # -- input construction ----------------------------------------------------
    def build_inputs(self, out: SchedulerOutput):
        ids, pos, slots, ctx, tables, logits_idx = [], [], [], [], [], []
        cu = [0]
        max_q = 0
        for ss in out.scheduled:
            seq = ss.seq
            start = seq.num_computed_tokens
            end = start + ss.num_new_tokens
            toks = seq.token_ids + seq.draft_token_ids
            ids.extend(toks[start:end])
            pos.extend(range(start, end))
            slots.extend(self.bm.slot_mapping(seq, start, end))
            cu.append(cu[-1] + ss.num_new_tokens)
            ctx.append(end)
            tables.append(seq.block_table)
            max_q = max(max_q, ss.num_new_tokens)
            if ss.samples:
                # logits for the last prompt token (first decode) + every draft position
                n_sample = 1 + len(seq.draft_token_ids) if not ss.is_prompt_chunk else 1
                logits_idx.extend(range(cu[-1] - n_sample, cu[-1]))
        max_blocks = max(len(t) for t in tables)
        bt = torch.zeros((len(tables), max_blocks), dtype=torch.int32)
        for i, t in enumerate(tables):
            bt[i, : len(t)] = torch.tensor(t, dtype=torch.int32)
        dev = self.device
        meta = AttnMetadata(
            positions=torch.tensor(pos, dtype=torch.int64, device=dev),
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev),
            block_tables=bt.to(dev),
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=dev),
            context_lens=torch.tensor(ctx, dtype=torch.int32, device=dev),
            max_q_len=max_q,
            logits_indices=torch.tensor(logits_idx, dtype=torch.int64, device=dev),
        )
        return torch.tensor(ids, dtype=torch.int64, device=dev), meta

    # -- execution -----------------------------------------------------------------
    @torch.no_grad()
    def execute(self, out: SchedulerOutput) -> torch.Tensor:
        """Returns logits [num_sampled_rows, vocab] in the order of ``meta.logits_indices``."""
        input_ids, meta = self.build_inputs(out)
        t0 = time.perf_counter()
        pure_decode = out.num_prefill_tokens == 0 and meta.max_q_len == 1
        n = len(out.scheduled)
        graph = self._graph_for(n) if (self.use_graphs and pure_decode) else None
        if graph is not None:
            logits = graph.replay(input_ids, meta.positions, meta.slot_mapping, meta.block_tables, meta.context_lens)
            self.stats.num_graph_replays += 1
        else:
            logits = self.model(input_ids, self.kv_caches, meta)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.stats.num_forward += 1
        self.stats.forward_time_s += time.perf_counter() - t0
        self.stats.tokens_processed += int(input_ids.shape[0])
        return logits
