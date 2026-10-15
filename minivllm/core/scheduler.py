"""Continuous-batching scheduler with chunked prefill and preemption.

Every engine step the scheduler builds one batch:

1. **Decode** sequences (prompt fully processed) each get one new token slot.
   They are served first because they are latency-critical.
2. Remaining token budget (``max_num_batched_tokens``) goes to **prefill**:
   running sequences that still have un-computed prompt tokens get a chunk, then
   waiting sequences are admitted (FIFO) as long as blocks and budget allow.
   Chunked prefill lets a long prompt be spread over several steps so decode
   tokens of other requests are not stalled behind it (bounded TTFT jitter and
   steady inter-token latency).
3. If a decode sequence cannot get a block, the **lowest-priority running
   sequence is preempted** (recompute mode: blocks freed, it re-enters the
   waiting queue at the front and will re-prefill, hitting the prefix cache for
   its already-full blocks).

Prefill and decode tokens of all sequences are flattened into a single
variable-length batch for the model runner.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..config import EngineConfig
from .block_manager import BlockManager
from .sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int          # tokens whose KV will be computed this step
    is_prompt_chunk: bool        # True if some of them are prompt tokens
    samples: bool = False        # logits sampled this step? (False for a prefill chunk that
                                 # does not reach the end of the prompt).  Frozen at schedule
                                 # time because the engine mutates num_computed_tokens later.

    def __post_init__(self):
        self.samples = self.seq.num_computed_tokens + self.num_new_tokens >= self.seq.num_tokens


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def num_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager):
        self.config = config
        self.bm = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    # -- queue management ----------------------------------------------------
    def add(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.append(seq)

    def abort(self, request_id: str) -> bool:
        for q in (self.waiting, self.running):
            for s in list(q):
                if s.request_id == request_id:
                    q.remove(s)
                    if s.block_table:
                        self.bm.free(s)
                    s.finish(SequenceStatus.FINISHED_ABORTED)
                    return True
        return False

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    # -- scheduling ----------------------------------------------------------
    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        seq.reset_for_recompute()
        seq.status = SequenceStatus.PREEMPTED
        self.waiting.appendleft(seq)
        out.preempted.append(seq)
        self.num_preemptions += 1

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_num_batched_tokens
        max_seqs = self.config.max_num_seqs

        # 1. decode sequences (one token each; speculative drafts count extra)
        for seq in list(self.running):
            if seq.is_prefill:
                continue
            n_new = 1 + len(seq.draft_token_ids)
            while not self.bm.can_append_slots(seq, n_new):
                victim = self.running[-1]
                if victim is seq:
                    self._preempt(seq, out)   # nothing lower-priority to evict: preempt self
                    break
                self._preempt(victim, out)
            if seq.status != SequenceStatus.RUNNING:
                continue
            self.bm.append_slots(seq, n_new)
            out.scheduled.append(ScheduledSeq(seq, n_new, is_prompt_chunk=False))
            out.num_decode_tokens += n_new
            budget -= n_new

        # 2. prefill chunks for running sequences still processing their prompt
        for seq in self.running:
            if not seq.is_prefill or budget <= 0:
                continue
            remaining = seq.num_tokens - seq.num_computed_tokens
            n_new = min(remaining, budget) if self.config.enable_chunked_prefill else remaining
            if n_new < remaining and not self.config.enable_chunked_prefill:
                continue
            out.scheduled.append(ScheduledSeq(seq, n_new, is_prompt_chunk=True))
            out.num_prefill_tokens += n_new
            budget -= n_new

        # 3. admit waiting sequences
        while self.waiting and budget > 0 and len(self.running) < max_seqs:
            seq = self.waiting[0]
            if seq.num_tokens + 1 > self.config.max_model_len or \
                    self.bm.blocks_needed(seq.num_tokens + 1) > self.bm.stats.num_blocks:
                # can never be served: longer than the model window / the whole cache
                self.waiting.popleft()
                seq.finish(SequenceStatus.FINISHED_LENGTH)
                continue
            if not self.bm.can_allocate(seq):
                break
            remaining_after_cache = seq.num_tokens  # unknown until allocate() resolves the cache
            if not self.config.enable_chunked_prefill and remaining_after_cache > budget:
                break
            self.waiting.popleft()
            self.bm.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            remaining = seq.num_tokens - seq.num_computed_tokens
            n_new = min(remaining, budget)
            out.scheduled.append(ScheduledSeq(seq, n_new, is_prompt_chunk=True))
            out.num_prefill_tokens += n_new
            budget -= n_new
        return out

    # -- post-step bookkeeping ---------------------------------------------------
    def finish_sequence(self, seq: Sequence, status: SequenceStatus) -> None:
        if seq in self.running:
            self.running.remove(seq)
        if seq.block_table:
            self.bm.free(seq)
        seq.finish(status)
