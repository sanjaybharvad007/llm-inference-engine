"""Paged KV-cache block manager with automatic prefix caching.

The KV cache is a pool of fixed-size physical blocks (``block_size`` tokens
each).  A sequence owns a *block table*: the list of physical blocks that hold
its logical token positions, in order.  This is the PagedAttention idea: no
contiguous per-request allocation, no reservation for the maximum length, and
~0 fragmentation beyond the last partial block.

Prefix caching: every *full* block is content-addressed by a hash chained over
all tokens from the start of the sequence (so the hash identifies the exact
prefix, not just the block's tokens).  A new request whose prompt starts with
an already-cached prefix reuses those blocks (ref-count +1) and skips their
prefill.  Blocks with zero references stay in an LRU "evictable" set until the
space is needed, so a system prompt shared by all requests stays resident.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from .sequence import Sequence


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    content_hash: Optional[int] = None    # set once the block is full and committed
    num_tokens: int = 0                   # tokens written (<= block_size)


def prefix_hash(prev_hash: Optional[int], tokens: tuple[int, ...]) -> int:
    return hash((prev_hash, tokens))


@dataclass
class BlockManagerStats:
    num_blocks: int = 0
    num_free: int = 0
    num_evictable: int = 0
    prefix_hits: int = 0
    prefix_queries: int = 0
    evictions: int = 0

    @property
    def usage(self) -> float:
        return 1.0 - self.num_free / max(self.num_blocks, 1)

    @property
    def hit_rate(self) -> float:
        return self.prefix_hits / max(self.prefix_queries, 1)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        assert num_blocks > 0 and block_size > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [PhysicalBlock(i) for i in range(num_blocks)]
        # Block 0 is reserved as the "null" block: padded rows of a CUDA-graph
        # batch read/write it, so it must never belong to a real sequence.
        self.NULL_BLOCK = 0
        self.free_blocks: list[int] = list(range(num_blocks - 1, 0, -1))   # never-used or fully released
        self.cached: dict[int, int] = {}                             # content hash -> block id
        self.evictable: OrderedDict[int, None] = OrderedDict()       # ref_count == 0 but cached (LRU order)
        self.stats = BlockManagerStats(num_blocks=num_blocks - 1, num_free=num_blocks - 1)

    # -- capacity ------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks) + len(self.evictable)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, seq: Sequence) -> bool:
        hits, hits_evictable = self._count_cached_prefix_blocks(seq) if self.enable_prefix_caching else (0, 0)
        # A cached block we are about to reuse that currently sits in the
        # evictable set is not free capacity for this allocation.
        return self.blocks_needed(seq.num_tokens) - hits <= self.num_free_blocks - hits_evictable

    def can_append_slots(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        need = self.blocks_needed(seq.num_tokens + num_new_tokens) - len(seq.block_table)
        return need <= self.num_free_blocks

    # -- allocation ------------------------------------------------------------
    def _take_free_block(self) -> int:
        if self.free_blocks:
            bid = self.free_blocks.pop()
        else:
            # Evict the least recently used cached block.
            bid, _ = self.evictable.popitem(last=False)
            blk = self.blocks[bid]
            if blk.content_hash is not None:
                self.cached.pop(blk.content_hash, None)
            self.stats.evictions += 1
        blk = self.blocks[bid]
        blk.ref_count = 1
        blk.content_hash = None
        blk.num_tokens = 0
        self._refresh_stats()
        return bid

    def _ref_cached_block(self, bid: int) -> None:
        blk = self.blocks[bid]
        if blk.ref_count == 0:
            self.evictable.pop(bid, None)
        blk.ref_count += 1
        self._refresh_stats()

    def _count_cached_prefix_blocks(self, seq: Sequence) -> tuple[int, int]:
        """(consecutive cached prefix blocks, how many of them are in the evictable set)."""
        n, n_evictable, h = 0, 0, None
        toks = seq.token_ids
        for b in range(len(toks) // self.block_size):
            h = prefix_hash(h, tuple(toks[b * self.block_size:(b + 1) * self.block_size]))
            bid = self.cached.get(h)
            if bid is None:
                break
            n += 1
            if bid in self.evictable:
                n_evictable += 1
        return n, n_evictable

    def allocate(self, seq: Sequence) -> None:
        """Give a (new or preempted) sequence blocks for all of its tokens,
        reusing cached prefix blocks where possible."""
        assert not seq.block_table, "sequence already has blocks"
        toks = seq.token_ids
        num_full = len(toks) // self.block_size
        total = self.blocks_needed(len(toks))
        # Pass 1: claim every cached prefix block *before* allocating anything,
        # so LRU eviction triggered by the fresh allocations below can never
        # evict a block this very sequence is about to reuse.
        h = None
        if self.enable_prefix_caching:
            for b in range(num_full):
                h = prefix_hash(h, tuple(toks[b * self.block_size:(b + 1) * self.block_size]))
                self.stats.prefix_queries += 1
                bid = self.cached.get(h)
                if bid is None:
                    break
                self.stats.prefix_hits += 1
                self._ref_cached_block(bid)
                seq.block_table.append(bid)
                seq.num_cached_tokens += self.block_size
        # Pass 2: fresh blocks for the rest.
        for _ in range(total - len(seq.block_table)):
            seq.block_table.append(self._take_free_block())
        # The prefill will start after the cached prefix.  Keep at least one
        # token to compute so the model produces logits for the next token.
        seq.num_computed_tokens = min(seq.num_cached_tokens, max(seq.num_tokens - 1, 0))
        seq.num_cached_tokens = seq.num_computed_tokens

    def append_slots(self, seq: Sequence, num_new_tokens: int = 1) -> None:
        """Make sure the next ``num_new_tokens`` positions have physical slots."""
        need = self.blocks_needed(seq.num_tokens + num_new_tokens) - len(seq.block_table)
        for _ in range(need):
            seq.block_table.append(self._take_free_block())

    def commit_full_blocks(self, seq: Sequence) -> None:
        """Register content hashes for blocks that became full (prefix caching).

        Called after tokens are appended; only the blocks whose tokens are all
        final (prompt or accepted output tokens) are hashed.
        """
        if not self.enable_prefix_caching:
            return
        toks = seq.token_ids
        # A block is reusable only once the KV of *all* its tokens is in the
        # cache, i.e. up to num_computed_tokens (the newest token's KV is
        # written in the step that consumes it as input).
        num_full = min(seq.num_computed_tokens // self.block_size, len(toks) // self.block_size, len(seq.block_table))
        h = None
        for b in range(num_full):
            chunk = tuple(toks[b * self.block_size:(b + 1) * self.block_size])
            h = prefix_hash(h, chunk)
            bid = seq.block_table[b]
            blk = self.blocks[bid]
            if blk.content_hash is None:
                blk.content_hash = h
                blk.num_tokens = self.block_size
                self.cached.setdefault(h, bid)
            elif blk.content_hash != h:
                # Block content changed (speculative rollback re-wrote it): re-key.
                if self.cached.get(blk.content_hash) == bid:
                    del self.cached[blk.content_hash]
                blk.content_hash = h
                self.cached.setdefault(h, bid)

    def free(self, seq: Sequence) -> None:
        for bid in seq.block_table:
            blk = self.blocks[bid]
            blk.ref_count -= 1
            assert blk.ref_count >= 0
            if blk.ref_count == 0:
                if blk.content_hash is not None and self.cached.get(blk.content_hash) == bid:
                    self.evictable[bid] = None          # keep for reuse until space is needed
                else:
                    blk.content_hash = None
                    self.free_blocks.append(bid)
        seq.block_table = []
        self._refresh_stats()

    def slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        """Physical slot (block_id * block_size + offset) for logical positions [start, end)."""
        bs = self.block_size
        return [seq.block_table[p // bs] * bs + p % bs for p in range(start, end)]

    def _refresh_stats(self) -> None:
        self.stats.num_free = len(self.free_blocks)
        self.stats.num_evictable = len(self.evictable)

    def __repr__(self) -> str:
        return f"BlockManager(blocks={self.num_blocks}, free={len(self.free_blocks)}, evictable={len(self.evictable)}, cached={len(self.cached)})"
