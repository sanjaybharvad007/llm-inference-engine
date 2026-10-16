"""Draft-model speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small model of the same tokenizer family proposes ``k`` tokens
autoregressively (greedy, k forward passes of a tiny model); the target then
scores all of them in one forward pass through the same ``verify`` path the
n-gram proposer uses.  Compared with n-gram lookup, a draft model proposes
useful tokens on *every* step, not only when the text repeats itself, so
acceptance rates of 60-80% (3-4 of 4 drafts) are typical for a 7B target with
a ~100M draft, i.e. ~2-3x fewer target steps.

Design
------
* The draft model has its **own paged KV cache** (small: ``draft_num_blocks``)
  with a trivial allocator, keyed by request id.  It is never touched by the
  target's block manager, prefix cache or preemption.
* The draft cache is kept consistent by *prefix matching*: we remember which
  token prefix each request's draft cache holds, and before proposing we feed
  only the tokens beyond the longest common prefix with the sequence's actual
  tokens.  Rejected drafts and preemption/recompute therefore need no
  explicit rollback — a diverged suffix is simply overwritten.
* Under tensor parallelism the draft model lives **replicated on the driver
  rank only** (``parallel_state.replicated()``): it is tiny, so sharding would
  cost more in collectives than it saves in FLOPs, and the worker ranks never
  see it — they only execute the target's verification batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..distributed import parallel_state as ps
from ..models.llama import AttnMetadata, LlamaForCausalLM
from .sequence import Sequence


@dataclass
class _DraftState:
    blocks: list[int] = field(default_factory=list)
    cached: list[int] = field(default_factory=list)   # tokens whose KV the draft cache holds, in order


class DraftModelProposer:
    def __init__(self, draft_model: LlamaForCausalLM, num_blocks: int, block_size: int, device, max_draft: int):
        self.block_size = block_size
        self.device = torch.device(device)
        self.max_draft = max_draft
        with ps.replicated():
            self.model = draft_model.eval().to(self.device)
            self.caches = self.model.allocate_kv_caches(num_blocks, block_size, self.device)
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, 0, -1))     # block 0 reserved (null block, like the target)
        self.states: dict[str, _DraftState] = {}
        self.max_pos = self.model.cfg.max_position_embeddings
        self.num_proposals = 0
        self.num_draft_forwards = 0

    # -- allocation ---------------------------------------------------------------
    def _ensure_capacity(self, st: _DraftState, num_tokens: int) -> bool:
        need = (num_tokens + self.block_size - 1) // self.block_size
        while len(st.blocks) < need:
            if not self._free:
                return False
            st.blocks.append(self._free.pop())
        return True

    def free(self, request_id: str) -> None:
        st = self.states.pop(request_id, None)
        if st is not None:
            self._free.extend(st.blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    # -- forward helpers ------------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, st: _DraftState, tokens: list[int], start: int) -> int:
        """Run ``tokens`` at positions start.. through the draft; return argmax of the last position."""
        end = start + len(tokens)
        bs = self.block_size
        dev = self.device
        slots = [st.blocks[p // bs] * bs + p % bs for p in range(start, end)]
        meta = AttnMetadata(
            positions=torch.arange(start, end, device=dev),
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev),
            block_tables=torch.tensor([st.blocks], dtype=torch.int32, device=dev),
            cu_seqlens_q=torch.tensor([0, len(tokens)], dtype=torch.int32, device=dev),
            context_lens=torch.tensor([end], dtype=torch.int32, device=dev),
            max_q_len=len(tokens),
            logits_indices=torch.tensor([len(tokens) - 1], dtype=torch.int64, device=dev),
        )
        with ps.replicated():
            logits = self.model(torch.tensor(tokens, dtype=torch.int64, device=dev), self.caches, meta)
        self.num_draft_forwards += 1
        return int(torch.argmax(logits[0]))

    # -- the proposal ---------------------------------------------------------------------
    def propose(self, seq: Sequence, max_draft: int | None = None) -> list[int]:
        k = min(self.max_draft if max_draft is None else max_draft, self.max_draft)
        toks = seq.token_ids
        if k <= 0 or len(toks) + k >= self.max_pos:
            return []
        st = self.states.setdefault(seq.request_id, _DraftState())
        if not self._ensure_capacity(st, len(toks) + k):
            return []                          # draft cache full: fall back to plain decoding this step

        # longest common prefix between what the draft cache holds and the real sequence
        n = 0
        lim = min(len(st.cached), len(toks) - 1)   # always re-feed at least the last token
        while n < lim and st.cached[n] == toks[n]:
            n += 1
        del st.cached[n:]
        drafts: list[int] = []
        nxt = self._forward(st, toks[n:], n)       # catch-up chunk (whole prompt on the first call)
        st.cached.extend(toks[n:])
        drafts.append(nxt)
        for _ in range(k - 1):
            nxt = self._forward(st, [drafts[-1]], len(st.cached))
            st.cached.append(drafts[-1])
            drafts.append(nxt)
        self.num_proposals += 1
        return drafts
