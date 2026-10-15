"""Request / sequence state tracked by the scheduler."""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional

from ..config import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED_STOPPED = "finished_stopped"      # EOS / stop token
    FINISHED_LENGTH = "finished_length"        # max_tokens or max_model_len
    FINISHED_ABORTED = "finished_aborted"

    @property
    def finished(self) -> bool:
        return self.name.startswith("FINISHED")


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.perf_counter)

    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_table: list[int] = field(default_factory=list)   # physical block ids, logical order
    num_computed_tokens: int = 0                            # tokens whose KV is already in the cache
    num_cached_tokens: int = 0                              # of those, how many came from the prefix cache
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    # Speculative decoding: tokens proposed but not yet verified.
    draft_token_ids: list[int] = field(default_factory=list)
    stats_accepted_drafts: int = 0
    stats_proposed_drafts: int = 0

    # -- token views -----------------------------------------------------
    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        """True while more than one token still needs its KV computed.

        For a fresh request that is the prompt; for a preempted request it is
        the prompt *plus* the tokens it had already generated (recompute mode).
        A decode step has exactly one uncomputed token: the last sampled one."""
        return self.num_computed_tokens < self.num_tokens - 1

    @property
    def is_finished(self) -> bool:
        return self.status.finished

    # -- mutation ----------------------------------------------------------
    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def rollback(self, n: int) -> None:
        """Drop the last n output tokens (rejected speculative drafts)."""
        if n > 0:
            del self.output_token_ids[-n:]
            self.num_computed_tokens = min(self.num_computed_tokens, self.num_tokens)

    def reset_for_recompute(self) -> None:
        self.block_table = []
        self.num_computed_tokens = 0
        self.num_cached_tokens = 0
        self.status = SequenceStatus.WAITING

    def finish(self, status: SequenceStatus) -> None:
        self.status = status
        self.finish_time = time.perf_counter()

    def __repr__(self) -> str:
        return f"Sequence({self.request_id}, {self.status.value}, {self.num_tokens} tok, computed={self.num_computed_tokens})"


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    output_token_ids: list[int]
    finished: bool
    finish_reason: Optional[str] = None
    num_cached_tokens: int = 0
