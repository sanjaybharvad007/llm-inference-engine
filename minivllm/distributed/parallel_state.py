"""Tensor-parallel process group state and collectives.

One process per GPU (``torchrun --nproc_per_node=N`` or ``multiprocessing``);
NCCL on CUDA, gloo on CPU (used by the tests).  When no group is initialised
everything degrades to single-process no-ops, so the same model code runs
with TP=1.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ParallelState:
    rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


_state = ParallelState()


def init_tensor_parallel(rank: int, world_size: int, backend: str | None = None,
                         init_method: str | None = None, device: str = "cuda") -> ParallelState:
    """Initialise the TP group.  Safe to call with world_size == 1 (no-op)."""
    global _state
    if world_size <= 1:
        _state = ParallelState()
        return _state
    backend = backend or ("nccl" if device == "cuda" and torch.cuda.is_available() else "gloo")
    init_method = init_method or os.environ.get("MINIVLLM_INIT_METHOD", "env://")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    _state = ParallelState(rank=rank, world_size=world_size, backend=backend)
    return _state


def destroy_tensor_parallel() -> None:
    global _state
    if dist.is_initialized():
        dist.destroy_process_group()
    _state = ParallelState()


def get_tp() -> ParallelState:
    return _state


@contextlib.contextmanager
def replicated():
    """Temporarily behave as TP=1: layers built / run inside see no sharding and
    issue no collectives.  Used for the driver-local draft model in
    speculative decoding, which is small enough that sharding would only add
    latency."""
    global _state
    saved = _state
    _state = ParallelState()
    try:
        yield
    finally:
        _state = saved


def tp_rank() -> int:
    return _state.rank


def tp_size() -> int:
    return _state.world_size


# -- collectives ----------------------------------------------------------------
def all_reduce(x: torch.Tensor) -> torch.Tensor:
    if _state.enabled:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Concatenate per-rank shards along the last dimension."""
    if not _state.enabled:
        return x
    parts = [torch.empty_like(x) for _ in range(_state.world_size)]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=-1)


def broadcast_object(obj, src: int = 0):
    """Broadcast an arbitrary picklable object from ``src`` (used for scheduler decisions)."""
    if not _state.enabled:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def barrier() -> None:
    if _state.enabled:
        dist.barrier()
