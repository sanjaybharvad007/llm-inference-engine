"""Tensor-parallel linear layers (Megatron-LM style).

* ``ColumnParallelLinear``: the weight's output dimension is split across ranks;
  each rank computes its slice of the output.  Used for q/k/v and gate/up
  projections, whose outputs are consumed shard-locally.
* ``RowParallelLinear``: the input dimension is split; each rank computes a
  partial product on its slice of the input and the shards are summed with one
  all-reduce.  Used for o_proj and down_proj, so every decoder layer costs
  exactly two all-reduces regardless of width.
* ``VocabParallelLinear``: the LM head's vocabulary is split; logits are
  all-gathered so sampling sees the full vocabulary on every rank.

Weights are stored already sharded; ``shard_state_dict`` slices a full
HuggingFace state dict for a given rank.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import parallel_state as ps


def _init_like_linear(weight: nn.Parameter, bias: nn.Parameter | None) -> None:
    """Same initialisation as nn.Linear so random (test) models are well-conditioned."""
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        bound = 1.0 / math.sqrt(weight.shape[1])
        nn.init.uniform_(bias, -bound, bound)


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        tp = ps.tp_size()
        assert out_features % tp == 0, f"out_features {out_features} not divisible by tp={tp}"
        self.in_features, self.out_features = in_features, out_features
        self.out_per_rank = out_features // tp
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank)) if bias else None
        _init_like_linear(self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        tp = ps.tp_size()
        assert in_features % tp == 0, f"in_features {in_features} not divisible by tp={tp}"
        self.in_features, self.out_features = in_features, out_features
        self.in_per_rank = in_features // tp
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        _init_like_linear(self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight)
        out = ps.all_reduce(partial)
        if self.bias is not None:
            out = out + self.bias          # added once, after the reduction
        return out


class VocabParallelLinear(nn.Module):
    def __init__(self, hidden: int, vocab_size: int):
        super().__init__()
        tp = ps.tp_size()
        assert vocab_size % tp == 0, f"vocab {vocab_size} not divisible by tp={tp}"
        self.vocab_per_rank = vocab_size // tp
        self.weight = nn.Parameter(torch.empty(self.vocab_per_rank, hidden))
        _init_like_linear(self.weight, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ps.all_gather_last_dim(F.linear(x, self.weight))


def shard_state_dict(full: dict[str, torch.Tensor], rank: int, world_size: int,
                     column: list[str], row: list[str], vocab: list[str]) -> dict[str, torch.Tensor]:
    """Slice a full (unsharded) state dict for one TP rank.

    column: parameter-name suffixes split along dim 0 (out_features)
    row:    suffixes split along dim 1 (in_features)
    vocab:  suffixes split along dim 0 (vocabulary rows)
    """
    out = {}
    for name, t in full.items():
        if any(name.endswith(s) for s in column + vocab):
            n = t.shape[0] // world_size
            out[name] = t[rank * n:(rank + 1) * n].clone()
        elif any(name.endswith(s) for s in row):
            n = t.shape[1] // world_size
            out[name] = t[:, rank * n:(rank + 1) * n].clone()
        else:
            out[name] = t
    return out
