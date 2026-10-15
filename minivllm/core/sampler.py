"""Token sampling: temperature, top-k, top-p, repetition penalty, greedy."""
from __future__ import annotations

import torch

from ..config import SamplingParams


def apply_repetition_penalty(logits: torch.Tensor, token_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not token_ids:
        return logits
    ids = torch.tensor(sorted(set(token_ids)), device=logits.device)
    vals = logits[ids]
    logits[ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def sample_one(logits: torch.Tensor, params: SamplingParams, history: list[int], generator: torch.Generator | None) -> int:
    """logits: [vocab] fp32 for one position."""
    logits = apply_repetition_penalty(logits.clone(), history, params.repetition_penalty)
    if params.greedy:
        return int(torch.argmax(logits))
    logits = logits / params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > params.top_p      # keep the smallest set whose mass >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def probs_for(logits: torch.Tensor, params: SamplingParams, history: list[int]) -> torch.Tensor:
    """Full sampling distribution (used by speculative rejection sampling)."""
    logits = apply_repetition_penalty(logits.clone(), history, params.repetition_penalty)
    if params.greedy:
        p = torch.zeros_like(logits)
        p[int(torch.argmax(logits))] = 1.0
        return p
    logits = logits / params.temperature
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        sorted_logits = sorted_logits.masked_fill(cum - probs > params.top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    return torch.softmax(logits, dim=-1)
