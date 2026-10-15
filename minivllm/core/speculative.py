"""N-gram ("prompt lookup") speculative decoding.

Draft tokens are proposed for free by matching the last ``n`` generated tokens
against earlier occurrences in the sequence and copying what followed.  The
target model then scores the draft tokens *in one forward pass* (they are
appended as a multi-token chunk, exactly like a prefill chunk), and the longest
prefix that the model agrees with is accepted.

* Greedy: accept while ``argmax(logits_i) == draft_i``; the first disagreement
  is replaced by the model's own argmax (so the output is token-for-token
  identical to plain greedy decoding — tested).
* Sampling: standard rejection sampling (Leviathan et al. 2023) against the
  draft distribution, which for a deterministic n-gram draft is a one-hot, so
  token ``d`` is accepted with probability ``p_target(d)`` and otherwise a token
  is drawn from the residual.

Rejected draft tokens are rolled back; their cache slots are simply overwritten
by the next tokens, so no block-table surgery is needed.
"""
from __future__ import annotations

import torch

from ..config import SamplingParams
from .sampler import probs_for
from .sequence import Sequence


def propose_ngram(seq: Sequence, ngram: int, max_draft: int) -> list[int]:
    toks = seq.token_ids
    if len(toks) < ngram + 1 or max_draft <= 0:
        return []
    tail = toks[-ngram:]
    # search most recent match first
    for i in range(len(toks) - ngram - 1, -1, -1):
        if toks[i:i + ngram] == tail:
            cont = toks[i + ngram:i + ngram + max_draft]
            if cont:
                return list(cont)
    return []


def verify(
    seq: Sequence,
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None,
) -> tuple[list[int], int]:
    """logits: [1 + num_drafts, vocab] — row i scores position of draft i (row 0
    scores the first draft, the last row scores the token after all drafts).

    Returns (accepted tokens incl. the bonus/correction token, num_accepted_drafts).
    """
    drafts = seq.draft_token_ids
    history = list(seq.token_ids)
    accepted: list[int] = []
    n_acc = 0
    for i, d in enumerate(drafts):
        p = probs_for(logits[i], params, history)
        if params.greedy:
            ok = int(torch.argmax(p)) == d
        else:
            u = torch.rand((), generator=generator, device="cpu").item()
            ok = u < float(p[d])
        if ok:
            accepted.append(d)
            history.append(d)
            n_acc += 1
            continue
        # reject: sample from the residual (p - q)+ where q is one-hot at d
        if params.greedy:
            tok = int(torch.argmax(p))
        else:
            residual = p.clone()
            residual[d] = 0.0
            residual = residual / residual.sum().clamp_min(1e-12)
            tok = int(torch.multinomial(residual, 1, generator=generator))
        accepted.append(tok)
        return accepted, n_acc
    # all drafts accepted: the bonus token comes from the last row
    p = probs_for(logits[len(drafts)], params, history)
    tok = int(torch.argmax(p)) if params.greedy else int(torch.multinomial(p, 1, generator=generator))
    accepted.append(tok)
    return accepted, n_acc
