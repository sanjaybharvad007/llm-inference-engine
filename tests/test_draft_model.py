"""Draft-model speculative decoding: greedy output must be token-identical to
plain greedy decoding of the target, whatever the draft proposes."""
import torch

from minivllm.config import EngineConfig, ModelConfig, SamplingParams
from minivllm.core.draft_model import DraftModelProposer
from minivllm.core.engine import LLMEngine
from minivllm.core.sequence import Sequence
from minivllm.models.llama import LlamaForCausalLM, reference_forward_full
from minivllm.tokenizer import ByteTokenizer

PROMPTS = ["hello world", "the quick brown fox jumps over the lazy dog", "abc " * 6, "x"]


def _cfg(**kw):
    base = dict(num_gpu_blocks=96, block_size=8, use_cuda_graphs=False, device="cpu", max_model_len=160,
                max_num_batched_tokens=24, max_num_seqs=8, draft_num_blocks=64)
    base.update(kw)
    return EngineConfig(**base)


def _models(seed=0):
    torch.manual_seed(seed)
    target = LlamaForCausalLM(ModelConfig.tiny(vocab_size=258, hidden=64, layers=2))
    draft = LlamaForCausalLM(ModelConfig.tiny(vocab_size=258, hidden=32, layers=1, heads=2, kv_heads=1))
    return target, draft


def test_proposer_matches_reference_greedy_continuation():
    """The draft's proposals are exactly what greedy decoding of the draft would produce."""
    _, draft = _models()
    prop = DraftModelProposer(draft, num_blocks=32, block_size=8, device="cpu", max_draft=4)
    seq = Sequence("r1", [5, 9, 13, 42, 7, 8], SamplingParams())
    drafts = prop.propose(seq)
    assert len(drafts) == 4
    toks = list(seq.token_ids)
    for d in drafts:                                        # replay with the unpaged reference
        assert int(reference_forward_full(draft, toks)[-1].argmax()) == d
        toks.append(d)
    # second call after the sequence advanced by two accepted drafts + a correction token
    seq.output_token_ids = drafts[:2] + [99]
    drafts2 = prop.propose(seq)
    toks = list(seq.token_ids)
    for d in drafts2:
        assert int(reference_forward_full(draft, toks)[-1].argmax()) == d
        toks.append(d)
    assert prop.states["r1"].cached == seq.token_ids + drafts2[:-1]


def test_proposer_frees_blocks_and_refuses_when_full():
    _, draft = _models()
    prop = DraftModelProposer(draft, num_blocks=3, block_size=8, device="cpu", max_draft=2)   # 2 usable blocks
    s1 = Sequence("a", list(range(5, 17)), SamplingParams())
    assert prop.propose(s1) and prop.num_free_blocks == 0
    s2 = Sequence("b", [1, 2, 3], SamplingParams())
    assert prop.propose(s2) == []            # no blocks left -> falls back to plain decode
    prop.free("a")
    assert prop.num_free_blocks == 2 and prop.propose(s2)


def test_draft_model_greedy_is_identical_to_plain_greedy():
    target, draft = _models()
    plain = LLMEngine(target, ByteTokenizer(), _cfg())
    params = SamplingParams(max_tokens=24, temperature=0.0, ignore_eos=True)
    ref = plain.generate(PROMPTS, params)
    spec = LLMEngine(target, ByteTokenizer(), _cfg(), draft_model=draft)
    got = spec.generate(PROMPTS, params)
    assert got == ref
    assert spec.stats.spec_proposed > 0 and spec.proposer.num_proposals > 0
    assert spec.proposer.num_free_blocks == 63      # everything freed when requests finish


def test_draft_model_with_preemption_and_stop_tokens():
    target, draft = _models(1)
    params = SamplingParams(max_tokens=30, temperature=0.0, stop_token_ids=[3], ignore_eos=True)
    plain = LLMEngine(target, ByteTokenizer(), _cfg(num_gpu_blocks=20))
    ref = plain.generate(PROMPTS, params)
    spec = LLMEngine(target, ByteTokenizer(), _cfg(num_gpu_blocks=20), draft_model=draft)
    got = spec.generate(PROMPTS, params)
    assert got == ref
    assert spec.scheduler.num_preemptions > 0


def test_draft_model_sampling_runs_and_respects_seed():
    target, draft = _models(2)
    params = SamplingParams(max_tokens=12, temperature=0.8, top_p=0.9, seed=7)
    e1 = LLMEngine(target, ByteTokenizer(), _cfg(), draft_model=draft)
    e2 = LLMEngine(target, ByteTokenizer(), _cfg(), draft_model=draft)
    o1, o2 = e1.generate(PROMPTS[:2], params), e2.generate(PROMPTS[:2], params)
    assert all(len(o) == 12 for o in o1) and o1 == o2


def test_preemption_drops_pending_drafts():
    """Regression: a sequence preempted with drafts pending used to re-enter as a
    prefill and run the verify path on a single logits row, corrupting
    num_computed_tokens (and eventually indexing past its block table)."""
    from minivllm.core.block_manager import BlockManager
    from minivllm.core.scheduler import Scheduler

    cfg = _cfg(num_gpu_blocks=4, block_size=8)
    bm = BlockManager(4, 8, True)
    sch = Scheduler(cfg, bm)
    seq = Sequence("p", list(range(5, 12)), SamplingParams(max_tokens=10))
    sch.add(seq)
    sch.schedule()
    seq.num_computed_tokens = seq.num_tokens - 1
    seq.draft_token_ids = [1, 2, 3]
    out = sch.schedule()
    assert out.scheduled and out.scheduled[0].num_new_tokens == 4
    sch._preempt(seq, out)
    assert seq.draft_token_ids == [] and seq.num_computed_tokens == 0
