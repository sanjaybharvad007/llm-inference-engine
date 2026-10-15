from minivllm.config import EngineConfig, SamplingParams
from minivllm.core.block_manager import BlockManager
from minivllm.core.scheduler import Scheduler
from minivllm.core.sequence import Sequence


def _sched(**kw):
    cfg = EngineConfig(block_size=4, max_num_batched_tokens=kw.pop("budget", 16), max_num_seqs=kw.pop("max_seqs", 4),
                       max_model_len=64, device="cpu", **kw)
    return Scheduler(cfg, BlockManager(33, 4))


def _seq(n, rid):
    return Sequence(rid, list(range(2, 2 + n)), SamplingParams(max_tokens=8))


def _advance(sched, out):
    """Emulate the engine: mark tokens computed, append one token for sampled seqs."""
    for ss in out.scheduled:
        ss.seq.num_computed_tokens += ss.num_new_tokens
        if ss.samples:
            ss.seq.append_token(7)
            ss.seq.num_computed_tokens = ss.seq.num_tokens - 1


def test_chunked_prefill_splits_long_prompt_and_prioritises_decode():
    s = _sched(budget=10)
    s.add(_seq(25, "long"))
    o1 = s.schedule()
    assert [(x.seq.request_id, x.num_new_tokens, x.samples) for x in o1.scheduled] == [("long", 10, False)]
    _advance(s, o1)
    s.add(_seq(3, "short"))
    o2 = s.schedule()                       # long consumes the whole budget again; short waits
    assert [(x.seq.request_id, x.num_new_tokens) for x in o2.scheduled] == [("long", 10)]
    assert s.num_waiting == 1
    _advance(s, o2)
    o3 = s.schedule()
    kinds = [(x.seq.request_id, x.num_new_tokens, x.is_prompt_chunk) for x in o3.scheduled]
    assert ("long", 5, True) in kinds
    _advance(s, o3)
    o4 = s.schedule()                       # now decode tokens come first
    assert o4.scheduled[0].is_prompt_chunk is False and o4.num_decode_tokens >= 1


def test_max_num_seqs_limits_admission():
    s = _sched(max_seqs=2, budget=100)
    for i in range(3):
        s.add(_seq(4, f"r{i}"))
    o = s.schedule()
    assert len(o.scheduled) == 2 and s.num_waiting == 1


def test_preemption_recompute_puts_victim_at_queue_front():
    cfg = EngineConfig(block_size=4, max_num_batched_tokens=100, max_num_seqs=8, max_model_len=64, device="cpu")
    s = Scheduler(cfg, BlockManager(5, 4))      # 4 usable blocks
    a, b = _seq(7, "a"), _seq(7, "b")           # 2 blocks each
    s.add(a)
    s.add(b)
    o = s.schedule()
    _advance(s, o)
    # both are now in decode with 8 tokens: each needs a 3rd block -> only one can get it
    o2 = s.schedule()
    assert len(o2.preempted) == 1 and o2.preempted[0] is b
    assert s.waiting[0] is b and b.block_table == [] and b.num_computed_tokens == 0
    assert [x.seq.request_id for x in o2.scheduled] == ["a"]


def test_abort_removes_from_any_queue():
    s = _sched()
    s.add(_seq(4, "x"))
    assert s.abort("x") and not s.has_unfinished()
    assert not s.abort("nope")
