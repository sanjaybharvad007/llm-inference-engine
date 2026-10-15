"""End-to-end: the paged / batched / chunked / cached / speculative engine must
reproduce token-for-token what a naive full-recompute greedy loop produces."""
import os

import pytest
import torch

from minivllm.config import SamplingParams
from minivllm.core.sequence import SequenceStatus

from conftest import greedy_reference, make_engine

PROMPTS = [
    "hello world, this is a test of the engine",
    "a much longer prompt that will need several blocks " * 3,
    "x",
    "the quick brown fox jumps over the lazy dog",
]


def _check_greedy(engine, prompts, n=10):
    params = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    outs = engine.generate(prompts, params)
    for p, o in zip(prompts, outs):
        ref = greedy_reference(engine, engine.tokenizer.encode(p), n)
        assert o == ref, (p[:20], o, ref)


def test_batched_greedy_matches_reference():
    _check_greedy(make_engine(), PROMPTS)


def test_chunked_prefill_matches_reference():
    # tiny token budget: every prompt is split over several steps and mixed with decodes
    eng = make_engine(max_num_batched_tokens=7)
    _check_greedy(eng, PROMPTS)
    assert eng.stats.num_steps > 10


def test_prefix_caching_reuses_blocks_and_matches():
    eng = make_engine(block_size=4)
    system = "You are a helpful assistant. " * 3
    params = SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True)
    first = eng.generate([system + "Question one?"], params)[0]
    assert eng.block_manager.stats.prefix_hits == 0
    # second request shares the long prefix -> cached blocks, no recompute, identical result
    rid = eng.add_request(None, system + "Question one?", params)
    out = None
    while eng.has_unfinished():
        for r in eng.step():
            if r.finished:
                out = r
    assert out.request_id == rid and out.output_token_ids == first
    assert out.num_cached_tokens > 0 and eng.block_manager.stats.prefix_hits > 0
    # and a different continuation is still correct
    _check_greedy(eng, [system + "Question two, different?"], n=6)


def test_preemption_under_memory_pressure_matches_reference():
    # the four prompts + 8 new tokens each need ~70 blocks; give them 51
    eng = make_engine(num_gpu_blocks=52, block_size=4, max_num_seqs=8)
    _check_greedy(eng, PROMPTS, n=8)
    assert eng.scheduler.num_preemptions > 0
    assert eng.block_manager.num_free_blocks == 51


def test_request_larger_than_cache_is_rejected_not_livelocked():
    eng = make_engine(num_gpu_blocks=6, block_size=4)   # 20 tokens of cache
    eng.add_request(None, "x" * 30, SamplingParams(max_tokens=4))
    outs = []
    for _ in range(5):
        outs += eng.step()
    assert not eng.has_unfinished() and outs == []


def test_speculative_ngram_greedy_identical_and_accepts():
    eng = make_engine(speculative_ngram_max=4, speculative_ngram_prompt_lookup=2)
    # a highly repetitive prompt makes n-gram drafts hit
    prompts = ["ab ab ab ab ab ab ab ab ab ab ab", PROMPTS[1]]
    _check_greedy(eng, prompts, n=16)
    assert eng.stats.spec_proposed > 0 and eng.stats.spec_accepted > 0


def test_speculative_sampling_is_valid_and_bounded():
    eng = make_engine(speculative_ngram_max=3)
    params = SamplingParams(max_tokens=20, temperature=0.8, top_p=0.9, ignore_eos=True)
    outs = eng.generate(["ab ab ab ab ab ab ab ab"] * 3, params)
    assert all(len(o) == 20 for o in outs)


def test_stop_conditions():
    eng = make_engine()
    enc = eng.tokenizer.encode("hello")
    params = SamplingParams(max_tokens=50, temperature=0.0, stop_token_ids=[], ignore_eos=True)
    # pick the first greedy token and use it as a stop token -> stops after 1
    first = greedy_reference(eng, enc, 1)[0]
    eng.add_request(None, "hello", SamplingParams(max_tokens=50, temperature=0.0, stop_token_ids=[first]))
    last = None
    while eng.has_unfinished():
        for r in eng.step():
            last = r
    assert last.finished and last.finish_reason == "stop" and last.output_token_ids == [first]
    # max_tokens
    eng.add_request(None, "hello", params)
    while eng.has_unfinished():
        for r in eng.step():
            last = r
    assert last.finish_reason == "length" and len(last.output_token_ids) == 50


def test_abort_frees_blocks():
    eng = make_engine()
    rid = eng.add_request(None, PROMPTS[1], SamplingParams(max_tokens=100, ignore_eos=True))
    eng.step()
    eng.step()
    assert eng.block_manager.num_free_blocks < 127
    assert eng.abort(rid)
    assert not eng.has_unfinished() and eng.block_manager.num_free_blocks == 127


def test_streaming_outputs_arrive_every_step():
    eng = make_engine()
    rid = eng.add_request(None, "stream me", SamplingParams(max_tokens=5, temperature=0.0, ignore_eos=True))
    seen = []
    while eng.has_unfinished():
        for r in eng.step():
            seen.extend(r.new_token_ids)
            assert r.request_id == rid
    assert len(seen) == 5


def test_sampling_reproducible_with_seed():
    a = make_engine(seed=1).generate(["seed test"], SamplingParams(max_tokens=8, temperature=1.0, top_k=20, ignore_eos=True))
    b = make_engine(seed=1).generate(["seed test"], SamplingParams(max_tokens=8, temperature=1.0, top_k=20, ignore_eos=True))
    assert a == b


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1" and not torch.cuda.is_available(), reason="Triton path")
def test_triton_paged_attention_engine_matches_reference():
    os.environ["MINIVLLM_ATTN_BACKEND"] = "auto"
    eng = make_engine(block_size=16)
    _check_greedy(eng, PROMPTS[:2], n=4)


def test_max_model_len_rejects_long_prompt():
    eng = make_engine(max_model_len=16)
    with pytest.raises(ValueError):
        eng.add_request(None, "x" * 40, SamplingParams())
    eng.add_request(None, "x" * 10, SamplingParams(max_tokens=100, ignore_eos=True))
    last = None
    while eng.has_unfinished():
        for r in eng.step():
            last = r
    assert last.finish_reason == "length" and eng.scheduler.running == []
    assert SequenceStatus.FINISHED_LENGTH.finished
