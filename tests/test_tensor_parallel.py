"""Tensor parallelism: a 2-rank engine (gloo on CPU, NCCL on GPUs) must produce
token-identical greedy output to the single-process engine with the same weights."""
import json
import os
import sys

import pytest
import torch
import torch.multiprocessing as mp

from minivllm.config import EngineConfig, ModelConfig, SamplingParams
from minivllm.distributed import parallel_state as ps
from minivllm.distributed.layers import shard_state_dict

PROMPTS = ["hello world, this is a tensor parallel test", "short", "the quick brown fox " * 4]


def _cfg():
    return EngineConfig(num_gpu_blocks=128, block_size=8, use_cuda_graphs=False, device="cpu", max_model_len=256,
                        max_num_batched_tokens=32, max_num_seqs=8)


def _tiny_cfg():
    return ModelConfig.tiny(vocab_size=258, hidden=64, layers=2, heads=4, kv_heads=2)


def _worker(rank, world_size, sd_path, out_path, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    ps.init_tensor_parallel(rank, world_size, backend="gloo", device="cpu")
    try:
        from minivllm.core.engine import LLMEngine
        from minivllm.models.llama import LlamaForCausalLM
        from minivllm.tokenizer import ByteTokenizer

        torch.manual_seed(1234 + rank)                       # different seeds: init must not matter
        model = LlamaForCausalLM(_tiny_cfg())
        model.load_full_state_dict(torch.load(sd_path))     # sharded per rank
        engine = LLMEngine(model, ByteTokenizer(), _cfg())
        if engine.is_driver:
            outs = engine.generate(PROMPTS, SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True))
            engine.shutdown()
            with open(out_path, "w") as f:
                json.dump({"outs": outs, "world": ps.tp_size(), "local_kv_heads": model.local_kv_heads}, f)
        else:
            engine.worker_loop()
    finally:
        ps.destroy_tensor_parallel()


@pytest.mark.skipif(sys.platform != "linux", reason="multiprocessing spawn + gloo test")
def test_two_rank_tensor_parallel_matches_single_process(tmp_path):
    from minivllm.core.engine import LLMEngine
    from minivllm.models.llama import LlamaForCausalLM
    from minivllm.tokenizer import ByteTokenizer

    torch.manual_seed(0)
    full_model = LlamaForCausalLM(_tiny_cfg())
    sd_path = str(tmp_path / "full.pt")
    torch.save(full_model.state_dict(), sd_path)
    ref = LLMEngine(full_model, ByteTokenizer(), _cfg()).generate(
        PROMPTS, SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True))

    out_path = str(tmp_path / "tp.json")
    port = 29000 + (os.getpid() % 1000)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, 2, sd_path, out_path, port)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    with open(out_path) as f:
        got = json.load(f)
    assert got["world"] == 2 and got["local_kv_heads"] == 1
    assert got["outs"] == ref


def test_shard_state_dict_slices_correctly():
    full = {"a.q_proj.weight": torch.arange(16.0).view(4, 4), "a.o_proj.weight": torch.arange(16.0).view(4, 4),
            "lm_head.weight": torch.arange(8.0).view(4, 2), "norm.weight": torch.ones(4)}
    r1 = shard_state_dict(full, 1, 2, ["q_proj.weight"], ["o_proj.weight"], ["lm_head.weight"])
    assert torch.equal(r1["a.q_proj.weight"], full["a.q_proj.weight"][2:])
    assert torch.equal(r1["a.o_proj.weight"], full["a.o_proj.weight"][:, 2:])
    assert torch.equal(r1["lm_head.weight"], full["lm_head.weight"][2:])
    assert torch.equal(r1["norm.weight"], full["norm.weight"])


def test_single_process_collectives_are_identity():
    x = torch.randn(3)
    assert ps.all_reduce(x) is x and ps.all_gather_last_dim(x) is x and ps.broadcast_object({"k": 1}) == {"k": 1}
    assert not ps.get_tp().enabled and ps.tp_size() == 1
