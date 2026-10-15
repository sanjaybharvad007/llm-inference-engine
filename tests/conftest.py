import pytest
import torch

from minivllm.config import EngineConfig, SamplingParams
from minivllm.core.engine import LLMEngine
from minivllm.models.llama import reference_forward_full


def make_engine(**overrides) -> LLMEngine:
    cfg = dict(num_gpu_blocks=128, block_size=8, use_cuda_graphs=False, device="cpu", max_model_len=256,
               max_num_batched_tokens=64, max_num_seqs=8)
    cfg.update(overrides)
    return LLMEngine.tiny(EngineConfig(**cfg), seed=0)


def greedy_reference(engine: LLMEngine, prompt_ids: list[int], n: int) -> list[int]:
    """Token-by-token full recompute with the un-paged forward: ground truth."""
    out: list[int] = []
    for _ in range(n):
        logits = reference_forward_full(engine.model, prompt_ids + out)
        out.append(int(torch.argmax(logits[-1])))
    return out


@pytest.fixture
def greedy():
    return SamplingParams(max_tokens=10, temperature=0.0, ignore_eos=True)
