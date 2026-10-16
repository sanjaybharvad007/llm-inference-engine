"""Engine, model and sampling configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """Llama-family architecture hyper-parameters."""

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: Optional[int] = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads

    @staticmethod
    def tiny(vocab_size: int = 256, hidden: int = 64, layers: int = 2, heads: int = 4, kv_heads: int = 2):
        """A random-weight model small enough for CPU tests."""
        return ModelConfig(
            vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2, num_layers=layers,
            num_heads=heads, num_kv_heads=kv_heads, max_position_embeddings=512, dtype=torch.float32,
            eos_token_id=0, bos_token_id=1,
        )


@dataclass
class EngineConfig:
    block_size: int = 16                     # tokens per KV-cache block
    num_gpu_blocks: Optional[int] = None     # None -> derive from gpu_memory_utilization
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256                  # max concurrent sequences in a step
    max_num_batched_tokens: int = 2048       # token budget per step (bounds prefill chunk size)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    max_model_len: int = 4096
    use_cuda_graphs: bool = True
    cuda_graph_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    tensor_parallel_size: int = 1            # one process per GPU; see minivllm.distributed
    speculative_ngram_max: int = 0           # >0 enables n-gram speculative decoding with this many draft tokens
    speculative_ngram_prompt_lookup: int = 3 # n-gram size to match
    speculative_draft_model: Optional[str] = None   # path to a small HF checkpoint used as draft model
    speculative_max_draft: int = 4           # draft tokens per step when a draft model is used
    draft_num_blocks: int = 512              # KV blocks for the draft model's own paged cache
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0
