"""minivllm: an LLM inference & serving engine (paged KV cache, continuous batching,
prefix caching, chunked prefill, CUDA graphs, CUDA/Triton paged attention, tensor
parallelism, INT8/FP8 weight-only quantization, n-gram and draft-model speculative
decoding, OpenAI API)."""
from .config import EngineConfig, ModelConfig, SamplingParams
from .core.engine import LLMEngine
from .quant import QuantConfig, quantize_model

__version__ = "0.3.0"
__all__ = ["EngineConfig", "ModelConfig", "SamplingParams", "LLMEngine", "QuantConfig", "quantize_model"]
