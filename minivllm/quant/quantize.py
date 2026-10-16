"""Weight-only INT8 / FP8 quantization for decode-bound inference.

Decode is memory-bandwidth bound: every step streams the whole weight matrix
for a handful of tokens.  Storing weights in 8 bits halves that traffic
(≈2x decode throughput at batch 1) while activations stay in fp16/bf16, so
no calibration data is needed and accuracy loss is negligible for 7B+ models.

Two formats, both **per-output-channel symmetric** (one fp32 scale per row):

* ``int8``:  w_q = round(w / s), s = max|w| / 127            (any GPU, any CPU)
* ``fp8``:   w_q = (w / s).to(float8_e4m3fn), s = max|w| / 448 (Ada/Hopper native)

The matmul is ``y = x @ dequant(w_q)^T`` where dequant fuses into the GEMM
inner loop in the Triton kernel (``w8a16_gemm``), so the 8-bit weight is never
materialised at 16 bits.  ``QuantLinear`` keeps the same interface (and TP
sharding) as the linear layers it replaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..distributed import parallel_state as ps
from ..distributed.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelLinear

INT8_MAX = 127.0
FP8_E4M3_MAX = 448.0
HAS_FP8 = hasattr(torch, "float8_e4m3fn")


@dataclass(frozen=True)
class QuantConfig:
    method: str = "int8"           # "int8" | "fp8"
    skip: tuple[str, ...] = ("lm_head",)   # module-name suffixes left in full precision
    min_features: int = 64         # don't bother quantizing tiny layers (tests)

    def __post_init__(self):
        if self.method not in ("int8", "fp8"):
            raise ValueError(f"unknown quant method {self.method!r}")
        if self.method == "fp8" and not HAS_FP8:
            raise RuntimeError("this torch build has no float8_e4m3fn dtype")


def quantize_weight(w: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    """w: [out, in] -> (w_q [out, in] int8/fp8, scale [out] fp32)."""
    wf = w.float()
    amax = wf.abs().amax(dim=1).clamp_min(1e-8)
    if method == "int8":
        scale = amax / INT8_MAX
        wq = torch.round(wf / scale[:, None]).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    else:
        scale = amax / FP8_E4M3_MAX
        wq = (wf / scale[:, None]).clamp_(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return wq, scale


def dequantize_weight(wq: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (wq.to(torch.float32) * scale[:, None]).to(dtype)


class QuantLinear(nn.Module):
    """Weight-only 8-bit linear with the same (possibly tensor-parallel) semantics
    as the layer it replaced: ``reduce`` all-reduces the output (RowParallel),
    ``gather`` all-gathers it along the vocab dim (VocabParallel)."""

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, act_dtype: torch.dtype,
                 reduce: bool = False, gather: bool = False, method: str = "int8"):
        super().__init__()
        self.register_buffer("weight", wq)             # [out, in] int8 / fp8
        self.register_buffer("scale", scale.float())   # [out]
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None
        self.act_dtype, self.reduce, self.gather, self.method = act_dtype, reduce, gather, method
        self.out_features, self.in_features = wq.shape

    @staticmethod
    def _use_triton(x: torch.Tensor) -> bool:
        from .kernels import HAS_TRITON

        return HAS_TRITON and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_triton(x):
            from .kernels import w8a16_gemm

            y = w8a16_gemm(x, self.weight, self.scale)
        else:
            y = F.linear(x, dequantize_weight(self.weight, self.scale, x.dtype))
        if self.reduce:
            y = ps.all_reduce(y)
        if self.gather:
            y = ps.all_gather_last_dim(y)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, {self.method}, reduce={self.reduce}, gather={self.gather}"


_QUANTIZABLE = (ColumnParallelLinear, RowParallelLinear, VocabParallelLinear, nn.Linear)


@torch.no_grad()
def quantize_model(model: nn.Module, cfg: QuantConfig) -> dict[str, int]:
    """Replace every linear projection in-place with a QuantLinear.  Returns stats.

    Works on an already TP-sharded model: each rank quantizes its own shard,
    and scales are per-row so sharding commutes with quantization."""
    act_dtype = getattr(getattr(model, "cfg", None), "dtype", torch.float16)
    stats = {"quantized": 0, "skipped": 0, "bytes_before": 0, "bytes_after": 0}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if not isinstance(child, _QUANTIZABLE):
                continue
            full = f"{name}.{child_name}" if name else child_name
            w = child.weight
            if any(full.endswith(s) or full.split(".")[-1] == s for s in cfg.skip) or min(w.shape) < cfg.min_features:
                stats["skipped"] += 1
                continue
            wq, scale = quantize_weight(w, cfg.method)
            q = QuantLinear(wq, scale, getattr(child, "bias", None), act_dtype,
                            reduce=isinstance(child, RowParallelLinear), gather=isinstance(child, VocabParallelLinear),
                            method=cfg.method)
            setattr(mod, child_name, q)
            stats["quantized"] += 1
            stats["bytes_before"] += w.numel() * w.element_size()
            stats["bytes_after"] += wq.numel() + scale.numel() * 4
    model.quant_config = cfg
    return stats
