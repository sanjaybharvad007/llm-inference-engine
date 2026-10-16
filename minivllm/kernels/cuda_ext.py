"""Loader for the compiled CUDA extension (``csrc/``).

Two ways to get it:
  * ``pip install -e .`` with nvcc on PATH builds ``minivllm._C`` ahead of time (setup.py);
  * ``MINIVLLM_JIT=1`` compiles it on first use with ``torch.utils.cpp_extension.load``.
Without either, the Triton kernel is used and nothing here is touched.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_ext = None
_tried = False
_CSRC = Path(__file__).resolve().parents[2] / "csrc"


def load_ext():
    global _ext, _tried
    if _ext is not None:
        return _ext
    if _tried:
        raise RuntimeError("minivllm CUDA extension is not available")
    _tried = True
    try:
        from minivllm import _C  # type: ignore[attr-defined]

        _ext = _C
        return _ext
    except ImportError:
        pass
    if os.environ.get("MINIVLLM_JIT", "0") == "1" and torch.cuda.is_available():
        from torch.utils.cpp_extension import load

        _ext = load(
            name="minivllm_C", sources=[str(_CSRC / "paged_attention.cu")], extra_include_paths=[str(_CSRC)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"], verbose=os.environ.get("MINIVLLM_JIT_VERBOSE") == "1",
        )
        return _ext
    raise RuntimeError("minivllm CUDA extension not built (pip install with nvcc, or MINIVLLM_JIT=1)")


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        load_ext()
        return True
    except Exception:
        return False
