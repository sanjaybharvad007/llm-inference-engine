"""Decode-path micro-benchmarks for the features that target the memory-bound
decode step: attention backends, weight-only quantization, speculative decoding.

    python benchmarks/bench_decode.py attn   --dtype float16 --ctx 4096 --batch 1 8 32   # CUDA vs Triton paged attention
    python benchmarks/bench_decode.py gemm   --in 4096 --out 11008 --m 1 4 16           # W8A16 vs fp16 cuBLAS
    python benchmarks/bench_decode.py spec   --model /path/to/7b --draft-model /path/to/160m
    python benchmarks/bench_decode.py --tiny spec                                        # CPU smoke run
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from minivllm.config import EngineConfig, ModelConfig, SamplingParams
from minivllm.core.engine import LLMEngine
from minivllm.models.llama import LlamaForCausalLM
from minivllm.tokenizer import ByteTokenizer


def timeit(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_attn(args):
    from minivllm.kernels.paged_attention import paged_attention_cuda, paged_attention_reference, paged_attention_triton

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = getattr(torch, args.dtype)
    bs, hq, hkv, d = args.block_size, 32, 8, 128
    for batch in args.batch:
        nblk = batch * (args.ctx // bs) + 1
        k = torch.randn(nblk, bs, hkv, d, device=dev, dtype=dt)
        v = torch.randn_like(k)
        q = torch.randn(batch, hq, d, device=dev, dtype=dt)
        bt = (torch.arange(batch * (args.ctx // bs), device=dev).view(batch, -1) + 1).int()
        cu = torch.arange(batch + 1, dtype=torch.int32, device=dev)
        ctx = torch.full((batch,), args.ctx, dtype=torch.int32, device=dev)
        bytes_moved = 2 * batch * args.ctx * hkv * d * k.element_size()
        rows = {}
        if dev == "cuda":
            try:
                rows["cuda"] = timeit(lambda: paged_attention_cuda(q, k, v, bt, cu, ctx))
            except Exception as e:  # extension not built
                rows["cuda"] = f"n/a ({type(e).__name__})"
            rows["triton"] = timeit(lambda: paged_attention_triton(q, k, v, bt, cu, ctx, 1))
        rows["reference"] = timeit(lambda: paged_attention_reference(q, k, v, bt, cu, ctx), iters=3, warmup=1)
        for name, ms in rows.items():
            bw = f"{bytes_moved / ms / 1e6:7.0f} GB/s" if isinstance(ms, float) else ""
            print(f"batch={batch:3d} ctx={args.ctx} {name:10s} {ms if isinstance(ms, str) else f'{ms:8.3f} ms'} {bw}")


def bench_gemm(args):
    from minivllm.quant import QuantLinear, quantize_weight

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    w = torch.randn(args.out, args.inp, device=dev) * 0.02
    for method in ("int8", "fp8") if hasattr(torch, "float8_e4m3fn") else ("int8",):
        wq, scale = quantize_weight(w, method)
        ql = QuantLinear(wq, scale, None, dt, method=method).to(dev)
        full = torch.nn.Linear(args.inp, args.out, bias=False, device=dev, dtype=dt)
        for m in args.m:
            x = torch.randn(m, args.inp, device=dev, dtype=dt)
            t_full = timeit(lambda: full(x))
            t_q = timeit(lambda: ql(x))
            wbytes = args.out * args.inp
            print(f"M={m:3d} K={args.inp} N={args.out}  fp16 {t_full:7.3f} ms ({wbytes * 2 / t_full / 1e6:6.0f} GB/s)   "
                  f"{method} {t_q:7.3f} ms ({wbytes / t_q / 1e6:6.0f} GB/s)   speedup {t_full / t_q:4.2f}x")


def bench_spec(args):
    if args.tiny:
        torch.manual_seed(0)
        target = LlamaForCausalLM(ModelConfig.tiny(vocab_size=258, hidden=256, layers=4, heads=8, kv_heads=2))
        draft = LlamaForCausalLM(ModelConfig.tiny(vocab_size=258, hidden=64, layers=1, heads=2, kv_heads=1))
        tok = ByteTokenizer()
        cfg = EngineConfig(num_gpu_blocks=512, block_size=16, use_cuda_graphs=False, device="cpu", max_model_len=512)
        prompts = ["The quick brown fox jumps over the lazy dog. " * 3] * 8
    else:
        from minivllm.tokenizer import HFTokenizer

        target = LlamaForCausalLM.from_hf(args.model, dtype=torch.float16, device="cuda")
        draft = LlamaForCausalLM.from_hf(args.draft_model, dtype=torch.float16, device="cuda")
        tok = HFTokenizer(args.model)
        cfg = EngineConfig()
        prompts = ["Write a short story about a robot learning to paint:"] * 8
    params = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    for name, kw in (("plain", {}), ("ngram-4", {"speculative_ngram_max": 4}), ("draft-model-k4", {"draft": draft})):
        d = kw.pop("draft", None)
        c = EngineConfig(**{**cfg.__dict__, **kw})
        eng = LLMEngine(target, tok, c, draft_model=d)
        t0 = time.perf_counter()
        outs = eng.generate(prompts, params)
        dt = time.perf_counter() - t0
        gen = sum(len(o) for o in outs)
        print(f"{name:16s} {gen / dt:8.1f} tok/s   steps={eng.stats.num_steps:4d}   "
              f"accept={eng.stats.spec_acceptance_rate:5.1%}   drafts/step={eng.stats.spec_proposed / max(eng.stats.num_steps, 1):.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["attn", "gemm", "spec"])
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--model")
    p.add_argument("--draft-model")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--in", dest="inp", type=int, default=4096)
    p.add_argument("--out", type=int, default=11008)
    p.add_argument("--m", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--output-len", type=int, default=64)
    args = p.parse_args()
    if args.tiny:
        os.environ.setdefault("MINIVLLM_ATTN_BACKEND", "reference")
    {"attn": bench_attn, "gemm": bench_gemm, "spec": bench_spec}[args.what](args)


if __name__ == "__main__":
    main()
