"""Offline throughput benchmark: tokens/s and requests/s vs. concurrency, with
the engine's features toggled individually (ablation).

    python benchmarks/bench_offline.py --model /path/to/llama --num-prompts 256 --input-len 512 --output-len 128
    python benchmarks/bench_offline.py --tiny          # CPU smoke run
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from minivllm.config import EngineConfig, SamplingParams
from minivllm.core.engine import LLMEngine


def run(engine: LLMEngine, prompts, out_len: int) -> dict:
    params = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
    t0 = time.perf_counter()
    outs = engine.generate(prompts, params)
    dt = time.perf_counter() - t0
    gen = sum(len(o) for o in outs)
    s = engine.stats
    return {
        "elapsed_s": dt, "requests_per_s": len(prompts) / dt, "gen_tokens_per_s": gen / dt,
        "total_tokens_per_s": (gen + s.prompt_tokens) / dt, "steps": s.num_steps,
        "mean_ttft_ms": 1e3 * sum(s.ttft_s) / max(len(s.ttft_s), 1),
        "p99_ttft_ms": 1e3 * sorted(s.ttft_s)[int(0.99 * (len(s.ttft_s) - 1))] if s.ttft_s else 0.0,
        "preemptions": engine.scheduler.num_preemptions, "prefix_hit_rate": engine.block_manager.stats.hit_rate,
        "graph_replays": engine.runner.stats.num_graph_replays, "spec_accept": s.spec_acceptance_rate,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--shared-prefix-len", type=int, default=0, help="tokens shared by all prompts (prefix caching)")
    p.add_argument("--max-num-seqs", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--ablate", action="store_true", help="also run with cuda graphs / prefix cache / chunked prefill off")
    a = p.parse_args()

    rng = random.Random(0)
    vocab = 250 if a.tiny else 30000
    shared = [rng.randrange(5, vocab) for _ in range(a.shared_prefix_len)]
    prompts = [shared + [rng.randrange(5, vocab) for _ in range(a.input_len - a.shared_prefix_len)] for _ in range(a.num_prompts)]

    def make(**kw):
        cfg = EngineConfig(max_model_len=a.input_len + a.output_len + 16, **kw)
        if a.tiny:
            cfg.device, cfg.use_cuda_graphs, cfg.num_gpu_blocks = "cpu", False, 2048
            return LLMEngine.tiny(cfg, hidden=128, layers=2, heads=4, kv_heads=2)
        return LLMEngine.from_pretrained(a.model, cfg, dtype=torch.float16)

    configs = [("default", {})]
    if a.ablate:
        configs += [("no_cuda_graphs", {"use_cuda_graphs": False}), ("no_prefix_cache", {"enable_prefix_caching": False}),
                    ("no_chunked_prefill", {"enable_chunked_prefill": False}), ("spec_ngram4", {"speculative_ngram_max": 4})]
    print(f"{'config':20s} {'max_seqs':>8s} {'req/s':>8s} {'gen tok/s':>10s} {'ttft ms':>8s} {'p99 ttft':>9s} {'preempt':>7s} {'hit':>5s}")
    for name, kw in configs:
        for ms in a.max_num_seqs:
            eng = make(max_num_seqs=ms, **kw)
            r = run(eng, prompts, a.output_len)
            print(f"{name:20s} {ms:8d} {r['requests_per_s']:8.2f} {r['gen_tokens_per_s']:10.1f} {r['mean_ttft_ms']:8.1f} "
                  f"{r['p99_ttft_ms']:9.1f} {r['preemptions']:7d} {r['prefix_hit_rate']:5.2f}")
            del eng


if __name__ == "__main__":
    main()
