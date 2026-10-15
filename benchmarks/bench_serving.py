"""Online serving benchmark against a running server: Poisson arrivals, measures
TTFT / inter-token latency / end-to-end latency percentiles and throughput.

    python -m minivllm serve --model ... &
    python benchmarks/bench_serving.py --url http://localhost:8000 --rate 4 --num-requests 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import httpx


async def one(client: httpx.AsyncClient, url: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    first, times = None, []
    async with client.stream("POST", f"{url}/v1/completions",
                             json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                                   "ignore_eos": True}, timeout=600) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            times.append(now)
            json.loads(line[6:])
    return {"ttft": first - t0, "e2e": times[-1] - t0, "tpot": (times[-1] - first) / max(len(times) - 1, 1)}


async def main_async(a):
    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "tensor", "cache", "block", "token", "kernel", "graph"]
    prompts = [" ".join(rng.choice(words) for _ in range(a.input_words)) for _ in range(a.num_requests)]
    results = []
    async with httpx.AsyncClient() as client:
        tasks = []
        t0 = time.perf_counter()
        for p in prompts:
            tasks.append(asyncio.create_task(one(client, a.url, p, a.max_tokens)))
            await asyncio.sleep(rng.expovariate(a.rate))
        results = await asyncio.gather(*tasks)
        total = time.perf_counter() - t0

    def pct(key, q):
        xs = sorted(r[key] for r in results)
        return xs[min(int(q * len(xs)), len(xs) - 1)] * 1e3

    print(f"requests: {len(results)}  rate: {a.rate}/s  total: {total:.1f}s  throughput: {len(results) * a.max_tokens / total:.1f} tok/s")
    for k in ("ttft", "tpot", "e2e"):
        print(f"  {k:5s} mean {statistics.mean(r[k] for r in results) * 1e3:8.1f} ms   p50 {pct(k, 0.5):8.1f}   p90 {pct(k, 0.9):8.1f}   p99 {pct(k, 0.99):8.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--rate", type=float, default=2.0, help="mean arrivals per second (Poisson)")
    p.add_argument("--num-requests", type=int, default=100)
    p.add_argument("--input-words", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=64)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
