"""Command line: ``python -m minivllm serve --model <hf-path>`` / ``--tiny``."""
from __future__ import annotations

import argparse
import logging

import torch


def main(argv=None):
    p = argparse.ArgumentParser(prog="minivllm")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the OpenAI-compatible server")
    s.add_argument("--model", help="path to a HuggingFace Llama-style checkpoint")
    s.add_argument("--tiny", action="store_true", help="random tiny model (demo / smoke test)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    s.add_argument("--block-size", type=int, default=16)
    s.add_argument("--max-num-seqs", type=int, default=256)
    s.add_argument("--max-num-batched-tokens", type=int, default=2048)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    s.add_argument("--num-gpu-blocks", type=int, default=None)
    s.add_argument("--no-cuda-graphs", action="store_true")
    s.add_argument("--no-prefix-caching", action="store_true")
    s.add_argument("--no-chunked-prefill", action="store_true")
    s.add_argument("--speculative-ngram", type=int, default=0, help="draft tokens per step (0 = off)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    from .config import EngineConfig
    from .core.engine import LLMEngine
    from .server.api import create_app

    cfg = EngineConfig(
        block_size=args.block_size, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks=args.num_gpu_blocks, use_cuda_graphs=not args.no_cuda_graphs,
        enable_prefix_caching=not args.no_prefix_caching, enable_chunked_prefill=not args.no_chunked_prefill,
        speculative_ngram_max=args.speculative_ngram,
    )
    if args.tiny:
        cfg.num_gpu_blocks = cfg.num_gpu_blocks or 512
        cfg.use_cuda_graphs = False
        engine = LLMEngine.tiny(cfg)
        name = "tiny-random"
    else:
        if not args.model:
            p.error("--model or --tiny is required")
        engine = LLMEngine.from_pretrained(args.model, cfg, dtype=getattr(torch, args.dtype))
        name = args.model
    import uvicorn

    uvicorn.run(create_app(engine, name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
