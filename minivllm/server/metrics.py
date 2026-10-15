"""Prometheus metrics (scraped at /metrics)."""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter("minivllm_requests_total", "Requests received", ["endpoint"])
FINISHED = Counter("minivllm_requests_finished_total", "Requests finished", ["reason"])
PROMPT_TOKENS = Counter("minivllm_prompt_tokens_total", "Prompt tokens processed")
GEN_TOKENS = Counter("minivllm_generation_tokens_total", "Tokens generated")
TTFT = Histogram("minivllm_time_to_first_token_seconds", "Time to first token",
                 buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
TPOT = Histogram("minivllm_time_per_output_token_seconds", "Inter-token latency",
                 buckets=(0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64))
E2E = Histogram("minivllm_e2e_latency_seconds", "Request latency",
                buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
RUNNING = Gauge("minivllm_num_requests_running", "Sequences running")
WAITING = Gauge("minivllm_num_requests_waiting", "Sequences waiting")
KV_USAGE = Gauge("minivllm_kv_cache_usage", "Fraction of KV blocks in use")
PREFIX_HIT = Gauge("minivllm_prefix_cache_hit_rate", "Prefix cache hit rate")
PREEMPTIONS = Gauge("minivllm_preemptions_total", "Preemptions so far")
SPEC_ACCEPT = Gauge("minivllm_spec_decode_acceptance_rate", "Speculative draft acceptance rate")


def update_gauges(snapshot: dict) -> None:
    RUNNING.set(snapshot["running"])
    WAITING.set(snapshot["waiting"])
    KV_USAGE.set(snapshot["kv_usage"])
    PREFIX_HIT.set(snapshot["prefix_hit_rate"])
    PREEMPTIONS.set(snapshot["preemptions"])
    SPEC_ACCEPT.set(snapshot["spec_acceptance_rate"])


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
