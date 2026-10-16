"""LLMEngine: the step loop that ties scheduler, model runner and sampler together."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from typing import Sequence as Seq

import torch

from ..config import EngineConfig, ModelConfig, SamplingParams
from ..distributed import parallel_state as ps
from ..models.llama import LlamaForCausalLM
from ..runner.model_runner import ModelRunner
from ..tokenizer import ByteTokenizer, HFTokenizer
from .block_manager import BlockManager
from .draft_model import DraftModelProposer
from .sampler import sample_one
from .scheduler import Scheduler
from .sequence import RequestOutput, Sequence, SequenceStatus
from .speculative import propose_ngram, verify

log = logging.getLogger("minivllm")


@dataclass
class EngineStats:
    num_requests: int = 0
    num_finished: int = 0
    num_steps: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    spec_proposed: int = 0
    spec_accepted: int = 0
    ttft_s: list[float] = field(default_factory=list)
    e2e_s: list[float] = field(default_factory=list)

    @property
    def spec_acceptance_rate(self) -> float:
        return self.spec_accepted / max(self.spec_proposed, 1)


class LLMEngine:
    def __init__(self, model: LlamaForCausalLM, tokenizer, config: EngineConfig,
                 draft_model: Optional[LlamaForCausalLM] = None):
        self.model = model.eval().to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        num_blocks = config.num_gpu_blocks or self._derive_num_blocks()
        num_blocks = ps.broadcast_object(num_blocks, src=0)   # all TP ranks must agree
        self.kv_caches = self.model.allocate_kv_caches(num_blocks, config.block_size, self.device)
        self.block_manager = BlockManager(num_blocks, config.block_size, config.enable_prefix_caching)
        self.scheduler = Scheduler(config, self.block_manager)
        self.runner = ModelRunner(self.model, config, self.block_manager, self.kv_caches)
        self.runner.capture_graphs()
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.proposer: Optional[DraftModelProposer] = None
        if draft_model is not None and self.is_driver:
            self.proposer = DraftModelProposer(draft_model, config.draft_num_blocks, config.block_size,
                                               config.device, config.speculative_max_draft)
        self.stats = EngineStats()
        self._counter = 0
        log.info("engine ready: %d KV blocks x %d tokens (%.1f MB), cuda graphs=%s", num_blocks, config.block_size,
                 num_blocks * self.model.kv_bytes_per_block(config.block_size) / 2**20, bool(self.runner.graphs))

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_pretrained(cls, path: str, config: Optional[EngineConfig] = None, dtype=torch.float16) -> "LLMEngine":
        config = config or EngineConfig()
        model = LlamaForCausalLM.from_hf(path, dtype=dtype, device=config.device)
        draft = None
        if config.speculative_draft_model:
            with ps.replicated():
                draft = LlamaForCausalLM.from_hf(config.speculative_draft_model, dtype=dtype, device=config.device)
        return cls(model, HFTokenizer(path), config, draft_model=draft)

    @classmethod
    def tiny(cls, config: Optional[EngineConfig] = None, seed: int = 0, **model_kwargs) -> "LLMEngine":
        """Random-weight tiny model with a byte tokenizer (tests / demos)."""
        torch.manual_seed(seed)
        mcfg = ModelConfig.tiny(vocab_size=ByteTokenizer.vocab_size, **model_kwargs)
        config = config or EngineConfig(num_gpu_blocks=256, use_cuda_graphs=False, device="cpu")
        return cls(LlamaForCausalLM(mcfg), ByteTokenizer(), config)

    @property
    def is_driver(self) -> bool:
        return ps.tp_rank() == 0

    def worker_loop(self) -> None:
        """Tensor-parallel worker ranks block here, executing the driver's steps."""
        self.runner.worker_loop()

    def shutdown(self) -> None:
        """Driver: release the worker ranks."""
        if ps.get_tp().enabled and self.is_driver:
            ps.broadcast_object(None, src=0)

    def _derive_num_blocks(self) -> int:
        per_block = self.model.kv_bytes_per_block(self.config.block_size)
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            budget = free * self.config.gpu_memory_utilization - 512 * 2**20   # headroom for activations
            return max(int(budget // per_block), 16)
        return 512

    # -- requests ----------------------------------------------------------------
    def add_request(self, request_id: Optional[str], prompt, params: SamplingParams) -> str:
        if request_id is None:
            self._counter += 1
            request_id = f"req-{self._counter}"
        ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if len(ids) >= self.config.max_model_len:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds max_model_len={self.config.max_model_len}")
        seq = Sequence(request_id, ids, params)
        self.scheduler.add(seq)
        self.stats.num_requests += 1
        self.stats.prompt_tokens += len(ids)
        return request_id

    def abort(self, request_id: str) -> bool:
        if self.proposer is not None:
            self.proposer.free(request_id)
        return self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    # -- the step --------------------------------------------------------------------
    def step(self) -> list[RequestOutput]:
        assert self.is_driver, "only tensor-parallel rank 0 drives the engine; workers call worker_loop()"
        out = self.scheduler.schedule()
        if out.is_empty:
            return []
        logits = self.runner.execute(out)
        self.stats.num_steps += 1
        results: list[RequestOutput] = []
        row = 0
        now = time.perf_counter()
        for ss in out.scheduled:
            seq = ss.seq
            prev_computed = seq.num_computed_tokens
            seq.num_computed_tokens += ss.num_new_tokens
            if not ss.samples:
                continue  # prefill chunk, prompt not finished yet
            params = seq.sampling_params
            if seq.draft_token_ids and not ss.is_prompt_chunk:
                nrows = 1 + len(seq.draft_token_ids)
                new_tokens, n_acc = verify(seq, logits[row:row + nrows], params, self.generator)
                row += nrows
                self.stats.spec_proposed += len(seq.draft_token_ids)
                self.stats.spec_accepted += n_acc
                seq.stats_proposed_drafts += len(seq.draft_token_ids)
                seq.stats_accepted_drafts += n_acc
                # KV of rejected drafts is garbage: only the verified prefix counts as computed.
                seq.num_computed_tokens = prev_computed + 1 + n_acc
                seq.draft_token_ids = []
            else:
                new_tokens = [sample_one(logits[row], params, seq.token_ids, self.generator)]
                row += 1
            if seq.first_token_time is None:
                seq.first_token_time = now
                self.stats.ttft_s.append(now - seq.arrival_time)

            emitted: list[int] = []
            finish: Optional[SequenceStatus] = None
            for tok in new_tokens:
                seq.append_token(tok)
                emitted.append(tok)
                self.stats.generated_tokens += 1
                if (tok == self.model.cfg.eos_token_id and not params.ignore_eos) or tok in params.stop_token_ids:
                    finish = SequenceStatus.FINISHED_STOPPED
                    break
                if seq.num_output_tokens >= params.max_tokens or seq.num_tokens >= self.config.max_model_len:
                    finish = SequenceStatus.FINISHED_LENGTH
                    break
            if finish is None:
                # tokens beyond the last appended-but-uncomputed one are not in the cache
                seq.num_computed_tokens = min(seq.num_computed_tokens, seq.num_tokens - 1)
            self.block_manager.commit_full_blocks(seq)

            if finish is not None:
                self.scheduler.finish_sequence(seq, finish)
                if self.proposer is not None:
                    self.proposer.free(seq.request_id)
                self.stats.num_finished += 1
                self.stats.e2e_s.append(time.perf_counter() - seq.arrival_time)
            elif self.proposer is not None or self.config.speculative_ngram_max > 0:
                limit = self.config.speculative_max_draft if self.proposer is not None else self.config.speculative_ngram_max
                room = min(limit, params.max_tokens - seq.num_output_tokens - 1,
                           self.config.max_model_len - seq.num_tokens - 2)
                if self.proposer is not None:
                    seq.draft_token_ids = self.proposer.propose(seq, room)
                else:
                    seq.draft_token_ids = propose_ngram(seq, self.config.speculative_ngram_prompt_lookup, room)
            results.append(RequestOutput(
                request_id=seq.request_id, new_token_ids=emitted, output_token_ids=list(seq.output_token_ids),
                finished=finish is not None, finish_reason=_reason(finish), num_cached_tokens=seq.num_cached_tokens,
            ))
        return results

    # -- offline convenience -----------------------------------------------------------
    def generate(self, prompts: Seq, params: SamplingParams | Seq[SamplingParams]) -> list[list[int]]:
        """Run all prompts to completion; returns output token ids per prompt (in order)."""
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        ids = [self.add_request(None, p, sp) for p, sp in zip(prompts, params)]
        outputs: dict[str, list[int]] = {}
        while self.has_unfinished():
            for r in self.step():
                if r.finished:
                    outputs[r.request_id] = r.output_token_ids
        return [outputs[i] for i in ids]

    def generate_text(self, prompts: Seq[str], params: SamplingParams) -> list[str]:
        return [self.tokenizer.decode(o) for o in self.generate(prompts, params)]

    # -- observability -----------------------------------------------------------------
    def snapshot(self) -> dict:
        bm = self.block_manager.stats
        return {
            "running": self.scheduler.num_running, "waiting": self.scheduler.num_waiting,
            "kv_usage": bm.usage, "prefix_hit_rate": bm.hit_rate, "evictions": bm.evictions,
            "preemptions": self.scheduler.num_preemptions, "steps": self.stats.num_steps,
            "generated_tokens": self.stats.generated_tokens, "graph_replays": self.runner.stats.num_graph_replays,
            "spec_acceptance_rate": self.stats.spec_acceptance_rate,
        }


def _reason(status: Optional[SequenceStatus]) -> Optional[str]:
    if status is None:
        return None
    return {"finished_stopped": "stop", "finished_length": "length", "finished_aborted": "abort"}.get(status.value)
