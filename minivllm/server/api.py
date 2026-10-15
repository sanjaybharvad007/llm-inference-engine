"""OpenAI-compatible HTTP API (``/v1/completions``, ``/v1/chat/completions``)
with server-sent-event streaming, plus ``/health`` and Prometheus ``/metrics``."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ..config import SamplingParams
from ..core.engine import LLMEngine
from . import metrics
from .async_engine import AsyncLLMEngine


class CompletionRequest(BaseModel):
    model: str = "minivllm"
    prompt: Union[str, list[int]]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stream: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)
    ignore_eos: bool = False
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "minivllm"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


def _params(r) -> SamplingParams:
    return SamplingParams(
        max_tokens=r.max_tokens, temperature=r.temperature, top_p=r.top_p, top_k=r.top_k,
        repetition_penalty=r.repetition_penalty, stop_token_ids=getattr(r, "stop_token_ids", []),
        ignore_eos=r.ignore_eos, seed=getattr(r, "seed", None),
    )


def _render_chat_prompt(tokenizer, messages: list[ChatMessage]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template([m.model_dump() for m in messages])
    return "".join(f"<|{m.role}|>\n{m.content}\n" for m in messages) + "<|assistant|>\n"


def create_app(engine: LLMEngine, model_name: str = "minivllm") -> FastAPI:
    aengine = AsyncLLMEngine(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await aengine.start()
        yield
        await aengine.stop()

    app = FastAPI(title="minivllm", lifespan=lifespan)
    app.state.engine = engine
    app.state.async_engine = aengine
    tok = engine.tokenizer

    @app.get("/health")
    async def health():
        return {"status": "ok", **aengine.snapshot()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "minivllm"}]}

    @app.get("/metrics")
    async def prom():
        metrics.update_gauges(aengine.snapshot())
        body, ctype = metrics.render()
        return Response(content=body, media_type=ctype)

    async def _run(prompt, params: SamplingParams, kind: str, stream: bool, request: Request):
        rid = f"{kind}-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        metrics.REQUESTS.labels(kind).inc()
        t0 = time.perf_counter()
        n_prompt = len(tok.encode(prompt) if isinstance(prompt, str) else prompt)
        metrics.PROMPT_TOKENS.inc(n_prompt)

        def chunk(delta_text: str, finish: Optional[str]):
            if kind == "chat":
                choice = {"index": 0, "delta": {"role": "assistant", "content": delta_text}, "finish_reason": finish}
                obj = "chat.completion.chunk"
            else:
                choice = {"index": 0, "text": delta_text, "finish_reason": finish}
                obj = "text_completion"
            return {"id": rid, "object": obj, "created": created, "model": model_name, "choices": [choice]}

        async def events():
            first, last_t = True, t0
            async for out in aengine.generate(prompt, params, rid):
                if await request.is_disconnected():
                    break
                now = time.perf_counter()
                if first:
                    metrics.TTFT.observe(now - t0)
                    first = False
                else:
                    metrics.TPOT.observe((now - last_t) / max(len(out.new_token_ids), 1))
                last_t = now
                metrics.GEN_TOKENS.inc(len(out.new_token_ids))
                text = tok.decode(out.new_token_ids)
                yield f"data: {json.dumps(chunk(text, out.finish_reason if out.finished else None))}\n\n"
                if out.finished:
                    metrics.FINISHED.labels(out.finish_reason or "unknown").inc()
                    metrics.E2E.observe(now - t0)
            yield "data: [DONE]\n\n"

        if stream:
            return StreamingResponse(events(), media_type="text/event-stream")

        tokens: list[int] = []
        finish = None
        first = True
        async for out in aengine.generate(prompt, params, rid):
            if first:
                metrics.TTFT.observe(time.perf_counter() - t0)
                first = False
            tokens = out.output_token_ids
            finish = out.finish_reason
        metrics.GEN_TOKENS.inc(len(tokens))
        metrics.FINISHED.labels(finish or "unknown").inc()
        metrics.E2E.observe(time.perf_counter() - t0)
        text = tok.decode(tokens)
        usage = {"prompt_tokens": n_prompt, "completion_tokens": len(tokens), "total_tokens": n_prompt + len(tokens)}
        if kind == "chat":
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
            return JSONResponse({"id": rid, "object": "chat.completion", "created": created, "model": model_name,
                                 "choices": [choice], "usage": usage})
        choice = {"index": 0, "text": text, "finish_reason": finish}
        return JSONResponse({"id": rid, "object": "text_completion", "created": created, "model": model_name,
                             "choices": [choice], "usage": usage})

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request):
        try:
            return await _run(req.prompt, _params(req), "completion", req.stream, request)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request):
        prompt = _render_chat_prompt(tok, req.messages)
        try:
            return await _run(prompt, _params(req), "chat", req.stream, request)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app
