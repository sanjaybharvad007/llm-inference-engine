"""Asyncio wrapper: one background loop drives ``engine.step()`` while there are
requests; each request gets an async queue of incremental outputs."""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import AsyncIterator, Optional

from ..config import SamplingParams
from ..core.engine import LLMEngine
from ..core.sequence import RequestOutput

log = logging.getLogger("minivllm.async")


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._queues: dict[str, asyncio.Queue] = {}
        self._loop_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._wake = asyncio.Event()

    async def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self.engine.has_unfinished():
                self._wake.clear()
                await self._wake.wait()
            # The forward pass releases the GIL in torch ops; run it in a worker
            # thread so the event loop keeps streaming tokens to clients.
            outputs = await loop.run_in_executor(None, self._step_locked)
            for out in outputs:
                q = self._queues.get(out.request_id)
                if q is not None:
                    q.put_nowait(out)
                    if out.finished:
                        self._queues.pop(out.request_id, None)
            await asyncio.sleep(0)

    def _step_locked(self) -> list[RequestOutput]:
        with self._lock:
            return self.engine.step()

    async def generate(self, prompt, params: SamplingParams, request_id: Optional[str] = None) -> AsyncIterator[RequestOutput]:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            rid = self.engine.add_request(request_id, prompt, params)
        self._queues[rid] = q
        self._wake.set()
        try:
            while True:
                out = await q.get()
                yield out
                if out.finished:
                    return
        finally:
            if rid in self._queues:  # client disconnected mid-stream
                self._queues.pop(rid, None)
                with self._lock:
                    self.engine.abort(rid)

    def snapshot(self) -> dict:
        return self.engine.snapshot()
