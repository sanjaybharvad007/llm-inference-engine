import json

import pytest
from fastapi.testclient import TestClient

from minivllm.server.api import create_app

from conftest import make_engine


@pytest.fixture
def client():
    app = create_app(make_engine(), model_name="tiny")
    with TestClient(app) as c:
        yield c


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"


def test_completion_non_streaming(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5, "temperature": 0, "ignore_eos": True})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 5 and body["choices"][0]["finish_reason"] == "length"


def test_completion_streaming_sse(client):
    with client.stream("POST", "/v1/completions",
                       json={"prompt": "hello", "max_tokens": 4, "temperature": 0, "stream": True, "ignore_eos": True}) as r:
        assert r.status_code == 200
        events = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[6:]) for e in events[:-1]]
    assert len(chunks) == 4 and chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert all(c["object"] == "text_completion" for c in chunks)


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
                                                   "temperature": 0, "ignore_eos": True})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 3


def test_concurrent_requests_are_batched(client):
    import concurrent.futures as cf

    def go(i):
        return client.post("/v1/completions", json={"prompt": f"request {i}", "max_tokens": 6, "temperature": 0,
                                                     "ignore_eos": True}).json()["usage"]["completion_tokens"]

    with cf.ThreadPoolExecutor(8) as ex:
        assert list(ex.map(go, range(8))) == [6] * 8
    snap = client.get("/health").json()
    assert snap["generated_tokens"] >= 48


def test_metrics_endpoint(client):
    client.post("/v1/completions", json={"prompt": "m", "max_tokens": 2, "ignore_eos": True})
    text = client.get("/metrics").text
    assert "minivllm_requests_total" in text and "minivllm_time_to_first_token_seconds" in text


def test_bad_request_too_long(client):
    r = client.post("/v1/completions", json={"prompt": "x" * 1000, "max_tokens": 2})
    assert r.status_code == 400
