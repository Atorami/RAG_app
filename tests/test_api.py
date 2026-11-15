import pytest
from fastapi.testclient import TestClient

from manual_assistant import api
from manual_assistant.pipeline import ManualAssistant

from conftest import NaiveLLM


@pytest.fixture
def client(monkeypatch, clean_index, embedder):
    fake = ManualAssistant(clean_index, embedder, NaiveLLM(), top_k=4, min_score=0.1)
    monkeypatch.setattr(api, "get_assistant", lambda: fake)
    api._requests.clear()
    return TestClient(api.app)


def test_ask_returns_answer_and_sources(client):
    res = client.post("/ask", json={"question": "What does alarm F-0021 mean?"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "answered"
    assert body["sources"]


def test_rejects_overlong_question(client):
    res = client.post("/ask", json={"question": "x" * 501})
    assert res.status_code == 422


def test_rate_limit_kicks_in(monkeypatch, clean_index, embedder):
    fake = ManualAssistant(clean_index, embedder, NaiveLLM(), top_k=4, min_score=0.1)
    monkeypatch.setattr(api, "get_assistant", lambda: fake)
    monkeypatch.setattr(api, "_rate_limited", _limiter(2))
    client = TestClient(api.app)
    codes = [client.post("/ask", json={"question": "What does alarm F-0021 mean?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def _limiter(limit):
    calls = {"n": 0}

    def limited(_client):
        calls["n"] += 1
        return calls["n"] > limit

    return limited


def test_cors_does_not_allow_arbitrary_origins(client):
    res = client.options(
        "/ask",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert res.headers.get("access-control-allow-origin") != "https://evil.example"
