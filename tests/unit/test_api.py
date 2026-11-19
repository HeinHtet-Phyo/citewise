"""Unit tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from citewise.api.main import app
from citewise.models import Answer


def _make_answer(refused: bool = False) -> Answer:
    from citewise.models import Citation
    return Answer(
        question="Am I entitled to SSP?",
        answer="You are entitled to SSP at £109.40 per week [1]." if not refused else "",
        citations=[
            Citation(
                n=1,
                title="Statutory Sick Pay",
                section="Overview",
                url="https://www.gov.uk/statutory-sick-pay",
                snippet="Statutory Sick Pay (SSP) is paid at £109.40 per week.",
                updated_at="2026-01-01",
            )
        ] if not refused else [],
        refused=refused,
        refusal_reason="low_evidence" if refused else None,
        config_name="hybrid_400_rerank",
        latency_ms=120,
        cost_gbp=0.0001,
        retrieved_chunk_ids=["statutory-sick-pay#overview-0"],
        trace_id="test-trace-id",
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_list_configs(client: TestClient) -> None:
    res = client.get("/configs")
    assert res.status_code == 200
    data = res.json()
    assert "configs" in data
    assert len(data["configs"]) >= 1


def test_ask_returns_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("citewise.api.main.answer_question", lambda q, c: _make_answer())
    res = client.post("/ask", json={"question": "Am I entitled to SSP?"})
    assert res.status_code == 200
    data = res.json()
    assert data["refused"] is False
    assert "[1]" in data["answer"]
    assert len(data["citations"]) == 1


def test_ask_refused_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("citewise.api.main.answer_question", lambda q, c: _make_answer(refused=True))
    res = client.post("/ask", json={"question": "What is the best pizza topping?"})
    assert res.status_code == 200
    assert res.json()["refused"] is True


def test_ask_invalid_config(client: TestClient) -> None:
    res = client.post("/ask", json={"question": "Am I entitled to SSP?", "config_name": "nonexistent"})
    assert res.status_code == 400


def test_ask_too_short_question(client: TestClient) -> None:
    res = client.post("/ask", json={"question": "hi"})
    assert res.status_code == 422


def test_index_serves_html(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert b"Citewise" in res.content
