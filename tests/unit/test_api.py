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
        ]
        if not refused
        else [],
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
    monkeypatch.setattr(
        "citewise.api.main.answer_question", lambda q, c: _make_answer(refused=True)
    )
    res = client.post("/ask", json={"question": "What is the best pizza topping?"})
    assert res.status_code == 200
    assert res.json()["refused"] is True


def test_ask_invalid_config(client: TestClient) -> None:
    res = client.post(
        "/ask", json={"question": "Am I entitled to SSP?", "config_name": "nonexistent"}
    )
    assert res.status_code == 400


def test_ask_too_short_question(client: TestClient) -> None:
    res = client.post("/ask", json={"question": "hi"})
    assert res.status_code == 422


def test_index_serves_html(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert b"Citewise" in res.content


def test_daily_budget_blocks_after_cap() -> None:
    from citewise.api.budget import DailyBudget

    budget = DailyBudget(cap_gbp=0.001)
    assert not budget.exhausted()
    budget.add(0.0006)
    assert not budget.exhausted()
    budget.add(0.0006)
    assert budget.exhausted()


def test_daily_budget_resets_at_utc_midnight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spend accumulated on day 1 is cleared when UTC date advances to day 2."""
    from citewise.api.budget import DailyBudget

    monkeypatch.setattr(DailyBudget, "_today", staticmethod(lambda: "2024-01-01"))
    budget = DailyBudget(cap_gbp=0.001)
    budget.add(0.001)
    assert budget.exhausted()

    monkeypatch.setattr(DailyBudget, "_today", staticmethod(lambda: "2024-01-02"))
    assert not budget.exhausted()
    assert budget._spent == 0.0
    assert budget._day == "2024-01-02"


def test_rate_limit_uses_forwarded_address() -> None:
    from starlette.requests import Request

    from citewise.api.main import _client_ip

    def make(headers: list[tuple[bytes, bytes]]) -> Request:
        return Request({"type": "http", "headers": headers, "client": ("10.0.0.1", 1)})

    assert _client_ip(make([(b"x-forwarded-for", b"203.0.113.7, 10.1.1.1")])) == "203.0.113.7"
    assert _client_ip(make([])) == "10.0.0.1"
