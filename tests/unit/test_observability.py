"""Tests for the optional Langfuse tracing: redaction, no-op path and failure isolation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from citewise import observability as obs
from citewise.config import settings
from citewise.models import Answer


def _answer() -> Answer:
    return Answer(
        question="q",
        answer="a [1]",
        citations=[],
        refused=False,
        refusal_reason=None,
        config_name="test",
        latency_ms=1,
        cost_gbp=0.0,
        retrieved_chunk_ids=[],
        trace_id="t",
    )


def test_redacts_national_insurance_number() -> None:
    assert obs.redact("My NI is AB 12 34 56 C ok") == "My NI is [NI_NUMBER] ok"
    assert "[NI_NUMBER]" in obs.redact("ab123456c")


def test_redacts_card_numbers() -> None:
    assert obs.redact("card 4111 1111 1111 1111 now") == "card [CARD_NUMBER] now"
    assert obs.redact("4111-1111-1111-1111") == "[CARD_NUMBER]"


def test_redacts_email_and_phone() -> None:
    out = obs.redact("mail a.b@example.co.uk or call 07700 900123")
    assert "[EMAIL]" in out and "[PHONE]" in out
    assert "example" not in out and "900123" not in out


def test_leaves_normal_questions_and_truncates() -> None:
    assert obs.redact("What is the State Pension age?") == "What is the State Pension age?"
    assert len(obs.redact("x" * 2000)) == obs.MAX_TEXT_CHARS


def test_no_op_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    assert obs.begin_trace("t", "question") is None
    with obs.trace_span("x") as meta:
        meta["k"] = 1
    obs.note(a=1)
    obs.record_llm("llm.generation", "m", 1, 2, 0.1, 5)
    obs.finish_trace(None, _answer())
    assert obs._executor is None


def test_recorder_collects_spans_and_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "langfuse_public_key", "pk")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk")
    rec = obs.begin_trace("t", "question")
    assert rec is not None
    with obs.trace_span("retrieval", mode="dense"):
        pass
    obs.record_llm("llm.generation", "m", 10, 5, 0.01, 7)
    obs.note(retrieved_urls=["u"])
    assert [s.name for s in rec.spans] == ["retrieval"]
    assert rec.llm_calls[0].input_tokens == 10
    assert rec.metadata == {"retrieved_urls": ["u"]}
    obs._current.set(None)


def test_send_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr(obs.httpx, "post", boom)
    rec = obs.TraceRecorder(trace_id="t", question="q")
    obs._send(rec, _answer(), datetime.now(UTC))


def test_otlp_payload_has_spans_generation_and_redacted_input() -> None:
    rec = obs.TraceRecorder(trace_id="a" * 32, question="NI AB 12 34 56 C, what is SSP?")
    with obs.trace_span("retrieval", urls=["u"]):
        pass
    rec.spans.append(obs._Span("retrieval", rec.started, rec.started, {"urls": ["u"]}))
    rec.llm_calls.append(obs._LlmCall("llm.generation", "m", 10, 5, 0.01, 7, datetime.now(UTC)))
    payload = obs.build_otlp(rec, _answer(), datetime.now(UTC))
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["ask", "retrieval", "llm.generation"]
    root_attrs = {a["key"]: a["value"] for a in spans[0]["attributes"]}
    assert "[NI_NUMBER]" in root_attrs["langfuse.trace.input"]["stringValue"]
    gen_attrs = {a["key"]: a["value"] for a in spans[2]["attributes"]}
    assert gen_attrs["langfuse.observation.type"]["stringValue"] == "generation"
    assert spans[1]["parentSpanId"] == spans[0]["spanId"]
