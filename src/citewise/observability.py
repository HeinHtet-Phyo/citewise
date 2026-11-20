"""Optional Langfuse tracing: one trace per request, sent in the background, never fatal.

Traces go to Langfuse's OpenTelemetry endpoint as OTLP/JSON over httpx, so no extra SDK is
needed (the old Langfuse v2 ingestion API was shut down on 16 November 2026).

When the Langfuse keys are missing every function here is a cheap no-op. Spans and LLM calls
are collected in memory during the request; after the answer is built they are handed to a
single background thread that talks to Langfuse. All errors are caught and logged at debug level.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from citewise.config import settings
from citewise.models import Answer

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 500
MAX_ANSWER_CHARS = 2000

# Order matters: card numbers are removed before the looser phone pattern runs.
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I), "[NI_NUMBER]"),
    (re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), "[CARD_NUMBER]"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"(?<!\w)(?:\+44\s?|0)\d(?:[\s-]?\d){9,10}\b"), "[PHONE]"),
]


def redact(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Replace National Insurance, card, email and phone numbers, then cap the length."""
    for pattern, label in _REDACTIONS:
        text = pattern.sub(label, text)
    return text[:max_chars]


@dataclass
class _Span:
    name: str
    start: datetime
    end: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LlmCall:
    name: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cost_gbp: float
    latency_ms: int
    end: datetime


@dataclass
class TraceRecorder:
    """In-memory record of one request: spans, LLM calls and free-form metadata."""

    trace_id: str
    question: str
    started: datetime = field(default_factory=lambda: datetime.now(UTC))
    spans: list[_Span] = field(default_factory=list)
    llm_calls: list[_LlmCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_current: ContextVar[TraceRecorder | None] = ContextVar("citewise_trace", default=None)
_executor: ThreadPoolExecutor | None = None


def begin_trace(trace_id: str, question: str) -> TraceRecorder | None:
    """Start recording a request; returns None (and records nothing) when tracing is off."""
    if not settings.tracing_enabled:
        return None
    recorder = TraceRecorder(trace_id=trace_id, question=question)
    _current.set(recorder)
    return recorder


@contextmanager
def trace_span(name: str, **metadata: Any) -> Iterator[dict[str, Any]]:
    """Time a pipeline stage; the yielded dict can be filled with extra metadata."""
    recorder = _current.get()
    meta: dict[str, Any] = dict(metadata)
    start = datetime.now(UTC)
    try:
        yield meta
    finally:
        if recorder is not None:
            recorder.spans.append(_Span(name, start, datetime.now(UTC), meta))


def note(**metadata: Any) -> None:
    """Attach metadata (for example retrieved URLs) to the current trace."""
    recorder = _current.get()
    if recorder is not None:
        recorder.metadata.update(metadata)


def record_llm(
    name: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_gbp: float,
    latency_ms: int,
) -> None:
    """Record one LLM call (generation or claim verification) on the current trace."""
    recorder = _current.get()
    if recorder is not None:
        call = _LlmCall(
            name, model, input_tokens, output_tokens, cost_gbp, latency_ms, datetime.now(UTC)
        )
        recorder.llm_calls.append(call)


def finish_trace(recorder: TraceRecorder | None, answer: Answer) -> None:
    """Hand the finished request to the background sender; never raises, never blocks."""
    _current.set(None)
    if recorder is None:
        return
    try:
        global _executor
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langfuse")
        _executor.submit(_send, recorder, answer, datetime.now(UTC))
    except Exception as exc:
        log.debug("Could not queue trace: %s", exc)


def _attr(key: str, value: Any) -> dict[str, Any]:
    """Build one OTLP/JSON attribute; lists and dicts are sent as JSON strings."""
    if isinstance(value, bool):
        wrapped: dict[str, Any] = {"boolValue": value}
    elif isinstance(value, int):
        wrapped = {"intValue": str(value)}
    elif isinstance(value, float):
        wrapped = {"doubleValue": value}
    elif isinstance(value, str):
        wrapped = {"stringValue": value}
    else:
        wrapped = {"stringValue": json.dumps(value, default=str)}
    return {"key": key, "value": wrapped}


def _nanos(moment: datetime) -> str:
    """Return a datetime as OTLP unix nanoseconds (a string)."""
    return str(int(moment.timestamp() * 1_000_000_000))


def _span_id() -> str:
    """Return a random 16-hex-character span id."""
    return uuid.uuid4().hex[:16]


def _otlp_span(
    trace_hex: str,
    name: str,
    start: datetime,
    end: datetime,
    attributes: dict[str, Any],
    parent: str | None = None,
) -> dict[str, Any]:
    """Build one OTLP span dict."""
    span: dict[str, Any] = {
        "traceId": trace_hex,
        "spanId": _span_id(),
        "name": name,
        "kind": 1,
        "startTimeUnixNano": _nanos(start),
        "endTimeUnixNano": _nanos(end),
        "attributes": [_attr(k, v) for k, v in attributes.items() if v is not None],
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def build_otlp(recorder: TraceRecorder, answer: Answer, ended: datetime) -> dict[str, Any]:
    """Convert a finished request into an OTLP/JSON payload using Langfuse's attribute names."""
    trace_hex = recorder.trace_id
    status = "refused" if answer.refused else "answered"
    root_attrs: dict[str, Any] = {
        "langfuse.trace.name": "ask",
        "langfuse.trace.input": redact(recorder.question),
        "langfuse.trace.output": redact(answer.answer, MAX_ANSWER_CHARS),
        "langfuse.trace.tags": [answer.config_name, status],
        "langfuse.trace.metadata.config": answer.config_name,
        "langfuse.trace.metadata.latency_ms": answer.latency_ms,
        "langfuse.trace.metadata.cost_gbp": answer.cost_gbp,
        "langfuse.trace.metadata.refused": answer.refused,
        "langfuse.trace.metadata.refusal_reason": answer.refusal_reason,
        "langfuse.trace.metadata.verifier_path": answer.verifier_path,
        "langfuse.observation.type": "span",
        "langfuse.observation.input": redact(recorder.question),
        "langfuse.observation.output": redact(answer.answer, MAX_ANSWER_CHARS),
    }
    for key, value in recorder.metadata.items():
        root_attrs[f"langfuse.trace.metadata.{key}"] = value
    root = _otlp_span(trace_hex, "ask", recorder.started, ended, root_attrs)
    spans = [root]
    for sp in recorder.spans:
        attrs = {f"langfuse.observation.metadata.{k}": v for k, v in sp.metadata.items()}
        attrs["langfuse.observation.type"] = "span"
        spans.append(_otlp_span(trace_hex, sp.name, sp.start, sp.end, attrs, root["spanId"]))
    for call in recorder.llm_calls:
        usage = {"input": call.input_tokens, "output": call.output_tokens}
        attrs = {
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": call.model,
            "langfuse.observation.usage_details": json.dumps(
                {k: v for k, v in usage.items() if v is not None}
            ),
            "langfuse.observation.metadata.cost_gbp": call.cost_gbp,
            "langfuse.observation.metadata.latency_ms": call.latency_ms,
        }
        start = call.end - timedelta(milliseconds=call.latency_ms)
        spans.append(_otlp_span(trace_hex, call.name, start, call.end, attrs, root["spanId"]))
    scope = {"scope": {"name": "citewise"}, "spans": spans}
    resource = {"attributes": [_attr("service.name", "citewise")]}
    return {"resourceSpans": [{"resource": resource, "scopeSpans": [scope]}]}


def _send(recorder: TraceRecorder, answer: Answer, ended: datetime) -> None:
    """POST the trace to Langfuse's OpenTelemetry endpoint (runs in the background thread)."""
    try:
        token = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        resp = httpx.post(
            f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
            json=build_otlp(recorder, answer, ended),
            headers={"Authorization": "Basic " + base64.b64encode(token).decode()},
            timeout=10,
        )
        if resp.status_code >= 300:
            log.warning("Langfuse rejected trace: HTTP %d", resp.status_code)
    except Exception as exc:
        log.debug("Langfuse send failed: %s", type(exc).__name__)


def flush() -> None:
    """Wait for queued traces to be sent (used in scripts and at shutdown)."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
