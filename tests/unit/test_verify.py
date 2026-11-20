"""Unit tests for claim verification, sentence dropping and small-to-big expansion."""

from __future__ import annotations

import pytest

from citewise.config import RagConfig
from citewise.guardrails import _parse_unsupported, drop_sentences, split_sentences
from citewise.models import Chunk
from citewise.rag import answer_question
from citewise.retrieve import RetrievedChunk, build_expanded_text


def test_split_sentences_keeps_markers() -> None:
    assert split_sentences("You get £10 [1]. It lasts 28 weeks [2].") == [
        "You get £10 [1].",
        "It lasts 28 weeks [2].",
    ]


def test_parse_unsupported_ignores_bad_numbers_and_text() -> None:
    assert _parse_unsupported('Sure {"unsupported": [2, 9]}', 3) == [2]
    assert _parse_unsupported("no json here", 3) == []


def test_drop_sentences_removes_flagged() -> None:
    answer = "Rate is £10 [1]. Limit is £99 [1]."
    assert drop_sentences(answer, ["Limit is £99 [1]."]) == "Rate is £10 [1]."


def test_expand_respects_budget_and_order() -> None:
    sibs = [(4, "a", 100), (5, "b", 100), (6, "c", 100), (7, "d", 100)]
    assert build_expanded_text(5, sibs, 300) == "a\n\nb\n\nc"
    assert build_expanded_text(5, sibs, 100) == "b"


def _rc() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="p#s-0", url="https://www.gov.uk/p", title="T", section=None,
        text="Rate is £10.", format="answer", updated_at="2026-01-01", tokens=5,
    )  # fmt: skip
    return RetrievedChunk(chunk=chunk, score=0.9)


def _patch(monkeypatch: pytest.MonkeyPatch, texts: list[str], flags: list[list[str]]) -> None:
    gen = iter(texts)
    ver = iter(flags)
    monkeypatch.setattr("citewise.rag.retrieve", lambda q, c: [_rc()])
    monkeypatch.setattr(
        "citewise.rag.generate_answer", lambda q, ch, cfg, fb=None: (next(gen), 0.0, 1)
    )
    monkeypatch.setattr("citewise.rag.verify_claims", lambda a, s: (next(ver), 0.0))


def test_verifier_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, ["Rate is £10 [1]."], [[]])
    a = answer_question("What is the rate?", RagConfig(verify_claims=True))
    assert a.verifier_path == "passed" and not a.refused


def test_verifier_repairs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, ["Rate is £99 [1].", "Rate is £10 [1]."], [["Rate is £99 [1]."], []])
    a = answer_question("What is the rate?", RagConfig(verify_claims=True))
    assert a.verifier_path == "repaired" and a.answer == "Rate is £10 [1]."


def test_verifier_drops_then_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = "Rate is £99 [1]."
    _patch(monkeypatch, [bad, bad], [[bad], [bad]])
    a = answer_question("What is the rate?", RagConfig(verify_claims=True))
    assert a.refused and a.refusal_reason == "unsupported_claims"
    assert a.verifier_path == "refused"
