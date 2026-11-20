"""Unit tests for eval/judge.py — litellm is mocked so no API calls are made."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))

import judge

VALID = {
    "correctness": "correct",
    "correctness_reason": "Matches.",
    "fully_faithful": True,
    "unsupported_claims": [],
    "made_up": False,
    "made_up_reason": "",
    "safe": True,
    "safe_reason": "n/a",
}


def _response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = text
    return response


@pytest.fixture(autouse=True)
def _no_cache_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge.settings, "llm_cache_enabled", False)
    monkeypatch.setattr(judge.time, "sleep", lambda _s: None)


def _patch_llm(monkeypatch: pytest.MonkeyPatch, outputs: list[str]) -> list[int]:
    calls: list[int] = []

    def fake(**_kw: Any) -> MagicMock:
        calls.append(1)
        return _response(outputs[min(len(calls), len(outputs)) - 1])

    monkeypatch.setattr(judge.litellm, "completion", fake)
    return calls


def test_parse_valid() -> None:
    v = judge.parse_verdict(json.dumps(VALID))
    assert v.correctness == "correct" and v.fully_faithful is True


def test_parse_with_extra_text_and_fences() -> None:
    text = "Sure!\n```json\n" + json.dumps(VALID) + "\n```\nHope that helps."
    assert judge.parse_verdict(text).made_up is False


def test_parse_skips_invalid_object_then_finds_valid() -> None:
    text = '{"foo": 1} then ' + json.dumps(VALID)
    assert judge.parse_verdict(text).correctness == "correct"


@pytest.mark.parametrize(
    "text",
    ["No JSON here.", "{not json}", json.dumps({**VALID, "correctness": "great"}), "{}"],
)
def test_parse_invalid_raises(text: str) -> None:
    with pytest.raises(ValueError):
        judge.parse_verdict(text)


def test_build_message_contains_parts() -> None:
    msg = judge.build_judge_message("Q?", "how_to", "Ref", ["A", "B"], "Ans [1]")
    assert "Question: Q?" in msg and "[2] B" in msg and "Reference answer: Ref" in msg
    assert "(none provided)" in judge.build_judge_message("Q?", None, None, [], "A")


def test_judge_answer_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(monkeypatch, [json.dumps(VALID)])
    v = judge.judge_answer("Q", "single_fact", "ref", ["src"], "ans")
    assert v is not None and v.correctness == "correct"


def test_judge_answer_retries_once_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_llm(monkeypatch, ["garbage", json.dumps(VALID)])
    assert judge.judge_answer("Q", None, None, ["s"], "a") is not None
    assert len(calls) == 2


def test_judge_answer_gives_up_after_two_bad_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_llm(monkeypatch, ["garbage"])
    assert judge.judge_answer("Q", None, None, ["s"], "a") is None
    assert len(calls) == 2


def test_empty_output_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_llm(monkeypatch, ["", "", json.dumps(VALID)])
    assert judge.judge_answer("Q", None, None, ["s"], "a") is not None
    assert len(calls) == 3


def test_judge_record_skips_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_llm(monkeypatch, [json.dumps(VALID)])
    assert judge.judge_record({"refused": True, "question_id": "q1"}, {}) is None
    assert calls == []


def test_judge_record_flags_error_and_uses_context_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(**kw: Any) -> MagicMock:
        seen.append(kw["messages"][1]["content"])
        return _response("garbage")

    monkeypatch.setattr(judge.litellm, "completion", fake)
    row = {"question_id": "q1", "question": "Q", "answer": "A", "context_texts": ["FULL TEXT"]}
    out = judge.judge_record(row, {"type": "single_fact"})
    assert out == {"question_id": "q1", "judge_error": True}
    assert "FULL TEXT" in seen[0]


def test_judge_record_success_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_llm(monkeypatch, [json.dumps(VALID)])
    row = {"question_id": "q1", "question": "Q", "answer": "A", "citations": [{"snippet": "s"}]}
    out = judge.judge_record(row, {})
    assert out is not None and out["judge_error"] is False and out["made_up"] is False


def test_cache_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(judge.settings, "llm_cache_enabled", True)
    monkeypatch.setattr(judge.settings, "llm_cache_dir", str(tmp_path))
    calls = _patch_llm(monkeypatch, [json.dumps(VALID)])
    judge.judge_answer("Q", None, None, ["s"], "a")
    judge.judge_answer("Q", None, None, ["s"], "a")
    assert len(calls) == 1
