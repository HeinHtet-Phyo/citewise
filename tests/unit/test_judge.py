"""Unit tests for eval/judge.py — mock LiteLLM so no API calls are made."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))

from judge import Judgement, _build_judge_message, _parse_judge_response, judge_answer


def test_parse_judge_response_valid() -> None:
    text = '{"faithfulness": 3, "relevance": 2, "citation_quality": 3, "rationale": "Good."}'
    result = _parse_judge_response(text)
    assert result["faithfulness"] == 3
    assert result["relevance"] == 2


def test_parse_judge_response_with_preamble() -> None:
    text = 'Here is my evaluation:\n{"faithfulness": 2, "relevance": 2, "citation_quality": 1, "rationale": "OK."}'
    result = _parse_judge_response(text)
    assert result["faithfulness"] == 2


def test_parse_judge_response_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _parse_judge_response("No JSON here.")


def test_build_judge_message_format() -> None:
    msg = _build_judge_message("What is SSP?", ["Source A", "Source B"], "SSP is £109 per week [1].")
    assert "Question: What is SSP?" in msg
    assert "[1] Source A" in msg
    assert "[2] Source B" in msg
    assert "SSP is £109 per week" in msg


def _make_mock_response(json_str: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = json_str
    return response


def test_judge_answer_calls_llm_and_parses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("judge.settings.llm_cache_enabled", False)
    monkeypatch.setattr(
        "judge.litellm.completion",
        lambda **kw: _make_mock_response(
            '{"faithfulness": 3, "relevance": 3, "citation_quality": 2, "rationale": "Accurate."}'
        ),
    )
    result = judge_answer(
        question_id="q001",
        question="Am I entitled to SSP?",
        sources=["Statutory Sick Pay is paid at £109.40 per week."],
        answer="You are entitled to SSP at £109.40 per week [1].",
        refused=False,
    )
    assert isinstance(result, Judgement)
    assert result.faithfulness == 3
    assert result.relevance == 3
    assert result.citation_quality == 2
    assert not result.refused


def test_judge_answer_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("judge.settings.llm_cache_enabled", True)
    monkeypatch.setattr("judge.settings.llm_cache_dir", str(tmp_path))

    cached_text = '{"faithfulness": 2, "relevance": 2, "citation_quality": 2, "rationale": "Cached."}'
    from judge import _prompt_hash, _JUDGE_MODEL
    messages = [
        {"role": "system", "content": "stub"},
        {"role": "user", "content": "stub"},
    ]

    # Pre-populate cache with a known response
    import hashlib, json as _json
    real_hash = _prompt_hash

    llm_called = {"count": 0}

    def fake_hash(model: str, msgs: list) -> str:
        return "fakehash"

    monkeypatch.setattr("judge._prompt_hash", fake_hash)

    cache_file = tmp_path / "fakehash.json"
    cache_file.write_text(_json.dumps({"text": cached_text}), encoding="utf-8")

    mock_llm = MagicMock(side_effect=AssertionError("LLM should not be called when cache is warm"))
    monkeypatch.setattr("judge.litellm.completion", mock_llm)

    result = judge_answer(
        question_id="q001",
        question="test",
        sources=["test source"],
        answer="test answer",
        refused=False,
    )
    assert result.faithfulness == 2
    mock_llm.assert_not_called()
