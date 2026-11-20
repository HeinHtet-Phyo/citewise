"""LLM-as-judge: one strict-JSON call per answer for correctness, faithfulness and safety.

The judge uses DeepSeek-V4.1-Flash (different family from the Qwen3 generator) to avoid
self-preference bias. Responses use the same file-based LLM cache as the RAG pipeline;
the prompt version is part of the cache key, so editing the prompt invalidates old entries.
Refused answers are never sent to the judge.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import litellm
from pydantic import BaseModel, ValidationError

from citewise.config import settings
from citewise.generate import _cache_path, _prompt_hash

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "src" / "citewise" / "prompts"
_JUDGE_SYSTEM = (_PROMPTS_DIR / "judge_combined.txt").read_text(encoding="utf-8").strip()

# Bump when judge_combined.txt or the message format changes (it is part of the cache key).
PROMPT_VERSION = "judge-v2"

# DeepSeek-V4.1-Flash — different family from the Qwen3 generator to avoid self-preference bias.
_JUDGE_MODEL = "openai/deepseek-ai/DeepSeek-V4.1-Flash-1017"
_EMPTY_RETRIES = 3
_PARSE_ATTEMPTS = 2
litellm.suppress_debug_info = True


class JudgeVerdict(BaseModel):
    """Validated judge output for one answer."""

    correctness: Literal["correct", "partly", "incorrect", "n/a"]
    correctness_reason: str = ""
    fully_faithful: bool
    unsupported_claims: list[str] = []
    made_up: bool
    made_up_reason: str = ""
    safe: bool = True
    safe_reason: str = ""


def build_judge_message(
    question: str,
    q_type: str | None,
    reference: str | None,
    sources: list[str],
    answer: str,
) -> str:
    """Format question, reference, numbered sources and answer for the judge."""
    numbered = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources)) or "(no sources)"
    return (
        f"Question: {question}\n"
        f"Question type: {q_type or 'unknown'}\n\n"
        f"Reference answer: {reference or '(none provided)'}\n\n"
        f"Sources:\n{numbered}\n\n"
        f"Answer: {answer}"
    )


def parse_verdict(text: str) -> JudgeVerdict:
    """Find the first valid JSON object in the text and validate it; raise ValueError if none."""
    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return JudgeVerdict.model_validate(obj)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
    raise ValueError(f"No valid judge JSON in response: {text[:200]!r} ({last_error})")


def _read_cache(key: str) -> JudgeVerdict | None:
    """Return a cached verdict, deleting the entry if it no longer parses."""
    if not settings.llm_cache_enabled:
        return None
    path = _cache_path(settings.llm_cache_dir, key)
    if not path.exists():
        return None
    try:
        return parse_verdict(json.loads(path.read_text(encoding="utf-8")).get("text", ""))
    except (ValueError, json.JSONDecodeError):
        log.warning("Cached judge response unparseable — re-calling LLM")
        path.unlink(missing_ok=True)
        return None


def _write_cache(key: str, text: str) -> None:
    """Store a valid judge response in the file cache."""
    if not settings.llm_cache_enabled:
        return
    Path(settings.llm_cache_dir).mkdir(parents=True, exist_ok=True)
    _cache_path(settings.llm_cache_dir, key).write_text(json.dumps({"text": text}), "utf-8")


def _call_llm(messages: list[dict[str, str]]) -> str:
    """Call the judge model, retrying on empty output."""
    for attempt in range(_EMPTY_RETRIES):
        response = litellm.completion(
            model=_JUDGE_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=4096,  # DeepSeek-V4.1-Flash spends ~600 tokens on internal reasoning
            base_url=settings.nebius_base_url or None,
            api_key=settings.nebius_api_key or None,
            timeout=90,
        )
        text: str = response.choices[0].message.content or ""
        if text.strip():
            return text
        log.warning("Judge returned empty response (attempt %d/%d)", attempt + 1, _EMPTY_RETRIES)
        time.sleep(3)
    return ""


def judge_answer(
    question: str,
    q_type: str | None,
    reference: str | None,
    sources: list[str],
    answer: str,
) -> JudgeVerdict | None:
    """Judge one answer; returns None after repeated failure (caller flags judge_error)."""
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": build_judge_message(question, q_type, reference, sources, answer),
        },
    ]
    key = _prompt_hash(f"{_JUDGE_MODEL}|{PROMPT_VERSION}", messages)
    cached = _read_cache(key)
    if cached is not None:
        return cached
    for attempt in range(_PARSE_ATTEMPTS):
        try:
            text = _call_llm(messages)
            verdict = parse_verdict(text)
        except Exception as exc:  # network, timeout, empty or malformed output
            log.warning("Judge attempt %d/%d failed: %s", attempt + 1, _PARSE_ATTEMPTS, exc)
            continue
        _write_cache(key, text)
        return verdict
    return None


def judge_record(row: dict[str, Any], question: dict[str, Any]) -> dict[str, Any] | None:
    """Judge one results row; None for refused rows (no call), judge_error dict on failure."""
    if row.get("refused"):
        return None
    sources = row.get("context_texts") or [c["snippet"] for c in row.get("citations") or []]
    verdict = judge_answer(
        question=row["question"],
        q_type=question.get("type"),
        reference=question.get("reference_answer"),
        sources=sources,
        answer=row["answer"],
    )
    if verdict is None:
        log.error("Judge failed for %s — flagged as judge_error", row.get("question_id"))
        return {"question_id": row.get("question_id"), "judge_error": True}
    return {"question_id": row.get("question_id"), "judge_error": False, **verdict.model_dump()}
