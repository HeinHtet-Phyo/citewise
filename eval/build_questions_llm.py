"""Nebius LLM drafting for eval questions, with a file cache and spend tracking."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import litellm

from citewise.config import settings

MODEL = "openai/Qwen/Qwen3-235B-A22B-Instruct-2507"
_PRICE_IN, _PRICE_OUT = 0.20e-6, 0.60e-6  # USD per token, as in citewise.generate
_CACHE = Path(".llm_cache/eval_questions")
litellm.suppress_debug_info = True


class Spend:
    """Accumulates token usage and USD cost across calls (cache hits cost nothing)."""

    calls = 0
    cached = 0
    usd = 0.0


def _extract_json(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def draft_json(system: str, user: str) -> dict[str, object] | None:
    """Call the model (temperature 0) and parse a JSON object from the reply."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    key = hashlib.sha256(json.dumps([MODEL, messages]).encode()).hexdigest()
    path = _CACHE / f"{key}.json"
    if path.exists():
        Spend.cached += 1
        return _extract_json(json.loads(path.read_text(encoding="utf-8"))["text"])
    resp = litellm.completion(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=700,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        timeout=120,
        num_retries=3,
        extra_body={"enable_thinking": False},
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    Spend.calls += 1
    Spend.usd += (usage.prompt_tokens or 0) * _PRICE_IN + (
        usage.completion_tokens or 0
    ) * _PRICE_OUT
    _CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text}), encoding="utf-8")
    return _extract_json(text)
