"""LLM generation with LiteLLM, file-based caching, and cost tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import litellm
from litellm.exceptions import RateLimitError, ServiceUnavailableError, Timeout

from citewise.config import RagConfig, settings
from citewise.retrieve import RetrievedChunk

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "answer_system.txt").read_text(encoding="utf-8").strip()

# Approximate USD → GBP conversion used for cost estimates.
_USD_TO_GBP = 0.79

# litellm is chatty at INFO; suppress unless debugging.
litellm.suppress_debug_info = True
# Some newer models (e.g. gemini-3.8-flash) don't support all params; drop silently.
litellm.drop_params = True


def _cache_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def _prompt_hash(model: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format numbered source passages followed by the question."""
    parts: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        heading = f" — {rc.chunk.section}" if rc.chunk.section else ""
        parts.append(f"[{i}] {rc.chunk.title}{heading}\n{rc.chunk.text}")
    sources = "\n\n".join(parts)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    config: RagConfig,
) -> tuple[str, float, int]:
    """Call the LLM and return (answer_text, cost_gbp, generation_latency_ms).

    Uses a file-based cache keyed on model + full prompt to avoid re-spending
    tokens on repeated questions during development.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(question, chunks)},
    ]

    cache_key = _prompt_hash(config.llm_model, messages)

    if settings.llm_cache_enabled:
        cached = _cache_path(settings.llm_cache_dir, cache_key)
        if cached.exists():
            log.debug("LLM cache hit: %s", cache_key[:12])
            data = json.loads(cached.read_text(encoding="utf-8"))
            return data["text"], data["cost_gbp"], 0

    t0 = time.monotonic()
    _backoff = 10.0
    for _attempt in range(3):
        try:
            response = litellm.completion(
                model=config.llm_model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=settings.gemini_api_key or None,
                timeout=60,
            )
            break
        except (RateLimitError, ServiceUnavailableError, Timeout) as exc:
            if _attempt == 2:
                raise
            log.warning(
                "Gemini rate-limit/503 (attempt %d/3) — backing off %.0fs: %s",
                _attempt + 1,
                _backoff,
                exc,
            )
            time.sleep(_backoff)
            _backoff = min(_backoff * 2, 20.0)
    latency_ms = int((time.monotonic() - t0) * 1000)

    text: str = response.choices[0].message.content or ""

    try:
        cost_usd: float = litellm.completion_cost(completion_response=response)
    except Exception:
        cost_usd = 0.0
    cost_gbp = round(cost_usd * _USD_TO_GBP, 6)

    log.info("LLM %s: %d ms, $%.6f (£%.6f)", config.llm_model, latency_ms, cost_usd, cost_gbp)

    if settings.llm_cache_enabled:
        cache_dir = Path(settings.llm_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(settings.llm_cache_dir, cache_key).write_text(
            json.dumps({"text": text, "cost_gbp": cost_gbp}),
            encoding="utf-8",
        )

    return text, cost_gbp, latency_ms
