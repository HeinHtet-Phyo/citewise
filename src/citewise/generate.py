"""LLM generation with OpenAI SDK, file-based caching, and cost tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from openai import APIStatusError, APITimeoutError, OpenAI, RateLimitError

from citewise.config import RagConfig, settings
from citewise.observability import record_llm
from citewise.retrieve import RetrievedChunk

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPTS = {
    "v1": (_PROMPTS_DIR / "answer_system.txt").read_text(encoding="utf-8").strip(),
    "v2": (_PROMPTS_DIR / "answer_v2.txt").read_text(encoding="utf-8").strip(),
    "v3": (_PROMPTS_DIR / "answer_v3.txt").read_text(encoding="utf-8").strip(),
}

# Approximate USD → GBP conversion used for cost estimates.
_USD_TO_GBP = 0.79

# Manual pricing for Nebius models (USD per token).
# Source: Nebius Token Factory pricing page, November 2026.
_NEBIUS_PRICING: dict[str, tuple[float, float]] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20e-6, 0.60e-6),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (0.07e-6, 0.28e-6),
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (0.14e-6, 0.28e-6),
    "deepseek-ai/DeepSeek-V4-Pro-0813": (1.32e-6, 3.96e-6),
}

# Qwen3 models emit <think>…</think> blocks in thinking mode; strip them.
_THINK_RE = re.compile(r"<think>.*</think>", re.DOTALL | re.IGNORECASE)

# Lazy singleton — created on first LLM call.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.nebius_api_key or "dummy",
            base_url=settings.nebius_base_url,
            timeout=120.0,
        )
    return _client


def _cache_path(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def _prompt_hash(model: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def context_text(rc: RetrievedChunk) -> str:
    """Text shown to the LLM: the expanded passage if present, else the chunk itself."""
    return rc.expanded_text or rc.chunk.text


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format numbered source passages followed by the question."""
    parts: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        heading = f" — {rc.chunk.section}" if rc.chunk.section else ""
        parts.append(f"[{i}] {rc.chunk.title}{heading}\n{context_text(rc)}")
    sources = "\n\n".join(parts)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    config: RagConfig,
    feedback: str | None = None,
) -> tuple[str, float, int]:
    """Call the LLM and return (answer_text, cost_gbp, generation_latency_ms).

    Uses a file-based cache keyed on model + full prompt to avoid re-spending
    tokens on repeated questions during development.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPTS[config.prompt_version]},
        {"role": "user", "content": _build_user_message(question, chunks)},
    ]
    if feedback:
        messages[1]["content"] += f"\n\n{feedback}"

    # Strip litellm's "openai/" router prefix — not used with the direct SDK.
    model_name = config.llm_model.removeprefix("openai/")
    cache_key = _prompt_hash(model_name, messages)

    if settings.llm_cache_enabled:
        cached = _cache_path(settings.llm_cache_dir, cache_key)
        if cached.exists():
            log.debug("LLM cache hit: %s", cache_key[:12])
            data = json.loads(cached.read_text(encoding="utf-8"))
            record_llm("llm.generation", model_name, None, None, data["cost_gbp"], 0)
            return data["text"], data["cost_gbp"], 0

    # Disable Qwen3 extended-thinking mode so the output is clean answer text.
    extra_body: dict[str, object] = {}
    if "Qwen3" in model_name:
        extra_body["enable_thinking"] = False

    t0 = time.monotonic()
    _backoff = 10.0
    response = None
    for _attempt in range(3):
        try:
            response = _get_client().chat.completions.create(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                extra_body=extra_body or None,
            )
            break
        except RateLimitError as exc:
            if _attempt == 2:
                raise
            log.warning(
                "LLM rate-limit (attempt %d/3) — backing off %.0fs: %s",
                _attempt + 1,
                _backoff,
                exc,
            )
            time.sleep(_backoff)
            _backoff = min(_backoff * 2, 20.0)
        except APIStatusError as exc:
            if exc.status_code != 503 or _attempt == 2:
                raise
            log.warning(
                "LLM 503 (attempt %d/3) — backing off %.0fs: %s",
                _attempt + 1,
                _backoff,
                exc,
            )
            time.sleep(_backoff)
            _backoff = min(_backoff * 2, 20.0)
        except APITimeoutError as exc:
            if _attempt == 2:
                raise
            log.warning(
                "LLM timeout (attempt %d/3) — backing off %.0fs: %s",
                _attempt + 1,
                _backoff,
                exc,
            )
            time.sleep(_backoff)
            _backoff = min(_backoff * 2, 20.0)
    latency_ms = int((time.monotonic() - t0) * 1000)

    assert response is not None  # loop always breaks or raises
    raw_text: str = response.choices[0].message.content or ""
    # Strip any residual <think>…</think> blocks.
    text = _THINK_RE.sub("", raw_text).strip()

    # Compute cost from manual pricing table.
    cost_usd = 0.0
    if response.usage and model_name in _NEBIUS_PRICING:
        _in, _out = _NEBIUS_PRICING[model_name]
        cost_usd = (response.usage.prompt_tokens or 0) * _in + (
            response.usage.completion_tokens or 0
        ) * _out
    cost_gbp = round(cost_usd * _USD_TO_GBP, 6)

    usage = response.usage
    record_llm(
        "llm.generation",
        model_name,
        usage.prompt_tokens if usage else None,
        usage.completion_tokens if usage else None,
        cost_gbp,
        latency_ms,
    )
    log.info("LLM %s: %d ms, $%.6f (£%.6f)", model_name, latency_ms, cost_usd, cost_gbp)

    if settings.llm_cache_enabled:
        cache_dir = Path(settings.llm_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        _cache_path(settings.llm_cache_dir, cache_key).write_text(
            json.dumps({"text": text, "cost_gbp": cost_gbp}),
            encoding="utf-8",
        )

    return text, cost_gbp, latency_ms
