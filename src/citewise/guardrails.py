"""Input and output guardrails for the RAG pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import litellm

from citewise.config import settings
from citewise.generate import _cache_path, _prompt_hash

# Patterns that indicate a question is unsafe or a prompt injection attempt.
_UNSAFE_RE = re.compile(
    r"""
    \b(how\s+to\s+(make|build|create|synthesise|synthesize)\s+(a\s+)?(bomb|weapon|explosive|poison|drug))
    | \b(self.?harm|suicide\s+method|kill\s+(yourself|myself|themselves))
    | ignore\s+(all\s+)?(previous|prior|above)\s+instructions?
    | disregard\s+(all\s+)?(previous|prior|above)\s+instructions?
    | you\s+are\s+now\s+(a|an)\s+\w
    | act\s+as\s+if\s+you
    | pretend\s+(you\s+are|to\s+be)
    | jailbreak
    | (do\s+anything\s+now|dan\s+mode)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def check_input(question: str) -> tuple[bool, str | None]:
    """Return (is_acceptable, refusal_reason) for a user question.

    Returns (True, None) when the question is safe to process.
    Returns (False, reason) when it should be refused without retrieval.
    """
    if _UNSAFE_RE.search(question):
        return False, "unsafe"
    if len(question.strip()) < 5:
        return False, "out_of_scope"
    return True, None


def check_evidence(top_score: float, threshold: float) -> tuple[bool, str | None]:
    """Return (should_refuse, reason) based on retrieval confidence.

    If the best-matching chunk scores below threshold, the question is likely
    outside the corpus and we refuse rather than hallucinate.
    """
    if threshold > 0.0 and top_score < threshold:
        return True, "low_evidence"
    return False, None


_VERIFY_PROMPT = (
    (Path(__file__).parent / "prompts" / "verify_claims.txt").read_text(encoding="utf-8").strip()
)
_VERIFY_MODEL = "openai/Qwen/Qwen3-30B-A3B-Instruct-2507"
_VERIFY_PRICE_USD = (0.07e-6, 0.28e-6)
_USD_TO_GBP = 0.79
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences (citation markers stay attached to their sentence)."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _parse_unsupported(text: str, n_sentences: int) -> list[int]:
    """Read the 1-based sentence numbers from the verifier's JSON; empty list if unreadable."""
    start = text.find("{")
    if start < 0:
        return []
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        numbers = [int(n) for n in obj.get("unsupported", [])]
    except (ValueError, TypeError, AttributeError):
        return []
    return [n for n in numbers if 1 <= n <= n_sentences]


def verify_claims(answer: str, sources: list[str]) -> tuple[list[str], float]:
    """Return (unsupported sentences, cost_gbp) by asking a small LLM to check each sentence.

    Fails open: if the verifier call or its JSON fails, no sentence is flagged.
    """
    sentences = split_sentences(answer)
    numbered_sources = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(sources))
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    messages = [
        {"role": "system", "content": _VERIFY_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{numbered_sources}\n\nAnswer:\n{numbered}"},
    ]
    key = _prompt_hash(_VERIFY_MODEL, messages)
    path = _cache_path(settings.llm_cache_dir, key)
    if settings.llm_cache_enabled and path.exists():
        text = json.loads(path.read_text(encoding="utf-8"))["text"]
        cost = 0.0
    else:
        try:
            response = litellm.completion(
                model=_VERIFY_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                base_url=settings.nebius_base_url or None,
                api_key=settings.nebius_api_key or None,
                timeout=60,
            )
        except Exception:
            return [], 0.0
        text = response.choices[0].message.content or ""
        usage = response.usage
        cost = round(
            (
                (usage.prompt_tokens or 0) * _VERIFY_PRICE_USD[0]
                + (usage.completion_tokens or 0) * _VERIFY_PRICE_USD[1]
            )
            * _USD_TO_GBP,
            6,
        )
        if settings.llm_cache_enabled:
            Path(settings.llm_cache_dir).mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text}), encoding="utf-8")
    flagged = _parse_unsupported(text, len(sentences))
    return [sentences[n - 1] for n in flagged], cost


def drop_sentences(answer: str, unsupported: list[str]) -> str:
    """Remove the unsupported sentences from an answer."""
    bad = set(unsupported)
    return " ".join(s for s in split_sentences(answer) if s not in bad)
