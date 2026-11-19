"""Input and output guardrails for the RAG pipeline."""

from __future__ import annotations

import re

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
