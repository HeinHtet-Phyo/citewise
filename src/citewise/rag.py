"""Main RAG pipeline entry point.

answer_question(question, config) -> Answer is the single public interface.
"""

from __future__ import annotations

import logging
import time
import uuid

from citewise.config import RagConfig
from citewise.generate import generate_answer
from citewise.guardrails import check_evidence, check_input
from citewise.models import Answer, Citation
from citewise.retrieve import RetrievedChunk, retrieve

log = logging.getLogger(__name__)

# LLM refusal phrase — if the model returns this, mark answer as refused.
_REFUSAL_PREFIX = "I cannot find sufficient information"


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Convert RetrievedChunks into numbered Citations with short snippets."""
    citations: list[Citation] = []
    for i, rc in enumerate(chunks, start=1):
        snippet = rc.chunk.text[:297] + "…" if len(rc.chunk.text) > 300 else rc.chunk.text
        citations.append(
            Citation(
                n=i,
                title=rc.chunk.title,
                section=rc.chunk.section,
                url=rc.chunk.url,
                snippet=snippet,
                updated_at=rc.chunk.updated_at,
            )
        )
    return citations


def _refused_answer(
    question: str,
    reason: str,
    config: RagConfig,
    latency_ms: int,
    trace_id: str,
    retrieved_ids: list[str] | None = None,
) -> Answer:
    return Answer(
        question=question,
        answer="",
        citations=[],
        refused=True,
        refusal_reason=reason,
        config_name=config.config_name,
        latency_ms=latency_ms,
        cost_gbp=0.0,
        retrieved_chunk_ids=retrieved_ids or [],
        trace_id=trace_id,
    )


def answer_question(question: str, config: RagConfig) -> Answer:
    """Run the full RAG pipeline and return a structured Answer.

    Steps:
    1. Input guardrail — refuse unsafe or trivially short queries immediately.
    2. Retrieve candidate chunks from Qdrant.
    3. Evidence guardrail — refuse if top score is below threshold.
    4. Generate an answer with the LLM.
    5. Detect LLM-level refusals in the response text.
    6. Return a fully populated Answer.
    """
    t0 = time.monotonic()
    trace_id = str(uuid.uuid4())

    # 1. Input guardrail
    safe, refusal_reason = check_input(question)
    if not safe:
        log.info("Input refused (%s): %.60s", refusal_reason, question)
        return _refused_answer(
            question, refusal_reason or "unsafe", config, 0, trace_id
        )

    # 2. Retrieve
    chunks = retrieve(question, config)
    log.info("Retrieved %d chunks for: %.60s", len(chunks), question)

    # 3. Evidence guardrail
    top_score = chunks[0].score if chunks else 0.0
    refuse, evidence_reason = check_evidence(top_score, config.refusal_threshold)
    if refuse or not chunks:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return _refused_answer(
            question,
            evidence_reason or "low_evidence",
            config,
            latency_ms,
            trace_id,
        )

    # 4. Generate
    answer_text, cost_gbp, _ = generate_answer(question, chunks, config)

    # 5. Detect LLM-level refusal
    refused = answer_text.strip().startswith(_REFUSAL_PREFIX)
    refusal_reason_out: str | None = "low_evidence" if refused else None

    latency_ms = int((time.monotonic() - t0) * 1000)
    citations = _build_citations(chunks) if not refused else []

    return Answer(
        question=question,
        answer=answer_text,
        citations=citations,
        refused=refused,
        refusal_reason=refusal_reason_out,
        config_name=config.config_name,
        latency_ms=latency_ms,
        cost_gbp=cost_gbp,
        retrieved_chunk_ids=[rc.chunk.chunk_id for rc in chunks],
        trace_id=trace_id,
    )
