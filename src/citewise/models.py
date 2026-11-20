"""Shared data models for the Citewise pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One indexed piece of GOV.UK text."""

    chunk_id: str  # stable: f"{base_path}#{section_slug}-{n}"
    url: str  # https://www.gov.uk/<path>[#anchor or /part-slug]
    title: str
    section: str | None
    text: str
    format: str  # answer | guide | detailed_guide
    updated_at: str  # ISO date from public_updated_at
    tokens: int


class Citation(BaseModel):
    """A source reference in an answer."""

    n: int
    title: str
    section: str | None
    url: str
    snippet: str = Field(..., max_length=300)
    updated_at: str


class Answer(BaseModel):
    """Full pipeline output for one question."""

    question: str
    answer: str  # plain English with [n] markers
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None  # "low_evidence" | "out_of_scope" | "unsafe" | None
    config_name: str
    latency_ms: int
    cost_gbp: float
    retrieved_chunk_ids: list[str]
    trace_id: str | None
    verifier_path: str | None = None  # none | passed | repaired | dropped | refused
    context_texts: list[str] = Field(default_factory=list, exclude=True)
