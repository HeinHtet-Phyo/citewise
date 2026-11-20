"""Unit tests for retrieve.py — mock Qdrant and embedding models."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import SparseVector

from citewise.config import RagConfig
from citewise.models import Chunk
from citewise.retrieve import RetrievedChunk, _dedup_by_url, _points_to_chunks, retrieve


def _make_chunk(url: str, chunk_id: str = "test#intro-0") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        url=url,
        title="Test Page",
        section=None,
        text="Test text.",
        format="answer",
        updated_at="2026-01-01",
        tokens=10,
    )


def _make_scored_point(url: str, score: float = 0.9, chunk_id: str = "test#intro-0") -> MagicMock:
    pt = MagicMock()
    pt.score = score
    pt.payload = {
        "chunk_id": chunk_id,
        "url": url,
        "title": "Test Page",
        "section": None,
        "text": "Test text.",
        "format": "answer",
        "updated_at": "2026-01-01",
        "tokens": 10,
    }
    return pt


def test_dedup_by_url_removes_duplicates() -> None:
    rc1 = RetrievedChunk(chunk=_make_chunk("https://www.gov.uk/page"), score=0.9)
    rc2 = RetrievedChunk(chunk=_make_chunk("https://www.gov.uk/page"), score=0.7)
    rc3 = RetrievedChunk(chunk=_make_chunk("https://www.gov.uk/other"), score=0.8)
    deduped = _dedup_by_url([rc1, rc2, rc3], top_k=10)
    urls = [rc.chunk.url for rc in deduped]
    assert urls.count("https://www.gov.uk/page") == 1
    assert len(deduped) == 2


def test_dedup_keeps_top_k() -> None:
    rcs = [
        RetrievedChunk(chunk=_make_chunk(f"https://gov.uk/page-{i}", f"test#intro-{i}"), score=0.9)
        for i in range(10)
    ]
    deduped = _dedup_by_url(rcs, top_k=5)
    assert len(deduped) == 5


def test_points_to_chunks_converts_payload() -> None:
    pt = _make_scored_point("https://www.gov.uk/test")
    chunks = _points_to_chunks([pt])
    assert len(chunks) == 1
    assert chunks[0].chunk.url == "https://www.gov.uk/test"
    assert chunks[0].score == pt.score


def test_retrieve_dense_no_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """retrieve() with dense-only config returns top_k_context chunks."""
    # Mock Qdrant client
    mock_points = [
        _make_scored_point(f"https://gov.uk/page-{i}", 0.9 - i * 0.05, f"p{i}#intro-0")
        for i in range(5)
    ]
    mock_result = MagicMock()
    mock_result.points = mock_points

    mock_client = MagicMock()
    mock_client.query_points.return_value = mock_result
    monkeypatch.setattr("citewise.retrieve._get_client", lambda: mock_client)

    # Mock embeddings
    monkeypatch.setattr("citewise.retrieve.embed_dense_single", lambda _: [0.1] * 384)

    config = RagConfig(
        config_name="test-dense",
        search_mode="dense",
        use_reranker=False,
        top_k_retrieve=10,
        top_k_context=3,
    )
    results = retrieve("What is the income tax rate?", config)
    assert len(results) <= 3
    for rc in results:
        assert isinstance(rc, RetrievedChunk)


def test_retrieve_hybrid_calls_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """retrieve() with hybrid config passes prefetch + RRF to Qdrant."""
    mock_points: list[MagicMock] = []
    mock_result = MagicMock()
    mock_result.points = mock_points

    mock_client = MagicMock()
    mock_client.query_points.return_value = mock_result
    monkeypatch.setattr("citewise.retrieve._get_client", lambda: mock_client)
    monkeypatch.setattr("citewise.retrieve.embed_dense_single", lambda _: [0.1] * 384)
    monkeypatch.setattr(
        "citewise.retrieve.embed_sparse_single",
        lambda _: SparseVector(indices=[1, 2], values=[0.5, 0.3]),
    )

    config = RagConfig(
        config_name="test-hybrid",
        search_mode="hybrid",
        use_reranker=False,
        top_k_retrieve=5,
        top_k_context=3,
    )
    retrieve("test query", config)

    call_kwargs = mock_client.query_points.call_args
    assert "prefetch" in call_kwargs.kwargs
    assert len(call_kwargs.kwargs["prefetch"]) == 2
