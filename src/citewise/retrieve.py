"""Retrieve relevant chunks from Qdrant using dense, sparse, or hybrid search.

Pipeline:
1. Embed the query (dense + sparse).
2. Issue a prefetch query (hybrid) or single query (dense) to Qdrant.
3. Optionally rerank with a cross-encoder.
4. Deduplicate: if two chunks share the same page URL, keep only the top-scoring one.
5. Return the top-k chunks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    NearestQuery,
    Prefetch,
    ScoredPoint,
    SparseVector,
)

from citewise.config import RagConfig, settings
from citewise.index.embed import (
    embed_dense_single,
    embed_sparse_single,
)
from citewise.models import Chunk

log = logging.getLogger(__name__)

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_reranker: TextCrossEncoder | None = None
_qdrant: QdrantClient | None = None


def _get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        log.info("Loading reranker %s", RERANKER_MODEL)
        _reranker = TextCrossEncoder(RERANKER_MODEL)
    return _reranker


def _get_client() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _qdrant


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its relevance score."""

    chunk: Chunk
    score: float


def _points_to_chunks(points: list[ScoredPoint]) -> list[RetrievedChunk]:
    """Convert Qdrant ScoredPoints to RetrievedChunk objects."""
    result: list[RetrievedChunk] = []
    for pt in points:
        payload = pt.payload or {}
        chunk = Chunk(
            chunk_id=payload["chunk_id"],
            url=payload["url"],
            title=payload["title"],
            section=payload.get("section"),
            text=payload["text"],
            format=payload["format"],
            updated_at=payload["updated_at"],
            tokens=payload["tokens"],
        )
        result.append(RetrievedChunk(chunk=chunk, score=pt.score))
    return result


def _dedup_by_url(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """
    Keep only the highest-scoring chunk per unique page URL.

    Guide parts have distinct URLs (e.g. /page/overview vs /page/what-you-get)
    and are kept separately; dedup only collapses duplicate hits on the same URL.
    """
    seen: dict[str, float] = {}
    deduped: list[RetrievedChunk] = []
    for rc in chunks:
        if rc.chunk.url not in seen:
            seen[rc.chunk.url] = rc.score
            deduped.append(rc)
        if len(deduped) >= top_k:
            break
    return deduped


def retrieve(query: str, config: RagConfig) -> list[RetrievedChunk]:
    """
    Retrieve and rerank relevant chunks for a query.

    Returns up to config.top_k_context chunks.
    """
    client = _get_client()

    if config.search_mode == "hybrid":
        dense_vec = embed_dense_single(query)
        sparse_vec = embed_sparse_single(query)
        points = client.query_points(
            collection_name=config.collection_name,
            prefetch=[
                Prefetch(
                    query=NearestQuery(nearest=dense_vec),
                    using="dense",
                    limit=config.top_k_retrieve,
                ),
                Prefetch(
                    query=NearestQuery(
                        nearest=SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        )
                    ),
                    using="sparse",
                    limit=config.top_k_retrieve,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=config.top_k_retrieve,
            with_payload=True,
        ).points
    else:
        dense_vec = embed_dense_single(query)
        points = client.query_points(
            collection_name=config.collection_name,
            query=NearestQuery(nearest=dense_vec),
            using="dense",
            limit=config.top_k_retrieve,
            with_payload=True,
        ).points

    candidates = _points_to_chunks(points)
    log.debug("Retrieved %d candidates from Qdrant", len(candidates))

    if config.use_reranker and candidates:
        reranker = _get_reranker()
        texts = [rc.chunk.text for rc in candidates]
        scores = list(reranker.rerank(query, texts))
        for rc, score in zip(candidates, scores, strict=False):
            rc.score = float(score)
        candidates.sort(key=lambda rc: rc.score, reverse=True)
        log.debug("Reranked %d candidates", len(candidates))

    return _dedup_by_url(candidates, config.top_k_context)
