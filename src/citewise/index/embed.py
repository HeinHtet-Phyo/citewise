"""Embed chunks using dense (bge-small-en-v1.5) and sparse (BM25) models."""

from __future__ import annotations

import logging

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client.models import SparseVector

log = logging.getLogger(__name__)

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 384
BATCH_SIZE = 64

# Module-level singletons — initialised lazily
_dense: TextEmbedding | None = None
_sparse: SparseTextEmbedding | None = None


_CPU_PROVIDER = [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def _get_dense() -> TextEmbedding:
    global _dense
    if _dense is None:
        log.info("Loading dense model %s", DENSE_MODEL)
        _dense = TextEmbedding(DENSE_MODEL, threads=1, providers=_CPU_PROVIDER)
    return _dense


def _get_sparse() -> SparseTextEmbedding:
    global _sparse
    if _sparse is None:
        log.info("Loading sparse model %s", SPARSE_MODEL)
        _sparse = SparseTextEmbedding(SPARSE_MODEL, threads=1, providers=_CPU_PROVIDER)
    return _sparse


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Return dense vectors for a batch of texts."""
    return [v.tolist() for v in _get_dense().embed(texts, batch_size=BATCH_SIZE)]


def embed_sparse(texts: list[str]) -> list[SparseVector]:
    """Return BM25 sparse vectors for a batch of texts."""
    results: list[SparseVector] = []
    for emb in _get_sparse().embed(texts, batch_size=BATCH_SIZE):
        sv = _as_sparse_vector(emb)
        results.append(sv)
    return results


def embed_dense_single(text: str) -> list[float]:
    """Embed a single query string with the dense model."""
    return embed_dense([text])[0]


def embed_sparse_single(text: str) -> SparseVector:
    """Embed a single query string with the sparse model."""
    return embed_sparse([text])[0]


def _as_sparse_vector(emb: SparseEmbedding) -> SparseVector:
    """Convert a fastembed sparse result to Qdrant SparseVector."""
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
