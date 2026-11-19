"""Build Qdrant collections from chunked data.

Creates (or recreates) a named collection with:
- dense vectors from BAAI/bge-small-en-v1.5 (384-dim, Cosine)
- sparse vectors from BM25 (Qdrant/bm25)

Payloads stored per point: chunk_id, url, title, section, format,
updated_at, tokens, text.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from citewise.config import settings
from citewise.index.embed import DENSE_DIM, embed_dense, embed_sparse
from citewise.ingest.chunk import load_chunks
from citewise.models import Chunk

log = logging.getLogger(__name__)

COLLECTION_DENSE_NAME = "dense"
COLLECTION_SPARSE_NAME = "sparse"
UPSERT_BATCH = 128


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


def _batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive batches of `size` from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _ensure_collection(client: QdrantClient, collection_name: str) -> None:
    """Create the collection if it does not exist."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        log.info("Collection %s already exists — skipping creation", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            COLLECTION_DENSE_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            COLLECTION_SPARSE_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    log.info("Created collection %s", collection_name)


def build_index(
    collection_name: str,
    chunk_size: int = 400,
    recreate: bool = False,
) -> int:
    """
    Embed all chunks and upsert them into a Qdrant collection.

    Returns the total number of points upserted.
    """
    client = _client()

    if recreate:
        try:
            client.delete_collection(collection_name)
            log.info("Deleted existing collection %s", collection_name)
        except Exception:
            pass

    _ensure_collection(client, collection_name)

    chunks = load_chunks(chunk_size)
    log.info("Loaded %d chunks from chunk_size=%d", len(chunks), chunk_size)

    total = 0
    for batch in _batched(chunks, UPSERT_BATCH):
        texts = [c.text for c in batch]
        dense_vecs = embed_dense(texts)
        sparse_vecs = embed_sparse(texts)

        points = [
            PointStruct(
                id=_stable_id(chunk),
                vector={
                    COLLECTION_DENSE_NAME: dense_vecs[i],
                    COLLECTION_SPARSE_NAME: sparse_vecs[i],
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "url": chunk.url,
                    "title": chunk.title,
                    "section": chunk.section,
                    "format": chunk.format,
                    "updated_at": chunk.updated_at,
                    "tokens": chunk.tokens,
                    "text": chunk.text,
                },
            )
            for i, chunk in enumerate(batch)
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(batch)
        log.info("Upserted %d / %d", total, len(chunks))

    log.info("Index built: %d points in %s", total, collection_name)
    return total


def _stable_id(chunk: Chunk) -> int:
    """Derive a stable 32-bit integer point ID from chunk_id."""
    return hash(chunk.chunk_id) & 0xFFFFFFFF


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collection = sys.argv[1] if len(sys.argv) > 1 else "citewise_400"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    n = build_index(collection_name=collection, chunk_size=size)
    print(f"Done: {n} points in {collection}")
