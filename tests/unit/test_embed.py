"""Unit tests for embed.py — test vector shapes and types without hitting real models."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client.models import SparseVector

from citewise.index.embed import (
    DENSE_DIM,
    _as_sparse_vector,
    embed_dense,
    embed_dense_single,
    embed_sparse,
    embed_sparse_single,
)


def _fake_dense_model(texts: list[str], batch_size: int) -> list[np.ndarray]:
    return [np.ones(DENSE_DIM, dtype=np.float32) for _ in texts]


def _fake_sparse_model(texts: list[str], batch_size: int) -> list[SparseEmbedding]:
    return [
        SparseEmbedding(
            indices=np.array([0, 1, 2], dtype=np.int32),
            values=np.array([0.5, 0.3, 0.2], dtype=np.float32),
        )
        for _ in texts
    ]


@pytest.fixture(autouse=True)
def mock_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch model singletons so tests run without downloading ONNX files."""
    dense_mock = MagicMock()
    dense_mock.embed.side_effect = _fake_dense_model

    sparse_mock = MagicMock()
    sparse_mock.embed.side_effect = _fake_sparse_model

    monkeypatch.setattr("citewise.index.embed._dense", dense_mock)
    monkeypatch.setattr("citewise.index.embed._sparse", sparse_mock)


def test_embed_dense_shape() -> None:
    vecs = embed_dense(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == DENSE_DIM


def test_embed_dense_single_shape() -> None:
    vec = embed_dense_single("test query")
    assert len(vec) == DENSE_DIM


def test_embed_sparse_returns_sparse_vector() -> None:
    vecs = embed_sparse(["hello", "world"])
    assert len(vecs) == 2
    for sv in vecs:
        assert isinstance(sv, SparseVector)
        assert len(sv.indices) == len(sv.values)


def test_embed_sparse_single() -> None:
    sv = embed_sparse_single("test query")
    assert isinstance(sv, SparseVector)


def test_as_sparse_vector_conversion() -> None:
    emb = SparseEmbedding(
        indices=np.array([10, 20], dtype=np.int32),
        values=np.array([0.9, 0.1], dtype=np.float32),
    )
    sv = _as_sparse_vector(emb)
    assert sv.indices == [10, 20]
    assert pytest.approx(sv.values[0], abs=1e-5) == 0.9
