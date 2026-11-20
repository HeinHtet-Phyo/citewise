"""Fixtures: in-memory Qdrant with three tiny chunks, a fake embedder and a fake LLM."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SparseVectorParams, VectorParams

from citewise.api import main as api_main
from citewise.api.budget import DailyBudget

DIM = 384
CHUNKS = [
    ("ssp#overview-0", "https://www.gov.uk/statutory-sick-pay", "Statutory Sick Pay",
     "Statutory Sick Pay is paid at 118.75 pounds a week for up to 28 weeks."),
    ("uc#amount-0", "https://www.gov.uk/universal-credit", "Universal Credit",
     "Universal Credit is a monthly payment to help with living costs."),
    ("mot#when-0", "https://www.gov.uk/getting-an-mot", "Getting an MOT",
     "Your car needs an MOT test once it is three years old."),
]  # fmt: skip


def fake_embed(text: str) -> list[float]:
    """Hash words into a 384-dim unit vector so similar wording gives similar vectors."""
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """TestClient over the real API and pipeline, with Qdrant, embeddings and the LLM faked."""
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        "citewise_400",
        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    qdrant.upsert(
        "citewise_400",
        [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, cid)),
                vector={"dense": fake_embed(text)},
                payload={
                    "chunk_id": cid, "url": url, "title": title, "section": None, "text": text,
                    "format": "answer", "updated_at": "2026-01-01", "tokens": 20, "position": i,
                },
            )
            for i, (cid, url, title, text) in enumerate(CHUNKS)
        ],
    )  # fmt: skip
    monkeypatch.setattr("citewise.retrieve._get_client", lambda: qdrant)
    monkeypatch.setattr("citewise.retrieve.embed_dense_single", fake_embed)
    monkeypatch.setattr(api_main, "_budget", DailyBudget(cap_gbp=10.0))
    api_main._rate_buckets.clear()
    yield TestClient(api_main.app)
    api_main._rate_buckets.clear()


def fake_llm(text: str, cost: float = 0.0004) -> object:
    """Return a stand-in for generate_answer that always answers with `text`."""
    return lambda question, chunks, config, feedback=None: (text, cost, 5)
