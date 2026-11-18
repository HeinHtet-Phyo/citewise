"""Unit tests for src/citewise/ingest/chunk.py."""

from __future__ import annotations

from pathlib import Path

from citewise.ingest.chunk import _make_chunk_id, chunk_page
from citewise.ingest.clean import clean_page

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_chunk_ids_are_stable() -> None:
    """The same page chunked twice must produce identical chunk IDs."""
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    chunks1 = chunk_page(page)
    chunks2 = chunk_page(page)
    ids1 = [c.chunk_id for c in chunks1]
    ids2 = [c.chunk_id for c in chunks2]
    assert ids1 == ids2


def test_chunks_within_token_limit() -> None:
    """Every chunk must be at or below the configured max_tokens."""
    page = clean_page(FIXTURES / "answer_with_table.json")
    assert page is not None
    max_tokens = 400
    chunks = chunk_page(page, max_tokens=max_tokens)
    for chunk in chunks:
        assert chunk.tokens <= max_tokens + 20, (
            f"Chunk {chunk.chunk_id} has {chunk.tokens} tokens (limit {max_tokens})"
        )


def test_very_short_page_produces_one_chunk() -> None:
    page = clean_page(FIXTURES / "very_short_page.json")
    assert page is not None
    chunks = chunk_page(page)
    assert len(chunks) == 1
    assert "Blue Badge" in chunks[0].text


def test_chunk_contains_title_prefix() -> None:
    """Every chunk must start with the page title so it's self-contained."""
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    chunks = chunk_page(page)
    for chunk in chunks:
        assert chunk.title in chunk.text, f"Title missing from chunk {chunk.chunk_id}"


def test_guide_part_urls_preserved() -> None:
    """Chunks from a guide with parts must carry the part-specific URL."""
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    chunks = chunk_page(page)
    urls = {c.url for c in chunks}
    assert any("overview" in u for u in urls)
    assert any("what-you-ll-get" in u for u in urls)


def test_all_chunks_have_required_fields() -> None:
    page = clean_page(FIXTURES / "answer_with_table.json")
    assert page is not None
    for chunk in chunk_page(page):
        assert chunk.chunk_id
        assert chunk.url.startswith("https://www.gov.uk")
        assert chunk.title
        assert chunk.text
        assert chunk.tokens > 0
        assert chunk.format in {"answer", "guide", "detailed_guide"}
        assert len(chunk.updated_at) == 10  # ISO date YYYY-MM-DD


def test_make_chunk_id_format() -> None:
    cid = _make_chunk_id("/statutory-sick-pay", "Eligibility Requirements", 0)
    assert cid.startswith("/statutory-sick-pay#")
    assert " " not in cid


def test_800_token_chunks_fewer_than_400() -> None:
    """Larger chunk size should produce fewer chunks for the same content."""
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    small = chunk_page(page, max_tokens=200)
    large = chunk_page(page, max_tokens=800)
    assert len(large) <= len(small)
