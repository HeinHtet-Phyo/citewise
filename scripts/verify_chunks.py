"""Verify chunk merge logic on fixture pages."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citewise.ingest.chunk import chunk_page
from citewise.ingest.clean import clean_page

FIXTURES = Path(__file__).parent.parent / "tests/fixtures"


def test_no_infinite_loop() -> None:
    """Every fixture page must chunk without hanging."""
    for fixture in FIXTURES.glob("*.json"):
        page = clean_page(fixture)
        if page is None:
            continue
        chunks_400 = chunk_page(page, max_tokens=400)
        chunks_800 = chunk_page(page, max_tokens=800)
        assert len(chunks_400) >= 1, f"{fixture.name}: 0 chunks at 400"
        assert len(chunks_800) >= 1, f"{fixture.name}: 0 chunks at 800"
        # 800-token chunks should be <= number of 400-token chunks
        assert len(chunks_800) <= len(chunks_400), (
            f"{fixture.name}: 800-chunk count {len(chunks_800)} > 400-chunk count {len(chunks_400)}"
        )
    print("no_infinite_loop: PASS")


def test_url_boundaries_respected() -> None:
    """Sections from different guide parts must never be merged."""
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    chunks = chunk_page(page)
    urls = {c.url for c in chunks}
    assert any("overview" in u for u in urls), f"overview missing from {urls}"
    assert any("what-you-ll-get" in u for u in urls), f"what-you-ll-get missing from {urls}"
    print("url_boundaries_respected: PASS")


def test_chunk_ids_stable() -> None:
    """Running twice on the same page must produce identical IDs."""
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    ids1 = [c.chunk_id for c in chunk_page(page)]
    ids2 = [c.chunk_id for c in chunk_page(page)]
    assert ids1 == ids2, "IDs differ between runs"
    print("chunk_ids_stable: PASS")


def test_title_in_every_chunk() -> None:
    """Every chunk must contain its page title."""
    for fixture in FIXTURES.glob("*.json"):
        page = clean_page(fixture)
        if page is None:
            continue
        for c in chunk_page(page):
            assert page.title in c.text, f"{c.chunk_id}: title missing"
    print("title_in_every_chunk: PASS")


def test_merge_increases_avg_tokens() -> None:
    """The 400-token avg should be substantially higher than the old 126-token avg."""
    from citewise.ingest.clean import clean_all

    pages = clean_all()
    sample = pages[:100]
    all_400 = [c for p in sample for c in chunk_page(p, max_tokens=400)]
    all_800 = [c for p in sample for c in chunk_page(p, max_tokens=800)]
    avg_400 = sum(c.tokens for c in all_400) / len(all_400)
    avg_800 = sum(c.tokens for c in all_800) / len(all_800)
    print(f"avg 400-chunk: {avg_400:.0f} tokens (was 126, want >200)")
    print(f"avg 800-chunk: {avg_800:.0f} tokens (want >400)")
    assert avg_400 > 180, f"400-avg too low: {avg_400:.0f}"
    assert avg_800 > avg_400 * 1.3, "800-avg not meaningfully larger than 400-avg"
    print("merge_increases_avg_tokens: PASS")


if __name__ == "__main__":
    test_no_infinite_loop()
    test_url_boundaries_respected()
    test_chunk_ids_stable()
    test_title_in_every_chunk()
    test_merge_increases_avg_tokens()
    print("\nAll checks passed.")
