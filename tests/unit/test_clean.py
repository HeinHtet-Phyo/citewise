"""Unit tests for src/citewise/ingest/clean.py."""

from __future__ import annotations

from pathlib import Path

from citewise.ingest.clean import _html_to_text, clean_page

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_clean_guide_with_parts() -> None:
    page = clean_page(FIXTURES / "guide_with_parts.json")
    assert page is not None
    assert page.title == "Statutory Sick Pay (SSP)"
    assert page.format == "guide"
    assert page.updated_at == "2024-01-15"
    # Two parts → at least two sections.
    assert len(page.sections) >= 2
    # Each section URL should include the part slug.
    urls = [s.url for s in page.sections]
    assert any("overview" in u for u in urls)
    assert any("what-you-ll-get" in u for u in urls)


def test_clean_answer_with_table() -> None:
    page = clean_page(FIXTURES / "answer_with_table.json")
    assert page is not None
    assert page.title == "Income Tax rates and Personal Allowances"
    # Table content should appear as text, not raw HTML.
    full_text = " ".join(s.text for s in page.sections)
    assert "<table>" not in full_text
    assert "Basic rate" in full_text
    assert "20%" in full_text


def test_clean_very_short_page() -> None:
    page = clean_page(FIXTURES / "very_short_page.json")
    assert page is not None
    assert len(page.sections) >= 1
    assert "Blue Badge" in page.sections[0].text


def test_non_english_page_skipped(tmp_path: Path) -> None:
    welsh = tmp_path / "welsh.json"
    welsh.write_text(
        '{"base_path":"/cy/test","title":"Test","document_type":"answer",'
        '"locale":"cy","public_updated_at":"2024-01-01T00:00:00Z",'
        '"details":{"body":"<p>Cymraeg</p>"}}',
        encoding="utf-8",
    )
    assert clean_page(welsh) is None


def test_page_missing_body_skipped(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(
        '{"base_path":"/test","title":"Test","document_type":"answer",'
        '"locale":"en","public_updated_at":"2024-01-01T00:00:00Z","details":{}}',
        encoding="utf-8",
    )
    assert clean_page(empty) is None


def test_updated_at_preferred_over_public_updated_at(tmp_path: Path) -> None:
    """updated_at (actual CMS edit time) beats public_updated_at when both are present."""
    p = tmp_path / "rate_page.json"
    p.write_text(
        '{"base_path":"/statutory-sick-pay","title":"SSP","document_type":"answer",'
        '"locale":"en",'
        '"public_updated_at":"2015-03-09T14:15:47Z",'
        '"updated_at":"2026-10-27T09:45:51Z",'
        '"details":{"body":"<p>SSP is £123.25 a week.</p>"}}',
        encoding="utf-8",
    )
    page = clean_page(p)
    assert page is not None
    assert page.updated_at == "2026-10-27"


def test_updated_at_falls_back_to_public_updated_at(tmp_path: Path) -> None:
    """Pages without updated_at still get a date from public_updated_at."""
    p = tmp_path / "old_page.json"
    p.write_text(
        '{"base_path":"/test","title":"Test","document_type":"answer",'
        '"locale":"en",'
        '"public_updated_at":"2024-06-01T00:00:00Z",'
        '"details":{"body":"<p>Some content.</p>"}}',
        encoding="utf-8",
    )
    page = clean_page(p)
    assert page is not None
    assert page.updated_at == "2024-06-01"


def test_html_to_text_headings() -> None:
    html = "<h2>Eligibility</h2><p>You must be over 16.</p>"
    text = _html_to_text(html)
    assert "## Eligibility" in text
    assert "You must be over 16." in text


def test_html_to_text_no_raw_tags() -> None:
    html = "<p>Hello <strong>world</strong></p><ul><li>Item one</li></ul>"
    text = _html_to_text(html)
    assert "<" not in text
    assert "Hello" in text
    assert "Item one" in text
