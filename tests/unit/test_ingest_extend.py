"""Tests for cleaning service-style pages that have no body."""

from __future__ import annotations

import json
from pathlib import Path

from citewise.ingest.clean import clean_page


def test_transaction_page_uses_fallback_text(tmp_path: Path) -> None:
    raw = {
        "base_path": "/check-thing",
        "title": "Check a thing",
        "document_type": "transaction",
        "public_updated_at": "2026-01-02T00:00:00Z",
        "description": "Find out about the thing.",
        "details": {"introductory_paragraph": "<p>Use this service to check it.</p>"},
    }
    path = tmp_path / "check-thing.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    page = clean_page(path)
    assert page is not None
    assert page.format == "answer"
    assert "Use this service to check it." in page.sections[0].text
