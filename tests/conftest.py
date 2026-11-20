"""Shared test fixtures."""

from __future__ import annotations

import pytest

from citewise.config import settings


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tracing off in every test so a developer's .env keys never cause network calls."""
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
