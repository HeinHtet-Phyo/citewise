"""Incremental refresh: re-fetch and re-index only changed or new pages."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import httpx

from citewise.config import settings
from citewise.ingest.fetch import _fetch_one, _raw_path

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/manifest.csv")
RAW_DIR = Path("data/raw")


def _stored_timestamp(path: str) -> str | None:
    """Return the public_updated_at from the cached raw JSON, or None if missing."""
    raw = _raw_path(path)
    if not raw.exists():
        return None
    try:
        data = json.loads(raw.read_text(encoding="utf-8"))
        ts: str = data.get("public_updated_at", "")
        return ts
    except (json.JSONDecodeError, OSError):
        return None


def refresh() -> tuple[int, int]:
    """
    Compare manifest timestamps against stored raw JSON.

    Re-fetches pages whose public_timestamp differs from what is cached, and
    fetches pages that have no cached file at all. Reports counts.

    Returns (refreshed_count, unchanged_count).
    """
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Manifest not found. Run discover first.")

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    to_refresh: list[str] = []
    unchanged = 0

    for row in rows:
        path = row["path"]
        manifest_ts = row.get("public_timestamp", "")
        stored_ts = _stored_timestamp(path)
        if stored_ts is None or stored_ts != manifest_ts:
            to_refresh.append(path)
        else:
            unchanged += 1

    log.info("%d pages to refresh, %d unchanged", len(to_refresh), unchanged)

    headers = {"User-Agent": settings.govuk_user_agent}
    refreshed = 0
    failures = 0

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for path in to_refresh:
            try:
                data = _fetch_one(client, path)
                dest = _raw_path(path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                refreshed += 1
            except Exception as exc:
                log.warning("Failed to refresh %s: %s", path, exc)
                failures += 1

    log.info("Refresh done: %d updated, %d unchanged, %d failed", refreshed, unchanged, failures)
    return refreshed, unchanged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    r, u = refresh()
    print(f"Refreshed {r} pages, {u} unchanged.")
