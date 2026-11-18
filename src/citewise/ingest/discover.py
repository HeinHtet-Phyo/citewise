"""Discover GOV.UK pages via the Search API and save a manifest CSV."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from citewise.config import settings

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.gov.uk/api/search.json"
MANIFEST_PATH = Path("data/manifest.csv")
MANIFEST_FIELDS = ["path", "title", "format", "public_timestamp", "description"]

# Formats and topic filters per the project spec.
# answer + guide: all topics (no filter needed).
# detailed_guide: only the five citizen topics.
FORMATS_ALL_TOPICS = ["answer", "guide"]
FORMATS_WITH_TOPICS = ["detailed_guide"]
# Taxonomy content IDs for citizen-facing detailed_guide topics.
# Retrieved via the Content API (https://www.gov.uk/api/content/<path>).
# Plain-text slugs (e.g. "benefits") do not work with filter_part_of_taxonomy_tree.
DETAILED_GUIDE_TOPICS = [
    "dded88e2-f92e-424f-b73e-6ad24a839c51",  # welfare / benefits
    "d0f1e5a3-c8f4-4780-8678-994f19104b21",  # work / employment rights
    "4794066e-e3cc-425e-8cc4-e7ff3edb4c39",  # housing and local community
    "a4038b29-b332-4f13-98b1-1c9709e216bc",  # transport / driving
]

# GOV.UK Search API page size limit.
PAGE_SIZE = 100

# Formats we explicitly skip (transactions, smart answers, etc.)
SKIP_FORMATS = {
    "transaction",
    "smart_answer",
    "calendar",
    "local_transaction",
    "place",
    "licence",
    "completed_transaction",
    "simple_smart_answer",
}


# The GOV.UK Search API returns a fixed set of fields by default; locale is included.
# Custom field selection causes 422 errors, so we request no specific fields.


def _fetch_page(
    client: httpx.Client,
    params: list[tuple[str, str | int | float | bool | None]],
) -> dict[str, Any]:
    """Fetch one page of search results, retrying on transient errors."""
    for attempt in range(5):
        try:
            resp = client.get(SEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()
            return result
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            wait = 2**attempt
            log.warning(
                "Search API error (attempt %d/5): %s — retrying in %ds", attempt + 1, exc, wait
            )
            time.sleep(wait)
    raise RuntimeError(f"Search API failed after 5 attempts for params {params}")


def _discover_format(
    client: httpx.Client,
    fmt: str,
    topic: str | None = None,
) -> list[dict[str, str]]:
    """Page through Search API results for one format (and optional topic)."""
    rows: list[dict[str, str]] = []
    start = 0
    while True:
        raw_params: list[tuple[str, str]] = [
            ("filter_format", fmt),
            ("count", str(PAGE_SIZE)),
            ("start", str(start)),
        ]
        if topic:
            raw_params.append(("filter_part_of_taxonomy_tree", topic))
        params: list[tuple[str, str | int | float | bool | None]] = list(raw_params)

        data = _fetch_page(client, params)
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            # Skip non-English pages.
            if item.get("locale", "en") != "en":
                continue
            path = item.get("link", "")
            if not path:
                continue
            rows.append(
                {
                    "path": path,
                    "title": item.get("title", ""),
                    "format": item.get("format", fmt),
                    "public_timestamp": item.get("public_timestamp", ""),
                    "description": item.get("description", ""),
                }
            )
        total = data.get("total", 0)
        start += len(results)
        log.info("  %s/%s (format=%s%s)", start, total, fmt, f", topic={topic}" if topic else "")
        if start >= total:
            break
        # Polite rate limiting.
        time.sleep(1.0 / settings.govuk_rate_limit_rps)

    return rows


def discover() -> int:
    """
    Discover all in-scope GOV.UK pages and write data/manifest.csv.

    Returns the total page count.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": settings.govuk_user_agent}
    all_rows: dict[str, dict[str, str]] = {}  # path -> row (dedup by path)

    with httpx.Client(headers=headers) as client:
        for fmt in FORMATS_ALL_TOPICS:
            log.info("Discovering format=%s (all topics)…", fmt)
            for row in _discover_format(client, fmt):
                all_rows[row["path"]] = row

        for fmt in FORMATS_WITH_TOPICS:
            for topic in DETAILED_GUIDE_TOPICS:
                log.info("Discovering format=%s, topic=%s…", fmt, topic)
                for row in _discover_format(client, fmt, topic):
                    all_rows[row["path"]] = row

    rows = list(all_rows.values())
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary by format.
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["format"]] = counts.get(row["format"], 0) + 1
    log.info("Manifest written to %s — %d pages total", MANIFEST_PATH, len(rows))
    for fmt, n in sorted(counts.items()):
        log.info("  %-20s %d", fmt, n)

    return len(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    total = discover()
    print(f"Done. {total} pages in manifest.")
