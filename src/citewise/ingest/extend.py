"""Add specific GOV.UK pages to an existing corpus without re-chunking the rest."""

from __future__ import annotations

import csv
import json
import logging
import time

import httpx

from citewise.config import settings
from citewise.ingest.chunk import PROCESSED_DIR, chunk_page, load_chunks
from citewise.ingest.clean import clean_page
from citewise.ingest.fetch import CONTENT_API, MANIFEST_PATH, _raw_path
from citewise.models import Chunk

log = logging.getLogger(__name__)

# Frequently visited citizen pages that the original discovery run missed.
EXTRA_PATHS = [
    "/state-pension-age",
    "/check-state-pension",
    "/government/publications/state-pension-age-timetable",
    "/council-tax-bands",
    "/guidance/council-tax-band-challenge-form",
    "/apply-council-tax-reduction",
    "/apply-for-council-tax-discount",
    "/check-national-insurance-record",
    "/register-to-vote",
    "/apply-blue-badge",
    "/check-mot-history",
    "/vehicle-tax",
    "/check-vehicle-tax",
    "/calculate-your-holiday-entitlement",
    "/estimate-income-tax",
    "/check-uk-visa",
    "/apply-first-provisional-driving-licence",
    "/book-driving-test",
    "/change-address-driving-licence",
    "/check-if-you-need-tax-return",
    "/sold-bought-vehicle",
    "/calculate-your-redundancy-pay",
    "/calculate-statutory-sick-pay",
    "/pension-credit-calculator",
]

SIZES = {400: 60, 800: 120}


def _known_paths() -> set[str]:
    """Return the paths already listed in the manifest."""
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return {row["path"] for row in csv.DictReader(fh)}


def _fetch_extra(paths: list[str]) -> list[dict[str, str]]:
    """Fetch pages to data/raw and return manifest rows for those that succeeded."""
    rows: list[dict[str, str]] = []
    headers = {"User-Agent": settings.govuk_user_agent}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for path in paths:
            resp = client.get(f"{CONTENT_API}{path}", timeout=30)
            if resp.status_code != 200:
                log.warning("Skipping %s (HTTP %d)", path, resp.status_code)
                continue
            data = resp.json()
            _raw_path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            rows.append(
                {
                    "path": path,
                    "title": data.get("title", ""),
                    "format": data.get("document_type", ""),
                    "public_timestamp": data.get("public_updated_at", ""),
                    "description": data.get("description", ""),
                }
            )
            time.sleep(1.0 / settings.govuk_rate_limit_rps)
    return rows


def _append_chunks(new_chunks: list[Chunk], size: int) -> int:
    """Append chunks whose ids are not yet in the chunk file; return how many were added."""
    existing = {c.chunk_id for c in load_chunks(size)}
    fresh = [c for c in new_chunks if c.chunk_id not in existing]
    with (PROCESSED_DIR / f"chunks_{size}.jsonl").open("a", encoding="utf-8") as fh:
        for chunk in fresh:
            fh.write(chunk.model_dump_json() + "\n")
    return len(fresh)


def extend_corpus(paths: list[str] = EXTRA_PATHS) -> dict[str, int]:
    """Fetch, clean and chunk the missing pages; append them to the manifest and chunk files."""
    todo = [p for p in paths if p not in _known_paths()]
    rows = _fetch_extra(todo)
    if rows:
        with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(
                fh, fieldnames=["path", "title", "format", "public_timestamp", "description"]
            ).writerows(rows)
    pages = [p for r in rows if (p := clean_page(_raw_path(r["path"])))]
    stats = {"pages_added": len(pages)}
    for size, overlap in SIZES.items():
        chunks = [c for page in pages for c in chunk_page(page, size, overlap)]
        stats[f"chunks_{size}_added"] = _append_chunks(chunks, size)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(extend_corpus())
