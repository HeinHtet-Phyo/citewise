"""Fetch raw GOV.UK Content API JSON for each page in the manifest."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from citewise.config import settings

log = logging.getLogger(__name__)

CONTENT_API = "https://www.gov.uk/api/content"
MANIFEST_PATH = Path("data/manifest.csv")
RAW_DIR = Path("data/raw")
FAILURES_PATH = Path("data/fetch_failures.csv")


def _raw_path(page_path: str) -> Path:
    """Return the local cache path for a GOV.UK content path."""
    # Strip leading slash; replace remaining slashes with __ for safe filenames.
    safe = page_path.lstrip("/").replace("/", "__")
    return RAW_DIR / f"{safe}.json"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def _fetch_one(client: httpx.Client, path: str) -> dict:  # type: ignore[type-arg]
    """Fetch a single page from the Content API."""
    url = f"{CONTENT_API}{path}"
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def fetch(force: bool = False) -> tuple[int, int]:
    """
    Fetch each page in the manifest from the Content API, caching to data/raw/.

    Args:
        force: re-fetch even if a cached file exists.

    Returns:
        (success_count, failure_count)
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest()
    if not rows:
        raise RuntimeError(f"Manifest not found or empty: {MANIFEST_PATH}")

    headers = {"User-Agent": settings.govuk_user_agent}
    min_interval = 1.0 / settings.govuk_rate_limit_rps

    successes = 0
    failures: list[dict[str, str]] = []

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for i, row in enumerate(rows):
            path = row["path"]
            dest = _raw_path(path)

            if dest.exists() and not force:
                successes += 1
                continue

            last = time.monotonic()
            try:
                data = _fetch_one(client, path)
                dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                successes += 1
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", path, exc)
                failures.append({"path": path, "error": str(exc)})

            # Rate limiting: sleep if we're going faster than the limit.
            elapsed = time.monotonic() - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            if (i + 1) % 500 == 0:
                log.info("  %d/%d fetched (%d failures)", i + 1, len(rows), len(failures))

    _write_failures(failures)
    log.info(
        "Fetch complete: %d succeeded, %d failed (see %s)",
        successes,
        len(failures),
        FAILURES_PATH,
    )
    return successes, len(failures)


def _load_manifest() -> list[dict[str, str]]:
    """Load the manifest CSV into a list of dicts."""
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_failures(failures: list[dict[str, str]]) -> None:
    """Write failed fetches to a CSV for inspection."""
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "error"])
        writer.writeheader()
        writer.writerows(failures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ok, fail = fetch()
    print(f"Done. {ok} succeeded, {fail} failed.")
