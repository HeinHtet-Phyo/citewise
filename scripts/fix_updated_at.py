"""Backfill updated_at in Qdrant from updated_at (not public_updated_at).

Scrolls every point in the collection, looks up the source raw JSON by base_path,
and sets the updated_at payload field where it differs. Vectors are untouched.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from citewise.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLLECTION = "citewise_400"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def _build_date_map() -> dict[str, str]:
    """Return {base_path: updated_at} from every cached raw JSON."""
    date_map: dict[str, str] = {}
    for f in RAW_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            base = d.get("base_path", "")
            if not base:
                continue
            new_date = (d.get("updated_at") or d.get("public_updated_at") or "")[:10]
            if new_date:
                date_map[base] = new_date
        except Exception as exc:
            log.warning("skip %s: %s", f, exc)
    return date_map


def _base_path_from_url(url: str) -> str:
    """Extract /base-path from https://www.gov.uk/base-path[/part]."""
    path = url.removeprefix("https://www.gov.uk")
    # Guide parts have URLs like /statutory-sick-pay/overview; the raw JSON is at /statutory-sick-pay
    return "/" + path.lstrip("/").split("/")[0]


def main() -> None:
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    date_map = _build_date_map()
    log.info("Loaded date map for %d pages", len(date_map))

    # Collect all points, group IDs by their required new updated_at
    updates: dict[str, list[str]] = {}  # new_date -> [point_id, ...]
    skipped = unchanged = 0
    offset = None

    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=None,
            limit=1000,
            offset=offset,
            with_payload=["url", "updated_at"],
            with_vectors=False,
        )
        for pt in result:
            payload = pt.payload or {}
            url: str = payload.get("url", "")
            current_date: str = payload.get("updated_at", "")
            base = _base_path_from_url(url)
            new_date = date_map.get(base)
            if new_date is None:
                skipped += 1
                continue
            if new_date == current_date:
                unchanged += 1
                continue
            updates.setdefault(new_date, []).append(str(pt.id))
        if offset is None:
            break

    total_to_update = sum(len(ids) for ids in updates.values())
    log.info(
        "Points to update: %d  unchanged: %d  no raw json: %d",
        total_to_update, unchanged, skipped,
    )

    if not updates:
        log.info("Nothing to do.")
        return

    # Apply updates grouped by new date (one API call per unique date value).
    # Use set_payload (partial update) not overwrite_payload (full replacement).
    done = 0
    for new_date, ids in sorted(updates.items()):
        client.set_payload(
            collection_name=COLLECTION,
            payload={"updated_at": new_date},
            points=ids,
        )
        done += len(ids)
        log.info("  set %s on %d points (%d / %d done)", new_date, len(ids), done, total_to_update)

    log.info("Done. Updated %d points across %d date values.", total_to_update, len(updates))


if __name__ == "__main__":
    main()
