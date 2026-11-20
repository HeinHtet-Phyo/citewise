"""Convert raw GOV.UK Content API JSON into clean text sections."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# HTML tags we keep as plain-text markers.
HEADING_TAGS = {"h1", "h2", "h3", "h4"}

# Boilerplate phrases to strip from the start/end of section text.
BOILERPLATE = re.compile(
    r"^(Contents|Back to top|Print this page|Related content|Explore the topic"
    r"|Elsewhere on the web|Is this page useful\?|Help us improve GOV\.UK)",
    re.IGNORECASE,
)


# Service-style GOV.UK document types are stored as the "answer" format in the Chunk contract.
_FORMAT_MAP = {
    "smart_answer": "answer",
    "simple_smart_answer": "answer",
    "transaction": "answer",
    "local_transaction": "answer",
    "policy_paper": "answer",
}

# Text-bearing fields of pages that have no `body` or `parts` (services and calculators).
_FALLBACK_FIELDS = (
    "introductory_paragraph",
    "introduction",
    "need_to_know",
    "what_you_need_to_know",
    "more_information",
    "other_ways_to_apply",
    "hidden_search_terms",
)


@dataclass
class Section:
    """One section of a GOV.UK page."""

    heading: str | None
    text: str
    url: str  # canonical URL for this section (may include /part-slug)


@dataclass
class CleanPage:
    """A cleaned GOV.UK page, split into sections."""

    path: str
    title: str
    format: str
    updated_at: str
    sections: list[Section]


def _html_to_text(html: str) -> str:
    """Convert HTML to plain text, preserving headings and list structure."""
    soup = BeautifulSoup(html, "lxml")

    # Replace heading tags with their text preceded by a newline marker.
    for tag in soup.find_all(HEADING_TAGS):
        if isinstance(tag, Tag):
            tag.replace_with(f"\n## {tag.get_text(strip=True)}\n")

    # Replace list items with bullet markers.
    for li in soup.find_all("li"):
        if isinstance(li, Tag):
            li.replace_with(f"• {li.get_text(strip=True)}\n")

    # Replace table cells with tab-separated text.
    for td in soup.find_all(["td", "th"]):
        if isinstance(td, Tag):
            td.replace_with(f"{td.get_text(strip=True)}\t")
    for tr in soup.find_all("tr"):
        if isinstance(tr, Tag):
            tr.replace_with(tr.get_text() + "\n")

    text = soup.get_text(separator="\n")
    # Collapse 3+ blank lines into 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sections(text: str, base_url: str) -> list[Section]:
    """Split plain text at ## heading markers into sections."""
    parts = re.split(r"\n##\s+", text)
    sections: list[Section] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        if i == 0:
            # Text before the first heading — use None as heading.
            body = part
            heading = None
        else:
            heading = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""

        # Skip boilerplate-only sections.
        if BOILERPLATE.match(body):
            continue
        if not body:
            continue

        sections.append(Section(heading=heading, text=body, url=base_url))
    return sections


def _fallback_html(data: dict, details: dict) -> str:  # type: ignore[type-arg]
    """Join the description and service-page text fields into one HTML string."""
    pieces = [f"<p>{data.get('description', '')}</p>"]
    for field in _FALLBACK_FIELDS:
        value = details.get(field)
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        if value:
            pieces.append(f"<p>{value}</p>" if "<" not in str(value) else str(value))
    return "\n".join(pieces)


def clean_page(raw_path: Path) -> CleanPage | None:
    """
    Parse one raw Content API JSON file and return a CleanPage.

    Returns None if the page should be skipped (non-English, missing body, etc.).
    """
    try:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s: %s", raw_path, exc)
        return None

    path: str = data.get("base_path", "")
    title: str = data.get("title", "")
    doc_type: str = data.get("document_type", data.get("schema_name", ""))
    fmt = _FORMAT_MAP.get(doc_type, doc_type)
    updated_at: str = data.get("public_updated_at", "")[:10]  # ISO date only
    locale: str = data.get("locale", "en")

    if locale != "en":
        return None

    base_url = f"https://www.gov.uk{path}"
    details = data.get("details", {})
    sections: list[Section] = []

    # Guide: iterate over parts, each with its own URL.
    if "parts" in details:
        for part in details["parts"]:
            part_slug: str = part.get("slug", "")
            part_title: str = part.get("title", "")
            part_body: str = part.get("body", "")
            if not part_body:
                continue
            part_url = f"{base_url}/{part_slug}" if part_slug else base_url
            text = _html_to_text(part_body)
            # Treat the part title as the top-level heading.
            full_text = f"{title} — {part_title}\n{text}"
            sections.extend(_split_into_sections(full_text, part_url))

    # Answer / detailed_guide: single body field.
    elif "body" in details:
        body = details["body"]
        if isinstance(body, list):
            # Some pages have body as a list of content blocks.
            html = " ".join(b.get("content", "") for b in body if isinstance(b, dict))
        else:
            html = str(body)
        text = _html_to_text(html)
        sections = _split_into_sections(text, base_url)

    if not sections and doc_type in _FORMAT_MAP:
        sections = _split_into_sections(_html_to_text(_fallback_html(data, details)), base_url)

    if not sections:
        return None

    # Prepend title to the first section so chunks are self-contained.
    sections[0] = Section(
        heading=sections[0].heading,
        text=f"{title}\n\n{sections[0].text}",
        url=sections[0].url,
    )

    return CleanPage(
        path=path,
        title=title,
        format=fmt,
        updated_at=updated_at,
        sections=sections,
    )


def clean_all(raw_dir: Path = RAW_DIR) -> list[CleanPage]:
    """Clean all raw JSON files and return CleanPage objects."""
    paths = sorted(raw_dir.glob("*.json"))
    pages: list[CleanPage] = []
    skipped = 0
    for p in paths:
        page = clean_page(p)
        if page:
            pages.append(page)
        else:
            skipped += 1
    log.info("Cleaned %d pages (%d skipped)", len(pages), skipped)
    return pages
