"""Split cleaned GOV.UK pages into overlapping token chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from citewise.ingest.clean import CleanPage, Section, clean_all
from citewise.models import Chunk

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
_TOKENISER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Return the number of tokens in text using cl100k_base."""
    return len(_TOKENISER.encode(text))


def _make_chunk_id(base_path: str, section_slug: str, n: int) -> str:
    """Create a stable, human-readable chunk ID."""
    slug = re.sub(r"[^a-z0-9-]", "-", section_slug.lower())[:40].strip("-")
    return f"{base_path}#{slug}-{n}"


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries and newlines."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and len(part) < 30 and not part.endswith((".", "!", "?")):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return merged


@dataclass
class _SectionGroup:
    """A group of adjacent sections merged for chunking."""

    sections: list[Section]
    body: str  # formatted body text (headings inline)
    tokens: int  # token count of body alone (without page-title prefix)
    url: str  # URL of first section
    first_heading: str | None


def _format_section(section: Section) -> str:
    """Format a section as heading + body for inclusion in a merged chunk."""
    if section.heading:
        return f"## {section.heading}\n{section.text}"
    return section.text


def _merge_sections(
    sections: list[Section],
    max_body_tokens: int,
) -> list[_SectionGroup]:
    """
    Greedily merge adjacent sections into groups that fit within max_body_tokens.

    Rules:
    - Sections from different guide parts (different URLs) are never merged,
      so citations always point to the correct page.
    - Each iteration advances i by at least one, guaranteeing termination.
    - Overlap between groups is handled by sentence-splitting inside _split_group.
    """
    groups: list[_SectionGroup] = []
    i = 0

    while i < len(sections):
        anchor_url = sections[i].url
        buffer: list[Section] = []
        buffer_tokens = 0

        while i < len(sections):
            sec = sections[i]
            sec_text = _format_section(sec)
            sec_tokens = _count_tokens(sec_text)
            sep_tokens = 2 if buffer else 0  # "\n\n" between sections

            # Stop at a URL boundary (different guide part).
            if sec.url != anchor_url:
                break
            # Stop if adding this section would overflow (but always take at least one).
            if buffer and buffer_tokens + sep_tokens + sec_tokens > max_body_tokens:
                break

            buffer.append(sec)
            buffer_tokens += sep_tokens + sec_tokens
            i += 1

        body = "\n\n".join(_format_section(s) for s in buffer)
        groups.append(
            _SectionGroup(
                sections=buffer,
                body=body,
                tokens=_count_tokens(body),
                url=buffer[0].url,
                first_heading=buffer[0].heading,
            )
        )

    return groups


def _split_group(
    group: _SectionGroup,
    page_title: str,
    base_path: str,
    page_format: str,
    updated_at: str,
    max_tokens: int,
    overlap_tokens: int,
    chunk_index_start: int,
) -> list[Chunk]:
    """
    Sentence-split a single _SectionGroup that exceeds max_tokens.

    Used as a fallback when a merged group is still too large.
    """
    slug = re.sub(r"\s+", "-", group.first_heading.lower())[:40] if group.first_heading else "intro"
    prefix = f"{page_title}\n\n"
    prefix_tokens = _count_tokens(prefix)
    effective_max = max_tokens - prefix_tokens

    sentences = _split_sentences(group.body)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    idx = chunk_index_start

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if sent_tokens > effective_max:
            if current:
                text = prefix + " ".join(current)
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(base_path, slug, idx),
                        url=group.url,
                        title=page_title,
                        section=group.first_heading,
                        text=text,
                        format=page_format,
                        updated_at=updated_at,
                        tokens=_count_tokens(text),
                    )
                )
                idx += 1
                current, current_tokens = [], 0
            text = prefix + sent
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(base_path, slug, idx),
                    url=group.url,
                    title=page_title,
                    section=group.first_heading,
                    text=text,
                    format=page_format,
                    updated_at=updated_at,
                    tokens=_count_tokens(text),
                )
            )
            idx += 1
            continue

        if current and current_tokens + sent_tokens > effective_max:
            text = prefix + " ".join(current)
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(base_path, slug, idx),
                    url=group.url,
                    title=page_title,
                    section=group.first_heading,
                    text=text,
                    format=page_format,
                    updated_at=updated_at,
                    tokens=_count_tokens(text),
                )
            )
            idx += 1
            overlap: list[str] = []
            overlap_total = 0
            for s in reversed(current):
                t = _count_tokens(s)
                if overlap_total + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_total += t
            current, current_tokens = overlap, overlap_total

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        text = prefix + " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(base_path, slug, idx),
                url=group.url,
                title=page_title,
                section=group.first_heading,
                text=text,
                format=page_format,
                updated_at=updated_at,
                tokens=_count_tokens(text),
            )
        )

    return chunks


def chunk_page(
    page: CleanPage,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """
    Chunk a CleanPage by first merging adjacent short sections, then sentence-splitting
    any groups that still exceed max_tokens.

    Strategy:
    1. Greedily merge adjacent sections into groups that fit within max_tokens
       (keeping section headings inline as ## markers).
    2. Groups that fit → emit directly as one chunk.
    3. Groups that overflow (single very long section) → sentence-split with overlap.

    This ensures the 400- and 800-token sizes produce chunks that differ meaningfully
    in size and context density.
    """
    if not page.sections:
        return []

    prefix = f"{page.title}\n\n"
    prefix_tokens = _count_tokens(prefix)
    max_body_tokens = max_tokens - prefix_tokens

    groups = _merge_sections(page.sections, max_body_tokens)

    all_chunks: list[Chunk] = []
    slug_counter: dict[str, int] = {}

    for group in groups:
        slug = (
            re.sub(r"\s+", "-", group.first_heading.lower())[:40]
            if group.first_heading
            else "intro"
        )
        idx = slug_counter.get(slug, 0)

        total_tokens = prefix_tokens + group.tokens

        if total_tokens <= max_tokens:
            # Group fits — emit as a single chunk.
            text = prefix + group.body
            all_chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(page.path, slug, idx),
                    url=group.url,
                    title=page.title,
                    section=group.first_heading,
                    text=text,
                    format=page.format,
                    updated_at=page.updated_at,
                    tokens=_count_tokens(text),
                )
            )
            slug_counter[slug] = idx + 1
        else:
            # Group too large — sentence-split.
            split_chunks = _split_group(
                group=group,
                page_title=page.title,
                base_path=page.path,
                page_format=page.format,
                updated_at=page.updated_at,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                chunk_index_start=idx,
            )
            all_chunks.extend(split_chunks)
            slug_counter[slug] = idx + len(split_chunks)

    return all_chunks


def chunk_all(
    max_tokens: int = 400,
    overlap_tokens: int = 60,
    raw_dir: Path | None = None,
) -> list[Chunk]:
    """
    Clean and chunk all raw pages; write data/processed/chunks_{max_tokens}.jsonl.

    Returns all Chunk objects.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"chunks_{max_tokens}.jsonl"

    kwargs = {"raw_dir": raw_dir} if raw_dir else {}
    pages = clean_all(**kwargs)
    all_chunks: list[Chunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page, max_tokens=max_tokens, overlap_tokens=overlap_tokens))

    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(chunk.model_dump_json() + "\n")

    tokens = [c.tokens for c in all_chunks]
    avg = sum(tokens) / len(tokens) if tokens else 0
    log.info(
        "Chunked %d pages → %d chunks (avg %.0f tokens) → %s",
        len(pages),
        len(all_chunks),
        avg,
        output_path,
    )
    return all_chunks


def load_chunks(chunk_size: int = 400) -> list[Chunk]:
    """Load chunks from the processed JSONL file."""
    path = PROCESSED_DIR / f"chunks_{chunk_size}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}. Run `make ingest` first.")
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks_400 = chunk_all(max_tokens=400, overlap_tokens=60)
    chunks_800 = chunk_all(max_tokens=800, overlap_tokens=120)
    print(f"400-token chunks: {len(chunks_400)}")
    print(f"800-token chunks: {len(chunks_800)}")
