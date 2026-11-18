"""Split cleaned GOV.UK pages into overlapping token chunks."""

from __future__ import annotations

import logging
import re
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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving bullet points as atomic units."""
    # First split on sentence-ending punctuation or newlines (covers bullet lists
    # and structured content that lacks terminal punctuation).
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    # Rejoin very short fragments with the previous sentence.
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


def _make_chunk_id(base_path: str, section_slug: str, n: int) -> str:
    """Create a stable, human-readable chunk ID."""
    slug = re.sub(r"[^a-z0-9-]", "-", section_slug.lower())[:40].strip("-")
    return f"{base_path}#{slug}-{n}"


def chunk_section(
    section: Section,
    page_title: str,
    base_path: str,
    page_format: str,
    updated_at: str,
    max_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """
    Split one section into overlapping token-bounded chunks.

    Each chunk starts with the page title and section heading so it makes
    sense when retrieved without surrounding context.
    """
    heading = section.heading or ""
    # Prefix prepended to every chunk from this section.
    prefix = f"{page_title}\n{heading}\n\n" if heading else f"{page_title}\n\n"
    prefix_tokens = _count_tokens(prefix)
    effective_max = max_tokens - prefix_tokens

    sentences = _split_sentences(section.text)
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    section_slug = re.sub(r"\s+", "-", heading.lower())[:40] if heading else "intro"
    chunk_index = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        # If a single sentence exceeds the limit, force-split it.
        if sent_tokens > effective_max:
            # Flush current buffer first.
            if current_sentences:
                text = prefix + " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(base_path, section_slug, chunk_index),
                        url=section.url,
                        title=page_title,
                        section=heading or None,
                        text=text,
                        format=page_format,
                        updated_at=updated_at,
                        tokens=_count_tokens(text),
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_tokens = 0
            # Emit the long sentence as its own chunk.
            text = prefix + sent
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(base_path, section_slug, chunk_index),
                    url=section.url,
                    title=page_title,
                    section=heading or None,
                    text=text,
                    format=page_format,
                    updated_at=updated_at,
                    tokens=_count_tokens(text),
                )
            )
            chunk_index += 1
            continue

        if current_tokens + sent_tokens > effective_max and current_sentences:
            # Emit the current buffer.
            text = prefix + " ".join(current_sentences)
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(base_path, section_slug, chunk_index),
                    url=section.url,
                    title=page_title,
                    section=heading or None,
                    text=text,
                    format=page_format,
                    updated_at=updated_at,
                    tokens=_count_tokens(text),
                )
            )
            chunk_index += 1
            # Keep the overlap: retain trailing sentences whose total <= overlap_tokens.
            overlap: list[str] = []
            overlap_total = 0
            for s in reversed(current_sentences):
                t = _count_tokens(s)
                if overlap_total + t > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_total += t
            current_sentences = overlap
            current_tokens = overlap_total

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Emit remaining sentences.
    if current_sentences:
        text = prefix + " ".join(current_sentences)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(base_path, section_slug, chunk_index),
                url=section.url,
                title=page_title,
                section=heading or None,
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
    """Chunk all sections of a CleanPage."""
    all_chunks: list[Chunk] = []
    for section in page.sections:
        all_chunks.extend(
            chunk_section(
                section=section,
                page_title=page.title,
                base_path=page.path,
                page_format=page.format,
                updated_at=page.updated_at,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
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

    # Token statistics.
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
