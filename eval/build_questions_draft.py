"""Prompts and drafting for answerable questions (existing and newly generated)."""

from __future__ import annotations

import random
import re

from build_questions_corpus import Corpus
from build_questions_llm import draft_json

_SYS_ANSWER = (
    "You write ground-truth for a question-answering evaluation using ONLY the numbered GOV.UK "
    "passages supplied. Reply with one JSON object and nothing else:\n"
    '{"answerable": bool, "type": "single_fact"|"multi_page"|"how_to", '
    '"reference_answer": "1-3 sentences, British spelling, copying figures exactly", '
    '"evidence": [{"url": "<passage url>", "quote": "<verbatim excerpt, 40-200 characters, '
    'copied exactly from that passage>"}]}\n'
    "Rules: answerable is false if the passages do not clearly answer the question for England "
    "or the UK. Use 1 evidence item for single_fact or how_to (type how_to for 'how do I' "
    "process questions), 2 items from two different pages for multi_page. Never use outside "
    "knowledge."
)

_SYS_NEW = (
    "You write evaluation questions for a GOV.UK guidance assistant using ONLY the numbered "
    "passages supplied. Reply with one JSON object and nothing else:\n"
    '{"usable": bool, "topic": "benefits"|"employment"|"housing"|"transport", "question": '
    '"a natural, specific, self-contained question a member of the public might ask", '
    '"type": "single_fact"|"multi_page"|"how_to", "reference_answer": "1-3 sentences, copying '
    'figures exactly", "evidence": [{"url": "<passage url>", "quote": "<verbatim excerpt, '
    '40-200 characters, copied exactly from that passage>"}]}\n'
    "Topics: benefits (benefits, pensions, statutory pay, childcare support), employment (work "
    "rights, pay, dismissal), housing (renting, buying, tenancy, planning), transport (driving "
    "licences, vehicles, MOT, tax, fines). usable is false if the passages fit none of these "
    "topics, or the question would need outside knowledge. Do not say 'the passage' in the "
    "question. For multi_page the question must need facts from two passages (two evidence "
    "items from different urls); otherwise use one evidence item."
)


_TOPIC_RE = re.compile(
    r"benefit|allowance|credit|pension|childcare|child |pay\b|leave|redundan|dismiss|wage|"
    r"employ|worker|tenan|landlord|rent|housing|mortgage|stamp duty|planning|buy|lease|"
    r"driving|driver|licen[cs]e|vehicle|MOT|tax your|number plate|penalt|blue badge|"
    r"holiday|sick|maternity|paternity|disab|carer|council tax",
    re.IGNORECASE,
)
_SKIP_RE = re.compile(r"visa|business|corporation|import|export|company|alcohol|apprentice", re.I)


def _format_passages(corpus: Corpus, idx: list[int]) -> str:
    parts = []
    for n, i in enumerate(idx, start=1):
        c = corpus.chunks[i]
        text = str(c["text"])[:1800]
        parts.append(f"[{n}] url: {c['url']}\ntitle: {c['title']}\n{text}")
    return "\n\n".join(parts)


def draft_existing(corpus: Corpus, question: str) -> dict[str, object] | None:
    """Draft reference answer and evidence for an existing question from BM25 passages."""
    idx = corpus.search(question, k=5)
    return draft_json(_SYS_ANSWER, f"{_format_passages(corpus, idx)}\n\nQuestion: {question}")


def draft_new(
    corpus: Corpus, seed_idx: int, want: str, with_related: bool
) -> dict[str, object] | None:
    """Draft a new question from a seed chunk (plus related pages for multi_page)."""
    seed = corpus.chunks[seed_idx]
    idx = [seed_idx]
    if with_related:
        used = {str(seed["url"])}
        idx += corpus.search(str(seed["title"]), k=2, exclude=used)
    return draft_json(_SYS_NEW, f"{_format_passages(corpus, idx)}\n\nRequested type: {want}")


def pick_seeds(corpus: Corpus, n: int, rng: random.Random) -> list[int]:
    """Choose n seed chunks from distinct pages, favouring substantial English text."""
    pool: dict[str, list[int]] = {}
    for i, c in enumerate(corpus.chunks):
        text = str(c["text"])
        title = str(c["title"])
        if (
            int(str(c["tokens"])) >= 200
            and " the " in text
            and _TOPIC_RE.search(title)
            and not _SKIP_RE.search(title)
        ):
            pool.setdefault(str(c["url"]).split("/")[3], []).append(i)
    pages = sorted(pool)
    rng.shuffle(pages)
    return [rng.choice(pool[p]) for p in pages[:n]]
