"""Corpus helpers for building questions: page text, lexical search, verification."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks_400.jsonl")
_TOKEN_RE = re.compile(r"[a-z0-9£]+")
_NUM_RE = re.compile(r"£?\d[\d,]*(?:\.\d+)?%?")
_STOP = {"the", "a", "an", "of", "to", "and", "in", "is", "i", "my", "do", "can", "for", "what"}
_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine", "10": "ten", "12": "twelve",
}  # fmt: skip


def base_path(url: str) -> str:
    """Return the page base path ('/benefit-cap') for any GOV.UK URL or chunk id."""
    path = url.removeprefix("https://www.gov.uk").split("#")[0]
    return "/" + path.strip("/").split("/")[0]


def normalise(text: str) -> str:
    """Lower-case and flatten whitespace, quotes and dashes for robust matching."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"')
    text = text.replace("”", '"').replace("–", "-").replace("—", "-")
    text = text.replace(" ", " ").replace("•", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, object]]:
    """Read chunk records from JSONL."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


class Corpus:
    """In-memory page text plus a small BM25 index over chunks."""

    def __init__(self, chunks: list[dict[str, object]]) -> None:
        self.chunks = chunks
        self.urls = {str(c["url"]) for c in chunks}
        pages: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            pages[base_path(str(c["url"]))].append(str(c["text"]))
        self.page_text = {p: normalise(" ".join(t)) for p, t in pages.items()}
        self._docs = [self._tokens(str(c["title"]) + " " + str(c["text"])) for c in chunks]
        self._df: Counter[str] = Counter(t for d in self._docs for t in set(d))
        self._avg = sum(len(d) for d in self._docs) / len(self._docs)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]

    def search(self, query: str, k: int = 6, exclude: set[str] | None = None) -> list[int]:
        """Return indices of the top-k chunks by BM25, one chunk per URL."""
        q = self._tokens(query)
        n = len(self._docs)
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            tf = Counter(doc)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(doc) / self._avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        out: list[int] = []
        seen: set[str] = set(exclude or ())
        for _, i in scored:
            url = str(self.chunks[i]["url"])
            if url not in seen:
                seen.add(url)
                out.append(i)
            if len(out) == k:
                break
        return out

    def verify(self, answer: str, evidence: list[dict[str, str]]) -> str | None:
        """Return None if the answer is grounded in the evidence pages, else a failure reason."""
        if not evidence:
            return "no_evidence"
        pages_text = ""
        for ev in evidence:
            url, quote = ev.get("url", ""), normalise(ev.get("quote", ""))
            if url not in self.urls:
                return "gold_url_not_in_corpus"
            text = self.page_text.get(base_path(url), "")
            if len(quote) < 20:
                return "quote_too_short"
            if quote not in text:
                return "quote_not_in_page"
            pages_text += " " + text
        return _numbers_missing(answer, pages_text)


def _numbers_missing(answer: str, text: str) -> str | None:
    """Check every figure in the answer appears in the page text (digits or number words)."""
    flat = text.replace(",", "")
    for num in _NUM_RE.findall(normalise(answer)):
        bare = num.replace(",", "").rstrip(".")
        digits = bare.lstrip("£").rstrip("%")
        if bare in flat or digits in flat:
            continue
        if _WORDS.get(digits, "\0") in flat:
            continue
        return f"number_not_in_page:{num}"
    return None
