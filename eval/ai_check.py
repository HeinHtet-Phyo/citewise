"""Independent AI check of reference answers (no human validation).

Two checks per answerable question, both against the text of the gold URL page(s):
1. Programmatic: every number in the reference answer appears in the page text.
2. A third-family model (Gemma, not Qwen or DeepSeek) reads the page and says whether the
   reference answer is fully supported and actually answers the question.
A question passes only if both checks pass. The result is a machine check, not a human one.

Usage:
    python eval/ai_check.py            # check every answerable question, write ai_check.json
    python eval/ai_check.py --csv      # also fill ai_check columns in hand_check_questions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import litellm
from generate_candidates import _load_questions
from metrics import url_matches

from citewise.config import settings
from citewise.generate import _cache_path, _prompt_hash
from citewise.ingest.chunk import load_chunks

log = logging.getLogger(__name__)

CHECK_MODEL = "openai/google/gemma-3-27b-it"
PROMPT_VERSION = "aicheck-v1"
QUESTIONS_DIR = Path(__file__).parent / "questions"
MAX_PAGE_CHARS = 24000
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
litellm.suppress_debug_info = True

_PROMPT = """You check a reference answer for a question about UK government guidance.
You get the text of the gold page(s) and a reference answer.
Decide two things using ONLY the page text:
- supported: every fact and figure in the reference answer is stated in the page text.
- answers_question: the reference answer directly answers the question.
Reply with JSON only: {"supported": true|false, "answers_question": true|false, "reason": "<one short sentence>"}"""


def page_text(gold_urls: list[str], pages: dict[str, str]) -> str:
    """Concatenate the text of every page whose URL matches a gold URL."""
    parts = [text for url, text in pages.items() if any(url_matches(g, url) for g in gold_urls)]
    return "\n\n".join(parts)


def _normalise(text: str) -> str:
    """Lower-case and drop thousands separators so '£1,000' and '1000' compare equal."""
    return re.sub(r"(?<=\d),(?=\d)", "", text.lower())


def missing_figures(reference: str, text: str) -> list[str]:
    """Numbers in the reference answer that do not appear in the page text."""
    haystack = _normalise(text)
    numbers = {_normalise(n).rstrip(".,") for n in _NUMBER_RE.findall(reference)}
    return sorted(n for n in numbers if n and n not in haystack)


def _ask_model(question: str, reference: str, text: str) -> dict[str, Any]:
    """Ask the third-family model (cached) and return its parsed JSON verdict."""
    user = (
        f"{_PROMPT}\n\nQuestion: {question}\n\nReference answer: {reference}\n\n"
        f"Page text:\n{text[:MAX_PAGE_CHARS]}"
    )
    messages = [{"role": "user", "content": user}]
    path = _cache_path(settings.llm_cache_dir, _prompt_hash(CHECK_MODEL + PROMPT_VERSION, messages))
    if settings.llm_cache_enabled and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))["text"]
    else:
        resp = litellm.completion(
            model=CHECK_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
            timeout=90,
        )
        raw = resp.choices[0].message.content or ""
        if settings.llm_cache_enabled:
            Path(settings.llm_cache_dir).mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": raw}), encoding="utf-8")
    start = raw.find("{")
    obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    return dict(obj)


def check_question(q: dict[str, Any], pages: dict[str, str]) -> dict[str, Any]:
    """Run both checks on one answerable question."""
    text = page_text(q["gold_urls"], pages)
    if not text:
        return {
            "question_id": q["question_id"],
            "ai_check": "fail",
            "reason": "gold page not found",
        }
    missing = missing_figures(q["reference_answer"], text)
    try:
        verdict = _ask_model(q["question"], q["reference_answer"], text)
    except Exception as exc:  # network or malformed JSON: record as an error, not a pass
        return {"question_id": q["question_id"], "ai_check": "error", "reason": str(exc)[:120]}
    problems = []
    if missing:
        problems.append("figures not in page: " + ", ".join(missing))
    if not verdict.get("supported"):
        problems.append("model: not supported")
    if not verdict.get("answers_question"):
        problems.append("model: does not answer the question")
    reason = "; ".join(problems) if problems else "figures found in page; model agrees"
    if problems and verdict.get("reason"):
        reason += f" ({verdict['reason']})"
    return {
        "question_id": q["question_id"],
        "ai_check": "fail" if problems else "pass",
        "reason": reason,
    }


def fill_csv(results: dict[str, dict[str, Any]]) -> None:
    """Rename human_ok to ai_check and fill it, with a reason column."""
    path = QUESTIONS_DIR / "hand_check_questions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    fields = [f for f in rows[0] if f != "human_ok"] + ["ai_check", "ai_check_reason"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            res = results.get(r["question_id"], {})
            r.pop("human_ok", None)
            r["ai_check"], r["ai_check_reason"] = res.get("ai_check", ""), res.get("reason", "")
            writer.writerow(r)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()
    pages: dict[str, str] = {}
    for c in load_chunks(400):
        pages[c.url] = pages.get(c.url, "") + "\n" + c.text
    questions = [q for q in _load_questions("all") if not q["must_refuse"]]
    results = {q["question_id"]: check_question(q, pages) for q in questions}
    (QUESTIONS_DIR / "ai_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r["ai_check"]] = counts.get(r["ai_check"], 0) + 1
    log.info("AI check: %s", counts)
    if args.csv:
        fill_csv(results)


if __name__ == "__main__":
    main()
