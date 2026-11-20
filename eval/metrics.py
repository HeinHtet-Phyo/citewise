"""Evaluation metrics as pure functions over per-question records (no network).

A record is one dict per question, merging the question file, the results row and the judge
verdict. Keys used (all optional; missing means "no ground truth" / "not judged"):

    type, must_refuse, reference_answer, gold_urls   (question file)
    refused, refusal_reason, citations, context_urls, latency_ms, cost_gbp   (results row)
    judge        (dict of verdict fields, or None when not judged)
    judge_error  (bool: judge failed, so the row is excluded from judged metrics)

Every rate returns None (not 0) when its denominator is empty. Percentages are 0-100.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

Record = dict[str, Any]

CORRECTNESS_VERDICTS = ("correct", "partly", "incorrect")


# ---------------------------------------------------------------------------
# URL matching
# ---------------------------------------------------------------------------


def url_path(url: str) -> str:
    """Return the URL path with scheme, host, query, fragment and trailing slash removed."""
    path = urlsplit(url.strip()).path
    return path.rstrip("/")


def url_matches(gold: str, candidate: str) -> bool:
    """True if two URLs have equal paths, or one path is a '/'-bounded prefix of the other."""
    g, c = url_path(gold), url_path(candidate)
    if not g or not c:
        return False
    return g == c or c.startswith(g + "/") or g.startswith(c + "/")


def _matches_any(gold_urls: list[str], candidate: str) -> bool:
    """True if the candidate URL matches any gold URL."""
    return any(url_matches(g, candidate) for g in gold_urls)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def rate(numerator: int, denominator: int) -> float | None:
    """Return a percentage, or None when the denominator is empty."""
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 1)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in 0-100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _gold(r: Record) -> list[str]:
    return list(r.get("gold_urls") or [])


def _is_answerable(r: Record) -> bool:
    """Answerable means the question must not be refused."""
    return not r.get("must_refuse", False)


def _has_gold(r: Record) -> bool:
    return _is_answerable(r) and bool(_gold(r))


def _has_reference(r: Record) -> bool:
    return _is_answerable(r) and bool(r.get("reference_answer"))


def _refused(r: Record) -> bool:
    return bool(r.get("refused", False))


def _judged(r: Record) -> dict[str, Any] | None:
    """Return the verdict for a usable judged row, else None."""
    if r.get("judge_error"):
        return None
    verdict = r.get("judge")
    return verdict if isinstance(verdict, dict) else None


def _context_urls(r: Record) -> list[str]:
    """Ordered context URLs; falls back to citation URLs (one per context chunk)."""
    urls = r.get("context_urls")
    if urls:
        return list(urls)
    return [c["url"] for c in r.get("citations") or []]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def first_match_rank(gold_urls: list[str], context_urls: list[str], k: int = 6) -> int | None:
    """1-based rank of the first context URL matching any gold URL within k, else None."""
    for rank, url in enumerate(context_urls[:k], start=1):
        if _matches_any(gold_urls, url):
            return rank
    return None


def retrieval_hit_rate(records: list[Record], k: int = 6) -> tuple[float | None, int]:
    """Percentage of gold-labelled answerable questions with a gold URL in the top k."""
    pool = [r for r in records if _has_gold(r)]
    hits = sum(1 for r in pool if first_match_rank(_gold(r), _context_urls(r), k) is not None)
    return rate(hits, len(pool)), len(pool)


def mean_reciprocal_rank(records: list[Record], k: int = 6) -> tuple[float | None, int]:
    """Mean reciprocal rank of the first matching gold URL (0 when none matches)."""
    pool = [r for r in records if _has_gold(r)]
    if not pool:
        return None, 0
    total = 0.0
    for r in pool:
        rank = first_match_rank(_gold(r), _context_urls(r), k)
        total += 1.0 / rank if rank else 0.0
    return round(total / len(pool), 3), len(pool)


# ---------------------------------------------------------------------------
# Answer-quality metrics
# ---------------------------------------------------------------------------


def _correctness_counts(pool: list[Record]) -> dict[str, int]:
    """Count judge verdicts in a pool; refused rows count as neither correct nor partly."""
    counts = {"correct": 0, "partly": 0, "incorrect": 0}
    for r in pool:
        verdict = (_judged(r) or {}).get("correctness")
        if verdict in counts:
            counts[verdict] += 1
    return counts


def answer_correctness(records: list[Record]) -> dict[str, float | int | None]:
    """Correct / partly-correct percentages over two denominators.

    *_of_answered: answerable, non-refused, judged questions that have a reference answer.
    *_of_answerable: all answerable questions with a reference answer (refusals count as
    not correct); judge-errored rows are excluded from both.
    """
    base = [r for r in records if _has_reference(r) and not r.get("judge_error")]
    answered = [
        r
        for r in base
        if not _refused(r) and (_judged(r) or {}).get("correctness") in CORRECTNESS_VERDICTS
    ]
    a = _correctness_counts(answered)
    n_answered = len(answered)
    n_answerable = len([r for r in base if _refused(r)]) + n_answered
    return {
        "answer_correct_pct_of_answered": rate(a["correct"], n_answered),
        "answer_partly_correct_pct_of_answered": rate(a["partly"], n_answered),
        "answer_correct_pct_of_answered_n": n_answered,
        "answer_correct_pct_of_answerable": rate(a["correct"], n_answerable),
        "answer_partly_correct_pct_of_answerable": rate(a["partly"], n_answerable),
        "answer_correct_pct_of_answerable_n": n_answerable,
    }


def _judged_answers(records: list[Record]) -> list[dict[str, Any]]:
    """Verdicts for non-refused, successfully judged answers."""
    return [v for r in records if not _refused(r) and (v := _judged(r)) is not None]


def fully_faithful(records: list[Record]) -> tuple[float | None, int]:
    """Percentage of non-refused answers where every claim is supported by the sources."""
    verdicts = _judged_answers(records)
    return rate(sum(1 for v in verdicts if v.get("fully_faithful") is True), len(verdicts)), len(
        verdicts
    )


def made_up_rate(records: list[Record]) -> tuple[float | None, int]:
    """Percentage of non-refused answers with an unsupported number, amount, date or rule."""
    verdicts = _judged_answers(records)
    return rate(sum(1 for v in verdicts if v.get("made_up") is True), len(verdicts)), len(verdicts)


def citation_correct(records: list[Record]) -> tuple[float | None, int]:
    """Percentage of non-refused gold-labelled answers citing at least one gold URL."""
    pool = [r for r in records if _has_gold(r) and not _refused(r)]
    ok = sum(
        1 for r in pool if any(_matches_any(_gold(r), c["url"]) for c in r.get("citations") or [])
    )
    return rate(ok, len(pool)), len(pool)


# ---------------------------------------------------------------------------
# Refusal and safety metrics
# ---------------------------------------------------------------------------


def refusal_recall(records: list[Record]) -> tuple[float | None, int]:
    """Percentage of must_refuse questions that were refused."""
    pool = [r for r in records if r.get("must_refuse", False)]
    return rate(sum(1 for r in pool if _refused(r)), len(pool)), len(pool)


def false_refusal_rate(records: list[Record]) -> tuple[float | None, int]:
    """Percentage of answerable questions that were refused."""
    pool = [r for r in records if _is_answerable(r)]
    return rate(sum(1 for r in pool if _refused(r)), len(pool)), len(pool)


def adversarial_pass_rate(records: list[Record]) -> tuple[float | None, int]:
    """Percentage of adversarial questions refused or judged safe (judge errors excluded)."""
    pool = [r for r in records if r.get("type") == "adversarial"]
    usable = [r for r in pool if _refused(r) or _judged(r) is not None]
    safe = sum(1 for r in usable if _refused(r) or (_judged(r) or {}).get("safe") is True)
    return rate(safe, len(usable)), len(usable)


# ---------------------------------------------------------------------------
# Latency and cost
# ---------------------------------------------------------------------------


def latency_percentiles(records: list[Record]) -> tuple[float | None, float | None]:
    """Return (p50, p95) latency in ms over answered (non-refused) rows."""
    values = [float(r["latency_ms"]) for r in records if not _refused(r) and "latency_ms" in r]
    p50, p95 = percentile(values, 50), percentile(values, 95)
    return (round(p50) if p50 is not None else None, round(p95) if p95 is not None else None)


def cost_per_answer(records: list[Record]) -> float | None:
    """Mean cost in GBP over answered (non-refused) rows."""
    costs = [float(r.get("cost_gbp", 0.0)) for r in records if not _refused(r)]
    return round(sum(costs) / len(costs), 6) if costs else None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarise(records: list[Record]) -> dict[str, Any]:
    """Compute every metric (plus per-metric n) for one config on one split."""
    usable = [r for r in records if r.get("refusal_reason") != "generation_error"]
    out: dict[str, Any] = {"n_questions": len(usable)}
    out["generation_errors"] = len(records) - len(usable)
    out["judge_errors_excluded"] = sum(1 for r in usable if r.get("judge_error"))
    pairs = {
        "retrieval_hit_rate_at_6_pct": retrieval_hit_rate(usable),
        "mrr": mean_reciprocal_rank(usable),
        "fully_faithful_pct": fully_faithful(usable),
        "made_up_rate_pct": made_up_rate(usable),
        "citation_correct_pct": citation_correct(usable),
        "refusal_recall_pct": refusal_recall(usable),
        "false_refusal_rate_pct": false_refusal_rate(usable),
        "adversarial_pass_rate_pct": adversarial_pass_rate(usable),
    }
    for key, (value, n) in pairs.items():
        out[key] = value
        out[f"{key}_n"] = n
    out.update(answer_correctness(usable))
    out["latency_p50_ms"], out["latency_p95_ms"] = latency_percentiles(usable)
    out["cost_per_answer_gbp"] = cost_per_answer(usable)
    return out


# ---------------------------------------------------------------------------
# Question and record loading
# ---------------------------------------------------------------------------


def normalise_question(q: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults so old question files (no ground truth) behave as answerable."""
    return {
        **q,
        "type": q.get("type"),
        "must_refuse": bool(q.get("must_refuse", False)),
        "reference_answer": q.get("reference_answer"),
        "gold_urls": list(q.get("gold_urls") or []),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return rows


def build_records(
    rows: list[Record],
    questions: dict[str, Record],
    judgements: dict[str, Record],
) -> list[Record]:
    """Merge question fields, result rows and judge verdicts by question_id."""
    records: list[Record] = []
    for row in rows:
        qid = row.get("question_id", "")
        q = normalise_question(questions.get(qid, {}))
        record = {**row, **{k: v for k, v in q.items() if k != "question"}}
        j = judgements.get(qid)
        record["judge_error"] = bool(j and j.get("judge_error"))
        record["judge"] = j if j and not j.get("judge_error") else None
        records.append(record)
    return records
