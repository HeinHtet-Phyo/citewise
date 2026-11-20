"""Unit tests for eval/metrics.py using small synthetic records (no network)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))

import metrics as m

G = "https://www.gov.uk/universal-credit"
OTHER = "https://www.gov.uk/other-page"


def rec(**kw: Any) -> dict[str, Any]:
    """Build a record with sensible answerable defaults."""
    base: dict[str, Any] = {
        "type": "single_fact",
        "must_refuse": False,
        "gold_urls": [G],
        "reference_answer": "ref",
        "refused": False,
        "context_urls": [OTHER, G + "/what-youll-get"],
        "citations": [{"url": OTHER}],
        "latency_ms": 100,
        "cost_gbp": 0.01,
        "judge": None,
        "judge_error": False,
    }
    return {**base, **kw}


def verdict(**kw: Any) -> dict[str, Any]:
    return {
        "correctness": "correct",
        "fully_faithful": True,
        "made_up": False,
        "safe": True,
        **kw,
    }


@pytest.mark.parametrize(
    ("gold", "cand", "expected"),
    [
        ("https://www.gov.uk/a", "https://www.gov.uk/a", True),
        ("https://www.gov.uk/a/", "http://gov.uk/a?x=1#frag", True),
        ("/universal-credit", "https://www.gov.uk/universal-credit/what-youll-get", True),
        ("https://www.gov.uk/universal-credit/what-youll-get", G, True),
        ("https://www.gov.uk/benefits", "https://www.gov.uk/benefits-calculators", False),
        ("https://www.gov.uk/a", "https://www.gov.uk/b", False),
        ("", "https://www.gov.uk/a", False),
    ],
)
def test_url_matches(gold: str, cand: str, expected: bool) -> None:
    assert m.url_matches(gold, cand) is expected


def test_hit_rate_and_mrr() -> None:
    records = [
        rec(),  # gold at rank 2
        rec(context_urls=[G, OTHER]),  # rank 1
        rec(context_urls=[OTHER]),  # miss
        rec(must_refuse=True),  # excluded
        rec(gold_urls=[]),  # excluded
    ]
    assert m.retrieval_hit_rate(records) == (66.7, 3)
    assert m.mean_reciprocal_rank(records) == (0.5, 3)


def test_hit_rate_respects_k() -> None:
    assert m.retrieval_hit_rate([rec()], k=1) == (0.0, 1)


def test_empty_denominators_return_none() -> None:
    assert m.retrieval_hit_rate([]) == (None, 0)
    assert m.mean_reciprocal_rank([]) == (None, 0)
    assert m.fully_faithful([rec(refused=True)]) == (None, 0)
    assert m.refusal_recall([rec()]) == (None, 0)
    assert m.adversarial_pass_rate([rec()]) == (None, 0)
    assert m.cost_per_answer([]) is None
    assert m.latency_percentiles([]) == (None, None)


def test_correctness_denominators() -> None:
    records = [
        rec(judge=verdict()),
        rec(judge=verdict(correctness="partly")),
        rec(judge=verdict(correctness="incorrect")),
        rec(refused=True),
        rec(judge_error=True),  # excluded entirely
        rec(reference_answer=None, judge=verdict(correctness="n/a")),  # no ground truth
    ]
    out = m.answer_correctness(records)
    assert out["answer_correct_pct_of_answered"] == 33.3
    assert out["answer_partly_correct_pct_of_answered"] == 33.3
    assert out["answer_correct_pct_of_answered_n"] == 3
    assert out["answer_correct_pct_of_answerable"] == 25.0
    assert out["answer_correct_pct_of_answerable_n"] == 4


def test_correctness_empty() -> None:
    out = m.answer_correctness([rec(reference_answer=None)])
    assert out["answer_correct_pct_of_answered"] is None
    assert out["answer_correct_pct_of_answerable"] is None


def test_faithful_and_made_up_exclude_refused_and_errors() -> None:
    records = [
        rec(judge=verdict()),
        rec(judge=verdict(fully_faithful=False, made_up=True)),
        rec(refused=True),
        rec(judge_error=True),
    ]
    assert m.fully_faithful(records) == (50.0, 2)
    assert m.made_up_rate(records) == (50.0, 2)


def test_citation_correct() -> None:
    records = [
        rec(citations=[{"url": OTHER}, {"url": G + "/eligibility"}]),
        rec(),  # cites only OTHER
        rec(refused=True),
        rec(must_refuse=True),
    ]
    assert m.citation_correct(records) == (50.0, 2)


def test_refusal_metrics() -> None:
    records = [
        rec(must_refuse=True, refused=True),
        rec(must_refuse=True, refused=False),
        rec(refused=True),
        rec(),
        rec(),
        rec(),
    ]
    assert m.refusal_recall(records) == (50.0, 2)
    assert m.false_refusal_rate(records) == (25.0, 4)


def test_adversarial_pass_rate() -> None:
    adv = {"type": "adversarial", "must_refuse": True}
    records = [
        rec(**adv, refused=True),
        rec(**adv, judge=verdict(safe=True)),
        rec(**adv, judge=verdict(safe=False)),
        rec(**adv, judge_error=True),  # excluded
    ]
    assert m.adversarial_pass_rate(records) == (66.7, 3)


def test_percentile_and_latency() -> None:
    assert m.percentile([10, 20, 30, 40], 50) == 25.0
    assert m.percentile([5], 95) == 5
    assert m.percentile([0, 100], 95) == 95.0
    records = [rec(latency_ms=v) for v in (100, 200, 300, 400, 500)] + [rec(refused=True)]
    assert m.latency_percentiles(records) == (300, 480)


def test_cost_per_answer_ignores_refused() -> None:
    records = [rec(cost_gbp=0.01), rec(cost_gbp=0.03), rec(refused=True, cost_gbp=0.0)]
    assert m.cost_per_answer(records) == 0.02


def test_summarise_keys_and_generation_errors() -> None:
    records = [rec(judge=verdict()), rec(refusal_reason="generation_error", refused=True)]
    out = m.summarise(records)
    assert out["n_questions"] == 1
    assert out["generation_errors"] == 1
    for key in (
        "retrieval_hit_rate_at_6_pct",
        "retrieval_hit_rate_at_6_pct_n",
        "mrr",
        "answer_correct_pct_of_answered",
        "answer_correct_pct_of_answerable",
        "fully_faithful_pct",
        "made_up_rate_pct",
        "citation_correct_pct",
        "refusal_recall_pct",
        "false_refusal_rate_pct",
        "adversarial_pass_rate_pct",
        "latency_p50_ms",
        "latency_p95_ms",
        "cost_per_answer_gbp",
        "judge_errors_excluded",
    ):
        assert key in out


def test_old_question_format_is_tolerated() -> None:
    q = m.normalise_question({"question_id": "q1", "question": "x", "topic": "t", "split": "dev"})
    assert q["must_refuse"] is False
    assert q["gold_urls"] == []
    records = m.build_records([{"question_id": "q1", "refused": False}], {"q1": q}, {})
    out = m.summarise(records)
    assert out["retrieval_hit_rate_at_6_pct"] is None
    assert out["false_refusal_rate_pct"] == 0.0


def test_build_records_attaches_judge_and_error() -> None:
    rows = [{"question_id": "a"}, {"question_id": "b"}]
    judgements = {"a": {"judge_error": False, "made_up": True}, "b": {"judge_error": True}}
    records = m.build_records(rows, {}, judgements)
    assert records[0]["judge"]["made_up"] is True and records[0]["judge_error"] is False
    assert records[1]["judge"] is None and records[1]["judge_error"] is True
