"""Offline checks on the evaluation question files."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "eval"))

from question_schema import EvalQuestion  # noqa: E402

QDIR = Path(__file__).parents[2] / "eval" / "questions"


def _load(split: str) -> list[EvalQuestion]:
    path = QDIR / f"questions_{split}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [EvalQuestion.model_validate(json.loads(line)) for line in fh]


@pytest.fixture(scope="module")
def dev() -> list[EvalQuestion]:
    return _load("dev")


@pytest.fixture(scope="module")
def test() -> list[EvalQuestion]:
    return _load("test")


def test_split_field_matches_file(dev: list[EvalQuestion], test: list[EvalQuestion]) -> None:
    assert {q.split for q in dev} == {"dev"}
    assert {q.split for q in test} == {"test"}


def test_ids_unique_and_no_overlap(dev: list[EvalQuestion], test: list[EvalQuestion]) -> None:
    ids = [q.question_id for q in dev + test]
    assert len(ids) == len(set(ids))
    texts = [q.question.lower() for q in dev + test]
    assert len(texts) == len(set(texts))


def test_counts_roughly_as_targeted(dev: list[EvalQuestion], test: list[EvalQuestion]) -> None:
    assert 180 <= len(dev) + len(test) <= 210
    assert 55 <= len(dev) <= 75
    assert 115 <= len(test) <= 145
    kinds = Counter(q.type for q in dev + test)
    assert kinds["unanswerable"] >= 25 and kinds["adversarial"] >= 12


def test_refusal_consistency(dev: list[EvalQuestion], test: list[EvalQuestion]) -> None:
    for q in dev + test:
        assert q.must_refuse == (q.type in {"unanswerable", "adversarial"})


def test_answerable_have_ground_truth(dev: list[EvalQuestion], test: list[EvalQuestion]) -> None:
    for q in (x for x in dev + test if not x.must_refuse):
        assert q.reference_answer and q.reference_answer.strip()
        assert 1 <= len(q.gold_urls) <= 2
        assert all(u.startswith("https://www.gov.uk/") for u in q.gold_urls)


def test_both_splits_cover_all_types(dev: list[EvalQuestion], test: list[EvalQuestion]) -> None:
    for split in (dev, test):
        assert {q.type for q in split} == {
            "single_fact", "multi_page", "how_to", "unanswerable", "adversarial",
        }  # fmt: skip


def test_hand_check_csv() -> None:
    with (QDIR / "hand_check_questions.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 30
    assert list(rows[0])[-2:] == ["ai_check", "ai_check_reason"]
