"""Pydantic schema for evaluation questions (shared contract with eval/metrics.py)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

QuestionType = Literal["single_fact", "multi_page", "how_to", "unanswerable", "adversarial"]
REFUSAL_TYPES = {"unanswerable", "adversarial"}


class EvalQuestion(BaseModel):
    """One evaluation question with ground truth."""

    question_id: str
    question: str
    topic: Literal["benefits", "employment", "housing", "transport"]
    split: Literal["dev", "test"]
    type: QuestionType
    must_refuse: bool
    reference_answer: str | None
    gold_urls: list[str]
    notes: str | None

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalQuestion:
        """Enforce the refusal / answerable invariants."""
        if self.must_refuse != (self.type in REFUSAL_TYPES):
            raise ValueError("must_refuse must be true exactly for unanswerable/adversarial")
        if self.must_refuse:
            if self.reference_answer is not None or self.gold_urls or not self.notes:
                raise ValueError("refusal questions need null answer, no gold URLs, and notes")
        else:
            if not self.reference_answer or not 1 <= len(self.gold_urls) <= 2:
                raise ValueError("answerable questions need an answer and 1-2 gold URLs")
        return self
