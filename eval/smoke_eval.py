"""Smoke eval: re-judge 10 committed dev answers with the real judge and compute metrics.

Needs only NEBIUS_API_KEY (no Qdrant). It checks that the judge call, JSON parsing and metric
code still work end to end. It does not re-measure quality, so it prints numbers but asserts
only that at least 8 of the 10 verdicts parsed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_candidates import _load_questions
from judge import judge_record
from metrics import build_records, read_jsonl, summarise

RESULTS = Path(__file__).parent / "results"
N = 10


def main() -> None:
    """Judge the first N answered dev rows and print the summary."""
    rows = [r for r in read_jsonl(RESULTS / "L_dense_dev.jsonl") if not r["refused"]][:N]
    questions = {q["question_id"]: q for q in _load_questions("dev")}
    verdicts = {}
    for row in rows:
        v = judge_record(row, questions[row["question_id"]])
        if v is not None:
            verdicts[row["question_id"]] = v
    ok = [v for v in verdicts.values() if not v.get("judge_error")]
    print(f"judged {len(ok)}/{len(rows)} answers")
    print(summarise(build_records(rows, questions, verdicts)))
    assert len(ok) >= 8, "judge failed on more than 2 of 10 answers"


if __name__ == "__main__":
    main()
