"""Write eval/questions/README.md with counts computed from the final report."""

from __future__ import annotations

from pathlib import Path


def _table(counts: dict[str, int]) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in counts.items())
    return f"| Group | Questions |\n|---|---|\n{rows}\n"


def write_readme(path: Path, report: dict[str, object]) -> None:
    """Render the README from the verification report (all numbers come from code)."""
    counts: dict[str, dict[str, int]] = report["counts"]  # type: ignore[assignment]
    drops: dict[str, int] = report["dropped_by_reason"]  # type: ignore[assignment]
    drop_lines = "\n".join(f"- `{k}`: {v}" for k, v in drops.items())
    text = f"""# Evaluation questions

{report["total"]} questions with ground truth, built by `eval/build_questions.py`
(seed {report["seed"]}). Schema: `eval/question_schema.py`.

## How they were made

1. The original {report["original_questions"]} questions (archived in `original_questions.jsonl`)
   were each matched to candidate GOV.UK passages by BM25 over `data/processed/chunks_400.jsonl`.
2. `{report["model"]}` (Nebius) drafted a 1-3 sentence reference answer, a question type and
   verbatim evidence quotes with URLs. {report["new_kept"]} further answerable questions were
   drafted the same way from seed passages sampled with the fixed seed.
3. Every draft was machine-checked: each evidence quote must appear verbatim (after whitespace
   and punctuation normalisation) in the text of its gold page, the gold URL must exist in the
   corpus, and every figure in the reference answer must appear in the gold page text.
   Drafts that failed were dropped, not repaired. {report["existing_kept"]} of the original
   questions survived.
4. {report["refusals_added"]} unanswerable and adversarial questions were written by hand
   (`build_questions_refusals.py`), each with a note giving the reason for refusal.
5. Questions were split into dev and test, stratified by type and topic, with the fixed seed.

Dropped by verification:

{drop_lines}

## Counts (computed by code)

### By split
{_table(counts["split"])}
### By type
{_table(counts["type"])}
### By topic
{_table(counts["topic"])}
### Split by type
{_table(counts["split_x_type"])}
### Split by topic
{_table(counts["split_x_topic"])}
## Dev/test rule

Dev and test do not overlap. Tune thresholds and configurations on dev only; report final
numbers from a single test run. Never choose anything based on test results.

## Limitations

- Reference answers are LLM-drafted. The machine check confirms the quoted evidence and the
  figures exist in the gold page, not that the answer is complete or perfectly phrased.
- Gold URLs list the page(s) the evidence was taken from; other pages may also answer the question.
- The "unanswerable" questions were chosen by hand and only
  spot-checked against the corpus.
- A sample of 30 answerable questions is in `hand_check_questions.csv` for the owner to review
  (`human_ok` is empty until that is done; agreement is [TBC]).
- Nebius spend for this build: ${report["nebius_spend_usd"]} over {report["nebius_calls"]} calls.
"""
    path.write_text(text, encoding="utf-8")
