"""Run experiment configs, judge answers, compute metrics and write summaries.

Usage:
    python eval/experiments.py --split dev
    python eval/experiments.py --split test --configs hybrid_400_rerank

Workflow per config:
1. Generate candidate answers if eval/results/<config>_<split>.jsonl is missing.
2. Judge each non-refused answer (cached) and save judgements_<config>_<split>.jsonl.
3. Compute metrics (eval/metrics.py).
Then write summary_<split>.json, summary_<split>.csv and summary.md. On dev only, tune the
refusal threshold for the best config and write threshold_<config>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_candidates import CONFIGS, _chunk_map, _load_questions
from generate_candidates import run as generate_run
from judge import judge_record
from metrics import build_records, read_jsonl, summarise
from report import write_csv, write_markdown

from citewise.config import RagConfig
from citewise.retrieve import retrieve

RESULTS_DIR = Path(__file__).parent / "results"
# Tuning constraint: choose the threshold with the best correctness while the false refusal
# rate stays at or below this percentage (a project choice, tuned on dev only).
MAX_FALSE_REFUSAL_PCT = 10.0
log = logging.getLogger(__name__)


def _has_generation_errors(rows: list[dict[str, Any]]) -> bool:
    """True if any row records a failed generation (API was down on the last run)."""
    return any(r.get("refusal_reason") == "generation_error" for r in rows)


def _ensure_results(config_name: str, split: str) -> Path:
    """Generate the results file if it is missing, empty or contains generation errors."""
    path = RESULTS_DIR / f"{config_name}_{split}.jsonl"
    if path.exists() and path.stat().st_size > 0 and not _has_generation_errors(read_jsonl(path)):
        log.info("Reusing existing results: %s", path.name)
        return path
    path.unlink(missing_ok=True)
    log.info("Generating candidates for config=%s split=%s", config_name, split)
    generate_run(split=split, config_name=config_name)
    return path


def _backfill_context(row: dict[str, Any], config: RagConfig) -> dict[str, Any]:
    """Add context_urls/context_texts to rows written before those fields existed."""
    if "context_urls" in row:
        return row
    by_id = _chunk_map(config.chunk_size)
    chunks = [by_id[i] for i in row.get("retrieved_chunk_ids") or [] if i in by_id]
    return {
        **row,
        "context_urls": [c.url for c in chunks],
        "context_texts": [c.text for c in chunks],
    }


def _judge_rows(rows: list[dict[str, Any]], questions: dict[str, Any]) -> dict[str, Any]:
    """Judge every non-refused row and return verdicts keyed by question_id."""
    judgements: dict[str, Any] = {}
    for row in rows:
        qid = row.get("question_id", "")
        log.info("Judging %s", qid)
        verdict = judge_record(row, questions.get(qid, {}))
        if verdict is not None:
            judgements[qid] = verdict
    return judgements


def evaluate_config(config_name: str, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate (if needed), judge and summarise one config; return (records, summary)."""
    config = CONFIGS[config_name]
    path = _ensure_results(config_name, split)
    rows = [_backfill_context(r, config) for r in read_jsonl(path)]
    questions = {q["question_id"]: q for q in _load_questions(split)}
    judgements = _judge_rows(rows, questions)
    out = RESULTS_DIR / f"judgements_{config_name}_{split}.jsonl"
    out.write_text("".join(json.dumps(j) + "\n" for j in judgements.values()), encoding="utf-8")
    records = build_records(rows, questions, judgements)
    return records, summarise(records)


def _top_scores(records: list[dict[str, Any]], config: RagConfig) -> list[float]:
    """Retrieval-only top-chunk score per question (no LLM call)."""
    scores: list[float] = []
    for r in records:
        try:
            chunks = retrieve(r["question"], config)
            scores.append(chunks[0].score if chunks else 0.0)
        except Exception as exc:
            log.warning("Retrieval failed for %s: %s", r.get("question_id"), exc)
            scores.append(0.0)
    return scores


def _simulate_threshold(
    records: list[dict[str, Any]], scores: list[float], threshold: float
) -> list[dict[str, Any]]:
    """Mark answered records as refused when their top score falls below the threshold."""
    return [
        {**r, "refused": True} if not r.get("refused") and s < threshold else r
        for r, s in zip(records, scores, strict=True)
    ]


def _sweep_row(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for one simulated threshold."""
    m = summarise(records)
    return {
        "refusal_recall": m["refusal_recall_pct"],
        "refusal_recall_n": m["refusal_recall_pct_n"],
        "false_refusal_rate": m["false_refusal_rate_pct"],
        "answered_correct_pct": m["answer_correct_pct_of_answered"],
        "answered_correct_n": m["answer_correct_pct_of_answered_n"],
        "correct_pct_of_answerable": m["answer_correct_pct_of_answerable"],
        "n_refused": sum(1 for r in records if r.get("refused")),
    }


def choose_threshold(sweep: dict[str, dict[str, Any]]) -> str | None:
    """Best correct%-of-answerable with false refusal within the cap; ties favour recall."""
    usable = {t: s for t, s in sweep.items() if s["correct_pct_of_answerable"] is not None}
    if not usable:
        return None
    ok = {
        t: s for t, s in usable.items() if (s["false_refusal_rate"] or 0) <= MAX_FALSE_REFUSAL_PCT
    }
    pool = ok or usable
    return max(
        pool,
        key=lambda t: (
            pool[t]["correct_pct_of_answerable"] if ok else -(pool[t]["false_refusal_rate"] or 0),
            pool[t]["refusal_recall"] or 0,
        ),
    )


def tune_threshold(config_name: str, records: list[dict[str, Any]]) -> float | None:
    """Sweep refusal thresholds on dev, write threshold_<config>.json, return the best.

    Only thresholds at or above the one the results were generated with can be simulated,
    because lower thresholds would need answers that were never generated.
    """
    config = CONFIGS[config_name]
    scores = _top_scores(records, config)
    start = round(config.refusal_threshold / 0.05)
    sweep: dict[str, dict[str, Any]] = {}
    for step in range(start, 21):
        t = round(step * 0.05, 2)
        sweep[str(t)] = _sweep_row(_simulate_threshold(records, scores, t))
    best = choose_threshold(sweep)
    output = {
        "config": config_name,
        "generated_at_threshold": config.refusal_threshold,
        "max_false_refusal_pct": MAX_FALSE_REFUSAL_PCT,
        "best_threshold": float(best) if best is not None else None,
        "best": sweep.get(best) if best is not None else None,
        "sweep": sweep,
    }
    (RESULTS_DIR / f"threshold_{config_name}.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return float(best) if best is not None else None


def run_experiments(split: str, config_names: list[str]) -> None:
    """Evaluate each config, write summaries, and tune the threshold on dev."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}
    all_records: dict[str, list[dict[str, Any]]] = {}
    for name in config_names:
        all_records[name], summary[name] = evaluate_config(name, split)

    (RESULTS_DIR / f"summary_{split}.json").write_text(json.dumps(summary, indent=2), "utf-8")
    write_csv(summary, RESULTS_DIR / f"summary_{split}.csv")
    md_path = write_markdown(RESULTS_DIR)
    print(md_path.read_text(encoding="utf-8"))

    if split != "dev":
        return
    scored = {c: m["answer_correct_pct_of_answerable"] for c, m in summary.items()}
    scored = {c: v for c, v in scored.items() if v is not None}
    if not scored:
        log.warning("No reference-answer correctness available — skipping threshold tuning")
        return
    best_config = max(scored, key=lambda c: scored[c])
    best = tune_threshold(best_config, all_records[best_config])
    print(f"\nRecommended refusal_threshold for {best_config}: {best}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    parser = argparse.ArgumentParser(description="Run eval experiments")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--configs", default=",".join(CONFIGS.keys()))
    args = parser.parse_args()
    run_experiments(args.split, [c.strip() for c in args.configs.split(",")])


if __name__ == "__main__":
    main()
