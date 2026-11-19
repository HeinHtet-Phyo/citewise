"""Generate hand-check CSVs for human review.

Samples 30 question-answer pairs and 30 citation pairs from the best
config's dev-split results for manual quality inspection.

Usage:
    python eval/hand_check.py --config hybrid_400_rerank

Writes:
    eval/hand_check_answers.csv   — 30 questions with full answers
    eval/hand_check_citations.csv — 30 citation (snippet, url) pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent


def _load_results(config_name: str, split: str = "dev") -> list[dict]:
    path = RESULTS_DIR / f"{config_name}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}\nRun generate_candidates.py first."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_answer_csv(rows: list[dict], out_path: Path) -> None:
    """Write 30 sampled question-answer rows for human review."""
    answered = [r for r in rows if not r.get("refused", False)]
    sample = random.sample(answered, min(30, len(answered)))
    sample.sort(key=lambda r: r.get("question_id", ""))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "question_id",
                "topic",
                "question",
                "answer",
                "n_citations",
                "latency_ms",
                "cost_gbp",
                "human_ok",
            ],
        )
        writer.writeheader()
        for r in sample:
            writer.writerow(
                {
                    "question_id": r.get("question_id", ""),
                    "topic": r.get("topic", ""),
                    "question": r["question"],
                    "answer": r["answer"].replace("\n", " "),
                    "n_citations": len(r.get("citations", [])),
                    "latency_ms": r.get("latency_ms", 0),
                    "cost_gbp": r.get("cost_gbp", 0.0),
                    "human_ok": "",  # reviewer fills in: Y / N / ?
                }
            )
    print(f"Wrote {len(sample)} rows → {out_path}")


def write_citation_csv(rows: list[dict], out_path: Path) -> None:
    """Write 30 sampled citations for source-quality review."""
    all_citations = []
    for r in rows:
        for c in r.get("citations", []):
            all_citations.append(
                {
                    "question_id": r.get("question_id", ""),
                    "question": r["question"],
                    "citation_n": c["n"],
                    "title": c["title"],
                    "section": c.get("section", ""),
                    "url": c["url"],
                    "snippet": c["snippet"],
                    "updated_at": c["updated_at"],
                    "human_relevant": "",  # reviewer fills in: Y / N / ?
                }
            )

    sample = random.sample(all_citations, min(30, len(all_citations)))
    sample.sort(key=lambda c: (c["question_id"], c["citation_n"]))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "question_id",
                "question",
                "citation_n",
                "title",
                "section",
                "url",
                "snippet",
                "updated_at",
                "human_relevant",
            ],
        )
        writer.writeheader()
        for c in sample:
            writer.writerow(c)
    print(f"Wrote {len(sample)} rows → {out_path}")


def main() -> None:
    random.seed(42)  # reproducible sampling
    parser = argparse.ArgumentParser(description="Generate hand-check CSVs")
    parser.add_argument("--config", default="hybrid_400_rerank")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    rows = _load_results(args.config, args.split)
    print(f"Loaded {len(rows)} results from {args.config}_{args.split}.jsonl")

    write_answer_csv(rows, OUTPUT_DIR / "hand_check_answers.csv")
    write_citation_csv(rows, OUTPUT_DIR / "hand_check_citations.csv")
    print("\nOpen in a spreadsheet, fill in the human_ok / human_relevant columns.")


if __name__ == "__main__":
    main()
