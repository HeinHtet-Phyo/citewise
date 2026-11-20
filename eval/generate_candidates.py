"""Generate RAG answer candidates for all eval questions.

Usage:
    python eval/generate_candidates.py --split dev --config hybrid_400
    python eval/generate_candidates.py --split all --config dense_400

Writes one JSONL file per (split, config) to eval/results/:
    eval/results/<config>_<split>.jsonl

Each line is a serialised Answer from citewise.models plus question_id, topic,
context_urls (ordered URLs of the chunks given to the LLM) and context_texts
(the full text of those chunks, same order).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from citewise.config import RagConfig
from citewise.ingest.chunk import load_chunks
from citewise.models import Answer, Chunk
from citewise.rag import answer_question
from citewise.retrieve import retrieve

log = logging.getLogger(__name__)

QUESTIONS_DIR = Path(__file__).parent / "questions"
RESULTS_DIR = Path(__file__).parent / "results"

# Named experiment configs — tune only on dev split.
CONFIGS: dict[str, RagConfig] = {
    "dense_400": RagConfig(
        config_name="dense_400",
        chunk_size=400,
        collection_name="citewise_400",
        search_mode="dense",
        top_k_retrieve=30,
        use_reranker=False,
        top_k_context=6,
    ),
    "hybrid_400": RagConfig(
        config_name="hybrid_400",
        chunk_size=400,
        collection_name="citewise_400",
        search_mode="hybrid",
        top_k_retrieve=30,
        use_reranker=False,
        top_k_context=6,
    ),
    "hybrid_400_rerank": RagConfig(
        config_name="hybrid_400_rerank",
        chunk_size=400,
        collection_name="citewise_400",
        search_mode="hybrid",
        top_k_retrieve=30,
        use_reranker=True,
        top_k_context=6,
        refusal_threshold=0.3,  # tuned on dev split (19 Nov 2026)
    ),
    "hybrid_800": RagConfig(
        config_name="hybrid_800",
        chunk_size=800,
        collection_name="citewise_800",
        search_mode="hybrid",
        top_k_retrieve=30,
        use_reranker=False,
        top_k_context=6,
    ),
    "hybrid_800_rerank": RagConfig(
        config_name="hybrid_800_rerank",
        chunk_size=800,
        collection_name="citewise_800",
        search_mode="hybrid",
        top_k_retrieve=30,
        use_reranker=True,
        top_k_context=6,
    ),
}


_CHUNK_MAPS: dict[int, dict[str, Chunk]] = {}


def _chunk_map(chunk_size: int) -> dict[str, Chunk]:
    """Load chunk_id -> Chunk for one chunk size once and cache it."""
    if chunk_size not in _CHUNK_MAPS:
        _CHUNK_MAPS[chunk_size] = {c.chunk_id: c for c in load_chunks(chunk_size)}
    return _CHUNK_MAPS[chunk_size]


def context_for_row(row: dict, config: RagConfig) -> tuple[list[str], list[str]]:
    """Return (urls, texts) of the context chunks in rank order for one result row.

    Uses retrieved_chunk_ids (the post-rerank, post-dedup context). Rows refused before
    generation carry no ids, so retrieval is re-run (no LLM call) to measure retrieval.
    """
    ids: list[str] = row.get("retrieved_chunk_ids") or []
    if ids:
        by_id = _chunk_map(config.chunk_size)
        chunks = [by_id[i] for i in ids if i in by_id]
        return [c.url for c in chunks], [c.text for c in chunks]
    if row.get("refusal_reason") in ("unsafe", "generation_error"):
        return [], []
    try:
        retrieved = retrieve(row["question"], config)
    except Exception as exc:
        log.warning("Retrieval for context failed for %s: %s", row.get("question_id"), exc)
        return [], []
    return [rc.chunk.url for rc in retrieved], [rc.chunk.text for rc in retrieved]


def _load_questions(split: str) -> list[dict]:
    """Load questions for the given split ('dev', 'test', or 'all')."""
    questions: list[dict] = []
    for path in sorted(QUESTIONS_DIR.glob("questions_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if split == "all" or q.get("split") == split:
                questions.append(q)
    return questions


def run(split: str, config_name: str) -> None:
    """Run the pipeline and write results to eval/results/."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(CONFIGS)}")

    config = CONFIGS[config_name]
    questions = _load_questions(split)
    log.info("Running %d questions with config=%s", len(questions), config_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{config_name}_{split}.jsonl"

    errors = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(questions, start=1):
            try:
                answer: Answer = answer_question(q["question"], config)
            except Exception as exc:
                log.error(
                    "[%d/%d] %s — LLM call failed, writing error placeholder: %s",
                    i,
                    len(questions),
                    q["question_id"],
                    exc,
                )
                errors += 1
                # Write a minimal error row so the question isn't silently skipped.
                row: dict = {
                    "question": q["question"],
                    "question_id": q["question_id"],
                    "topic": q["topic"],
                    "answer": f"[generation error: {type(exc).__name__}]",
                    "citations": [],
                    "refused": True,
                    "refusal_reason": "generation_error",
                    "config_name": config_name,
                    "latency_ms": 0,
                    "cost_gbp": 0.0,
                    "retrieved_chunk_ids": [],
                    "context_urls": [],
                    "context_texts": [],
                    "trace_id": None,
                }
                fh.write(json.dumps(row) + "\n")
                continue

            log.info(
                "[%d/%d] %s — %dms (refused=%s)",
                i,
                len(questions),
                q["question_id"],
                answer.latency_ms,
                answer.refused,
            )
            row = answer.model_dump()
            row["question_id"] = q["question_id"]
            row["topic"] = q["topic"]
            row["context_urls"], row["context_texts"] = context_for_row(row, config)
            fh.write(json.dumps(row) + "\n")

    if errors:
        log.warning(
            "Completed with %d/%d generation errors → %s", errors, len(questions), output_path
        )
    else:
        log.info("Wrote %d answers → %s", len(questions), output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Generate RAG answer candidates")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="dev",
        help="Question split to run (default: dev)",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS),
        default="hybrid_400_rerank",
        help="Experiment config name (default: hybrid_400_rerank)",
    )
    args = parser.parse_args()
    run(args.split, args.config)


if __name__ == "__main__":
    main()
