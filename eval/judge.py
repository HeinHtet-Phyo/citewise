"""LLM-as-judge: score RAG answers on faithfulness, relevance, and citation quality.

The judge uses DeepSeek-V4.1-Flash (different family from the Qwen3 generator) to
avoid self-preference bias. Applies the same file-based LLM cache used by the RAG
pipeline — re-scoring the same answer costs nothing after the first run.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import litellm

from citewise.config import settings
from citewise.generate import _cache_path, _prompt_hash

log = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "src" / "citewise" / "prompts"
_JUDGE_SYSTEM = (_PROMPTS_DIR / "judge_system.txt").read_text(encoding="utf-8").strip()

# DeepSeek-V4.1-Flash — different family from the Qwen3 generator to avoid self-preference bias.
_JUDGE_MODEL = "openai/deepseek-ai/DeepSeek-V4.1-Flash-1017"
litellm.suppress_debug_info = True


@dataclass
class Judgement:
    """Scores from the LLM judge for one question-answer pair."""

    question_id: str
    faithfulness: int  # 0–3
    relevance: int  # 0–3
    citation_quality: int  # 0–3
    rationale: str
    refused: bool


def _build_judge_message(question: str, sources: list[str], answer: str) -> str:
    """Format question, sources, and answer for the judge."""
    numbered = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources))
    return f"Question: {question}\n\nSources:\n{numbered}\n\nAnswer: {answer}"


def _parse_judge_response(text: str) -> dict[str, int | str]:
    """Extract JSON from judge response, tolerating minor formatting issues."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in judge response: {text!r}")
    return json.loads(match.group())


def judge_answer(
    question_id: str,
    question: str,
    sources: list[str],
    answer: str,
    refused: bool,
) -> Judgement:
    """Score one question-answer pair using Nebius Llama-3.1-70B as judge.

    Returns a Judgement with per-dimension scores and a brief rationale.
    Uses the same LLM cache as the RAG pipeline.
    """
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _build_judge_message(question, sources, answer)},
    ]
    cache_key = _prompt_hash(_JUDGE_MODEL, messages)

    if settings.llm_cache_enabled:
        cached = _cache_path(settings.llm_cache_dir, cache_key)
        if cached.exists():
            log.debug("Judge cache hit: %s", cache_key[:12])
            data = json.loads(cached.read_text(encoding="utf-8"))
            parsed = data.get("text", data)
            if isinstance(parsed, str):
                parsed = _parse_judge_response(parsed)
        else:
            parsed = None
    else:
        parsed = None

    if parsed is None or not isinstance(parsed, dict):
        response = litellm.completion(
            model=_JUDGE_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            base_url=settings.nebius_base_url or None,
            api_key=settings.nebius_api_key or None,
        )
        text: str = response.choices[0].message.content or ""
        parsed = _parse_judge_response(text)

        if settings.llm_cache_enabled:
            cache_dir = Path(settings.llm_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            _cache_path(settings.llm_cache_dir, cache_key).write_text(
                json.dumps({"text": text}),
                encoding="utf-8",
            )

    return Judgement(
        question_id=question_id,
        faithfulness=int(parsed.get("faithfulness", 0)),
        relevance=int(parsed.get("relevance", 0)),
        citation_quality=int(parsed.get("citation_quality", 0)),
        rationale=str(parsed.get("rationale", "")),
        refused=refused,
    )


def score_results_file(results_path: Path) -> list[Judgement]:
    """Load a generate_candidates results file and judge every answer."""
    judgements: list[Judgement] = []
    lines = results_path.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        question_id = row.get("question_id", f"q{i:03d}")
        question = row["question"]
        answer = row["answer"]
        refused = bool(row.get("refused", False))
        citations = row.get("citations", [])
        sources = [c["snippet"] for c in citations] if citations else []

        log.info("Judging %s", question_id)
        j = judge_answer(
            question_id=question_id,
            question=question,
            sources=sources,
            answer=answer,
            refused=refused,
        )
        judgements.append(j)

    return judgements
