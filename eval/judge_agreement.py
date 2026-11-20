"""Judge-judge agreement: DeepSeek judge vs a third-family model (Gemma) on the same answers.

This replaces judge-human agreement: no human labelled any answer. It measures whether two
different model families give the same verdicts, not whether either is correct.

Usage:
    python eval/judge_agreement.py --config hybrid_400_rerank --n 30
Writes eval/results/judge_agreement.json and eval/results/hand_check_answers.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import judge
import litellm
from generate_candidates import _load_questions
from metrics import read_jsonl

from citewise.config import settings
from citewise.generate import _cache_path, _prompt_hash

RESULTS_DIR = Path(__file__).parent / "results"
SECOND_MODEL = "openai/google/gemma-3-27b-it"
SEED = 20261120
FIELDS = ("correctness", "fully_faithful", "made_up")


def cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    """Cohen's kappa for two equal-length label lists; None when chance agreement is 1."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 3)


def second_verdict(row: dict[str, Any], question: dict[str, Any]) -> judge.JudgeVerdict | None:
    """Judge one answer with the second model, using the same prompt as the DeepSeek judge."""
    user = judge.build_judge_message(
        row["question"],
        question.get("type"),
        question.get("reference_answer"),
        row["context_texts"],
        row["answer"],
    )
    messages = [{"role": "user", "content": f"{judge._JUDGE_SYSTEM}\n\n{user}"}]
    key = _prompt_hash(f"{SECOND_MODEL}|{judge.PROMPT_VERSION}", messages)
    path = _cache_path(settings.llm_cache_dir, key)
    if settings.llm_cache_enabled and path.exists():
        text = json.loads(path.read_text(encoding="utf-8"))["text"]
    else:
        resp = litellm.completion(
            model=SECOND_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
            timeout=90,
        )
        text = resp.choices[0].message.content or ""
        Path(settings.llm_cache_dir).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}), encoding="utf-8")
    try:
        return judge.parse_verdict(text)
    except ValueError:
        return None


def run(config: str, n: int) -> dict[str, Any]:
    """Sample n answered test rows, get both verdicts, and compute agreement."""
    rows = {r["question_id"]: r for r in read_jsonl(RESULTS_DIR / f"{config}_test.jsonl")}
    verdicts = {
        v["question_id"]: v
        for v in read_jsonl(RESULTS_DIR / f"judgements_{config}_test.jsonl")
        if not v.get("judge_error")
    }
    questions = {q["question_id"]: q for q in _load_questions("test")}
    ids = sorted(verdicts)
    random.Random(SEED).shuffle(ids)
    pairs: list[tuple[str, dict[str, Any], judge.JudgeVerdict]] = []
    for qid in ids:
        if len(pairs) == n:
            break
        other = second_verdict(rows[qid], questions[qid])
        if other is not None:
            pairs.append((qid, verdicts[qid], other))
    summary: dict[str, Any] = {"config": config, "second_model": SECOND_MODEL, "n": len(pairs)}
    for f in FIELDS:
        a = [p[1][f] for p in pairs]
        b = [getattr(p[2], f) for p in pairs]
        summary[f] = {
            "agreement_pct": round(
                100 * sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 1
            ),
            "kappa": cohen_kappa(a, b),
        }
    (RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(summary, indent=2), "utf-8")
    with (RESULTS_DIR / "hand_check_answers.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["question_id", "question", "answer", *[f"deepseek_{f}" for f in FIELDS],
                    *[f"gemma_{f}" for f in FIELDS], "all_agree"])  # fmt: skip
        for qid, d, g in pairs:
            dv = [d[f] for f in FIELDS]
            gv = [getattr(g, f) for f in FIELDS]
            w.writerow([qid, rows[qid]["question"], rows[qid]["answer"], *dv, *gv, dv == gv])
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hybrid_400_rerank")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.n), indent=2))


if __name__ == "__main__":
    main()
